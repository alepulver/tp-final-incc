from collections import defaultdict, Counter
import book_classification as bc
import random
import pandas
import numpy


class BookCollection:
    def __init__(self, books, books_by_author):
        self._books = books
        self._books_by_author = books_by_author

    def __len__(self):
        return len(self._books)

    def books(self):
        return self._books

    def books_by(self, author):
        return self._books_by_author[author]

    def authors(self):
        return self._books_by_author.keys()

    def fold(self, func_author, func_book, base_author, base_book):
        final_result = base_author
        for author in self.authors():
            partial_result = base_book
            for book in self.books_by(author):
                partial_result = func_book(book, partial_result)
            final_result = func_author(author, partial_result, final_result)
        return final_result

    def as_dataframe(self):
        result = []
        for book in self.books():
            result.append([book.title(), book.author(), len(book.contents()), book])
        return pandas.DataFrame(result, columns=["Title", "Author", "Size", "Object"])

    def as_arrays(self):
        books = numpy.array(self.books())
        authors = numpy.array([book.author() for book in self.books()])
        return books, authors

    def selection(self):
        return BookCollectionSelection(self)

    @classmethod
    def from_books(cls, books):
        # XXX: book order need to be deterministic, if list of books is used to construct another collection
        _books = list(books)
        _books_by_author = defaultdict(set)
        for b in _books:
            _books_by_author[b.author()].add(b)
        return cls(_books, _books_by_author)

    @classmethod
    def from_file_path_list(cls, path_list):
        books = [bc.Book.from_file_path(path) for path in path_list]
        return cls.from_books(books)

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls.from_books(dataframe['Object'])


class BookCollectionAnalysis:
    def __init__(self, book_collection, tokenizer):
        self._book_collection = book_collection
        self._tokenizer = tokenizer
        extractor = bc.VocabulariesExtractor(self._tokenizer)
        self._vocabulary = bc.CollectionHierarchialFeatures.from_book_collection(
            self._book_collection, extractor)

    def vocabulary(self):
        return self._vocabulary

    def shared_words_by_books(self):
        books_for_word = Counter()
        for book in self._book_collection.books():
            for word in self._vocabulary.by_book(book).keys():
                books_for_word[word] += 1

        word_count_by_n = Counter()
        for word, count in books_for_word.items():
            word_count_by_n[count] += 1

        return word_count_by_n

    def shared_words_by_authors(self):
        authors_for_word = Counter()
        for author in self._book_collection.authors():
            for word in self._vocabulary.by_author(author).keys():
                authors_for_word[word] += 1

        word_count_by_n = Counter()
        for word, count in authors_for_word.items():
            word_count_by_n[count] += 1

        return word_count_by_n

    def vocabulary_size_by_book(self):
        result = []
        for book in self._book_collection.books():
            unique_words = len(self._vocabulary.by_book(book))
            result.append([book, unique_words])
        return pandas.DataFrame(result, columns=["Book", "Unique words"])

    def vocabulary_size_by_author(self):
        result = []
        for author in self._book_collection.authors():
            unique_words = len(self._vocabulary.by_author(author))
            result.append([author, unique_words])
        return pandas.DataFrame(result, columns=["Author", "Unique words"])


class BookCollectionSelection:
    def __init__(self, book_collection):
        self._book_collection = book_collection

    def find_duplicates(self):
        dataframe = self._book_collection.as_dataframe()
        duplicates = dataframe[dataframe.duplicated('Title')]
        return BookCollection.from_dataframe(duplicates)

    def remove_duplicates(self):
        dataframe = self._book_collection.as_dataframe()
        return BookCollection.from_dataframe(dataframe.drop_duplicates('Title'))

    def filter_authors(self, condition):
        result = []
        for author in self._book_collection.authors():
            for book in self._book_collection.books_by(author):
                if condition(author):
                    result.append(book)
        return BookCollection.from_books(result)

    def filter_books(self, condition):
        return BookCollection.from_books(
            filter(condition, self._book_collection.books()))

    def partition_books(self, condition):
        books_one = set()
        books_two = set()

        for b in self._book_collection.books():
            if condition(b):
                books_one.add(b)
            else:
                books_two.add(b)

        c1 = BookCollection.from_books(books_one)
        c2 = BookCollection.from_books(books_two)
        return c1, c2

    def exclude_authors_below(self, n):
        def condition(author):
            return len(self._book_collection.books_by(author)) >= n
        return self.filter_authors(condition)

    def exclude_authors_above(self, n):
        def condition(author):
            return len(self._book_collection.books_by(author)) <= n
        return self.filter_authors(condition)

    def split_per_author_number(self, n):
        assert(n > 0)

        author_sizes = {}
        for author in self._book_collection.authors():
            author_sizes[author] = n

        return self.split_per_author_with_sizes(author_sizes)

    def split_per_author_percentage(self, percentage):
        assert(0 < percentage < 1)

        author_sizes = {}
        for author in self._book_collection.authors():
            n = len(self._book_collection.books_by(author))
            author_sizes[author] = min(n-1, round(percentage*n))

        return self.split_per_author_with_sizes(author_sizes)

    def split_per_author_with_sizes(self, quantities):
        assert(len(quantities) > 0)
        #for author, size in quantities.items():
        #    if size < 2:
        #        raise Exception("can not partition author '%s' with less than 2 books" % author)

        def condition(book):
            author = book.author()
            if quantities[author] > 0:
                quantities[author] -= 1
                return True
            else:
                return False
        return self.partition_books(condition)

    def sample_authors(self, n):
        authors = random.sample(list(self._book_collection.authors()), n)
        return self.filter_authors(lambda x: x in authors)

    def sample_books(self, n):
        books = random.sample(list(self._book_collection.books()), n)
        return self.filter_books(lambda x: x in books)

    def sample_books_per_author(self, n):
        result = []
        for author in self._book_collection.authors():
            books = self._book_collection.books_by(author)
            result.extend(random.sample(list(books), n))
        return self.filter_books(lambda x: x in result)

    def sample_authors_with_books(self, num_authors, num_books):
        seq = self._book_collection.authors()
        cond = lambda author: len(self._book_collection.books_by(author)) >= num_books
        authors = list(filter(cond, seq))

        authors = random.sample(authors, num_authors)
        books = []
        for a in authors:
            author_books = list(self._book_collection.books_by(a))
            selected_books = random.sample(author_books, num_books)
            books.extend(selected_books)

        return BookCollection.from_books(books)
