from collections import Counter, defaultdict
import book_classification as bc
import random
from functools import reduce
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
		# XXX: need to be deterministic, so that shuffling and partitioning produce the same results
		books_list = list(self.books())
		books_list.sort(key=lambda x: x.title())
		
		books = numpy.array(books_list)
		authors = numpy.array([book.author() for book in books_list])
		return books, authors

	def selection(self):
		return BookCollectionSelection(self)

	@classmethod
	def from_books(cls, books):
		_books = set(books)
		_books_by_author = defaultdict(set)
		for b in _books:
			_books_by_author[b.author()].add(b)
		return cls(_books, _books_by_author)

	@classmethod
	def from_file_path_list(cls, path_list):
		books = [bc.Book.from_file_path(path) for path in path_list]
		return cls.from_books(books)

class BookCollectionSelection:
	def __init__(self, book_collection):
		self._book_collection = book_collection

	def find_duplicates(self):
		raise NotImplementedError()

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
		for author,size in quantities.items():
			if size < 2:
				raise Exception("can not partition author '%s' with less than 2 books" % author)

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