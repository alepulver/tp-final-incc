from collections import Counter, defaultdict
import book_classification as bc
import random

class BookCollection:
	# FIXME: remove duplicates! or add method to do so

	# TODO: add functional fold, and visitor pattern for all other
	# algorithms and feature constructors (like HierarchialFeatures)

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

	def filter_authors(self, condition):
		result = set()
		for author,books in self._books_by_author.items():
			if condition(author):
				result.update(books)
		return self.__class__.from_books(result)

	def filter_books(self, condition):
		return self.__class__.from_books(filter(condition, self.books()))

	def partition_books(self, condition):
		books_one = set()
		books_two = set()

		for b in self.books():
			if condition(b):
				books_one.add(b)
			else:
				books_two.add(b)

		c1 = self.__class__.from_books(books_one)
		c2 = self.__class__.from_books(books_two)
		return c1, c2

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

	# TODO: move these methods somewhere else
	def exclude_authors_with_less_than(self, n):
		def condition(author):
			return len(self.books_by(author)) >= n
		return self.filter_authors(condition)

	def split_at_number_per_author(self, n):
		selected = Counter()
		def condition(book):
			if selected[book.author()] < n:
				selected[book.author()] += 1
				return True
			else:
				return False
		return self.partition_books(condition)

	def sample_authors(self, n):
		authors = random.sample(list(self.authors()), n)
		return self.filter_authors(lambda x: x in authors)
	def sample_books(self, n):
		books = random.sample(list(self.books()), n)
		return self.filter_books(lambda x: x in books)