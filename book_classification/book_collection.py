import random

class BookCollection:
	def __init__(self, books):
		self.books = set(books)

		self.authors = {}
		for b in self.books:
			if b.author() not in self.authors:
				self.authors[b.author()] = set()
			self.authors[b.author()].add(b)

	def __len__(self):
		return len(self.books)
	def __iter__(self):
		return iter(self.books)

	def filter(self, condition):
		return BookCollection(filter(condition, self.books))

	def partition(self, condition):
		books_one = set()
		books_two = set()

		for b in self.books:
			if condition(b):
				books_one.add(b)
			else:
				books_two.add(b)

		return BookCollection(books_one), BookCollection(books_two)

	def sample_authors(self, n):
		# FIXME: make deterministic while allowing to change seed
		random.seed(123)
		authors = set(random.sample(sorted(self.authors.keys()), n))
		def condition(book):
			return book.author() in authors
		return self.filter(condition)

	def only_authors_with_or_more_than(self, n):
		def condition(book):
			return len(self.authors[book.author()]) >= n
		return self.filter(condition)

	def separate_by_at_most_per_author(self, n):
		selected = {}
		def condition(book):
			if book.author() not in selected:
				selected[book.author()] = 0
			if selected[book.author()] < n:
				selected[book.author()] += 1
				return True
			else:
				return False

		return self.partition(condition)

	@classmethod
	def from_books(self, books):
		pass

	@classmethod
	def from_file_path_list(self, path_list):
		for path in path_list:
			pass