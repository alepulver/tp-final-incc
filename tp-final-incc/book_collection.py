class BookCollection:
	def __init__(self, books):
		self.books = books

		self.authors = {}
		for b in self.books:
			if b.author not in self.authors:
				self.authors[b.author] = set()
			self.authors[b.author].add(b)

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

	def only_authors_with_or_more_than(self, n):
		def condition(book):
			return len(self.authors[book.author]) >= n
		return self.filter(condition)

	def separate_by_at_most_per_author(self, n):
		selected = {}
		def condition(book):
			if book.author not in selected:
				selected[book.author] = 0
			if selected[book.author] < n:
				selected[book.author] += 1
				return True
			else:
				return False

		return self.partition(condition)