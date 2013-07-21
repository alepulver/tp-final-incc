import nltk
from collections import Counter

class TextFeatures:
	def __init__(self, book):
		self.book = book

	def words(self):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(self.book.contents.lower())

		return filter(is_word, tokens)

	def features(self):
		raise NotImplementedError()

class WordFrequencies(TextFeatures):
	def features(self):
		counts = Counter()
		for w in self.words():
			counts[w] += 1
		return counts

class WordEntropies(TextFeatures):
	def features(self):
		# TODO: implement when windows and aggregation are ready
		pass