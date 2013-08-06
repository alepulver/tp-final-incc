from collections import Counter, defaultdict
import book_classification as bc
import pandas
from functools import reduce

class ExtractionEnvironment:
	def __init__(self, tokenizer, grouper, window):
		self._tokenizer = tokenizer
		self._grouper = grouper
		self._window = window

	def tokens_from(self, text):
		return self._tokenizer.tokens_from(text)
	def tokenized_groups_from(self, text):
		return self._grouper.parts_from(self.tokens_from(text))

	def frequencies(self, text):
		return bc.TokenFrequencies.from_tokens(self.tokens_from(text))
	def series(self, text):
		return bc.TokenSeries.from_tokens(self.tokens_from(text))
	def entropies(self, text):
		return bc.TokenEntropies.from_parts(self.tokenized_groups_from(text))
	def pairwise_associations(self, text):
		return bc.TokenPairwiseAssociations.from_tokens(self.tokens_from(text), self._window)

	def vocabulary(self):
		return self._tokenizer.vocabulary()
	def restrict_vocabulary(self, words):
		return self.__class__(self._tokenizer.restrict_vocabulary(words), self._grouper, self._window)

	@classmethod
	def default(cls):
		tokenizer = bc.BasicTokenizer()
		grouper = bc.BasicGrouper(500)
		weighter = bc.WeightingWindow.uniform(501)
		return cls(tokenizer, grouper, weighter)

class PossibleFeatureAnalyzer:
	def __init__(self, collection, extraction_env, frequencies, entropies):
		self._collection = collection
		self._extraction_env = extraction_env
		self._frequencies = frequencies
		self._entropies = entropies

	# TODO
	# and make returning entropies lazy, because they can't be patched like frequencies

	def prune_quantiles(self, pairs, low, high):
		assert(low < high)
		assert(0 <= low <= 1)
		assert(0 <= high <= 1)

		pairs.sort()
		current = 0
		words = set()

		for v,k in pairs:
			if current > high:
				break
			if low <= current:
				words.add(k)
			current += v

		extraction_env = self._extraction_env.restrict_vocabulary(words)
		return self.__class__.from_book_collection(self._collection, extraction_env)

	# FIXME: does not work
	#def prune_entropies_quantiles(self, low, high):
	#	pairs = [(v,k) for (k,v) in self._entropies.total().items()]
	#	return self.prune_quantiles(pairs, low, high)
	
	def prune_frequencies_quantiles(self, low, high):
		pairs = [(v,k) for (k,v) in self._frequencies.total().items()]
		return self.prune_quantiles(pairs, low, high)

	def frequencies(self):
		return self._frequencies
	def entropies(self):
		return self._entropies
	def extraction_env(self):
		return self._extraction_env
	def collection(self):
		return self._collection

	@classmethod
	def from_book_collection(cls, collection, extraction_env=None):
		if extraction_env is None:
			extraction_env = ExtractionEnvironment.default()

		frequencies = bc.HierarchialFeatures.from_book_collection(
			collection, lambda x: extraction_env.frequencies(x))
		entropies = bc.HierarchialFeatures.from_book_collection(
			collection, lambda x: extraction_env.entropies(x))
		extraction_env = extraction_env.restrict_vocabulary(frequencies.total().keys())

		return cls(collection, extraction_env, frequencies, entropies)

class PossibleVocabularyAnalyzer:
	def __init__(self, collection, tokenizer, vocabularies_by_book, vocabularies_by_author, words_presence):
		self._vocabularies_by_book = vocabularies_by_book
		self._vocabularies_by_author = vocabularies_by_author
		self._words_presence = words_presence

	def xxx(self):
		pass

	def from_book_collection(cls, collection, tokenizer):
		vocabularies_by_book = {}
		vocabularies_by_author = defaultdict(set)
		words_presence_by_book = Counter()
		words_presence_by_author = Counter()

		for book in collection.books():
			vocabularies_by_book[book] = set(tokenizer.tokens_from(book.contents()))
			vocabularies_by_author[book.author()].update(vocabularies_by_book[book])
			for word in vocabularies_by_book[book]:
				words_presence_by_book[word] += 1
		for word in vocabularies_by_author.keys():
			for word in vocabularies_by_book[book]:
				words_presence[word] += 1

		return cls(collection, tokenizer, vocabularies_by_book, vocabularies_by_author, words_presence)

class FeaturesExtractor:
	def __init__(self, extraction_env):
		self._extraction_env = extraction_env
	def extract_from(self, book):
		return self._extraction_env.entropies(book.contents())
		#entropies = self._extraction_env.entropies(book.contents())
		#frequencies = self._extraction_env.frequencies(book.contents())
		#result = {}
		#for k,v in frequencies.items():
		#	result[k] = v * entropies[k]
		#return result
	def vocabulary(self):
		return self._extraction_env.vocabulary()