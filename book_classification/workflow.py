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

	def restrict_vocabulary(self, words):
		# return new ExtractionEnvironment replacing tokenizer with tokenizer.restrict_vocabulary
		raise NotImplementedError()

	@classmethod
	def default(cls):
		tokenizer = bc.BasicTokenizer()
		grouper = bc.BasicGrouper(500)
		weighter = bc.WeightingWindow.uniform(501)
		return cls(tokenizer, grouper, weighter)

class PossibleFeatureAnalyzer:
	# TODO: allow pruning quantiles

	def __init__(self, collection, extraction_env, frequencies, entropies):
		self._collection = collection
		self._extraction_env = extraction_env
		self._frequencies = frequencies
		self._entropies = entropies

	# change filters to only remove words
	# if necessary, add a "restrict" method to Features to recalculate with only certain words
	# and make returning entropies lazy, because they can't be patched like frequencies

	def prune_less_occurrences_than(self, occurrences):
		counts = Counter()
		total = 0
		for k,v in self._counts.items():
			if v >= occurrences:
				counts[k] = v
				total += v
		return self.__class__(counts, total)

	def prune_last_words(self, n):
		pairs = [(v,k) for (k,v) in self._counts.items()]
		pairs.sort(reverse=True)

		counts = Counter()
		total = 0
		current = 0
		for v,k in pairs:
			if current >= n:
				counts[k] = v
				total += v
			current += 1
		return self.__class__(counts, total)

	def as_dataframe(self):
		words = list(self._counts.keys())
		counts = list(self._counts.values())
		frequencies = list(v/self._total for v in self._counts.values())
		return pandas.DataFrame({'Word': words, 'Count': counts, 'Frequency': frequencies})

	# graph frequencies/entropies per author vs total, each column as a series

	def environment(self):
		pass

	@classmethod
	def from_book_collection(cls, collection, extraction_env=None):
		if extraction_env is None:
			extraction_env = ExtractionEnvironment.default()

		frequencies = bc.CollectionFeatures.from_book_collection(
			collection, lambda x: extraction_env.frequencies(x))
		entropies = bc.CollectionFeatures.from_book_collection(
			collection, lambda x: extraction_env.entropies(x))

		return cls(collection, extraction_env, frequencies, entropies)

class PossibleMatrixAnalyzer:
	pass

class FeatureAggregator:
	# XXX: list of features to matrix, maybe with different indexes and type of feature

	# mutable, but returns immutable matrices
	pass