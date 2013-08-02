from collections import Counter, defaultdict
import book_classification as bc
import pandas

class ExtractionEnvironment:
	def __init__(self, indexer, tokenizer, grouper, window):
		self._indexer = indexer
		self._tokenizer = tokenizer
		self._grouper = grouper
		self._window = window
	def tokens_from(text):
		return (t for t in self._tokenizer.tokens_from(text) if self._indexer.can_encode(t))

	def frequencies(self, text):
		return bc.TokenFrequencies.from_tokens(self.tokens_from(text))
	def series(self, text):
		return bc.TokenSeries.from_tokens(self.tokens_from(text))
	def entropies(self, text):
		return bc.TokenEntropies.from_parts(self._grouper.parts_from(self.tokens_from(text)))
	def pairwise_associations(self, text):
		return bc.TokenPairwiseAssociations.from_tokens(self.tokens_from(text), self._window)

	@classmethod
	def default(cls):
		indexer = bc.NumericIndexer()
		tokenizer = bc.BasicTokenizer()
		splitter = bc.BasicSplitter()
		weighter = bc.WeightingWindow()
		return cls(indexer, tokenizer, splitter, weighter)

class FeatureAggregator:
	# XXX: list of features to matrix, maybe with different indexes and type of feature
	pass

class PossibleFeatureAnalyzer:
	# allow pruning quantiles

	# also calculate entropies, everything for each book and total

	def __init__(self, counts, total):
		self._counts = counts
		self._total = total

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

	def build_indexer(self):
		return bc.NumericIndexer(self._counts.keys())

	@classmethod
	def from_documents(cls, tokenizer, documents):
		counts = Counter()
		total = 0

		for doc in documents:
			for token in tokenizer.tokens_from(doc):
				counts[token] += 1
				total += 1
		return cls(counts, total)

	@classmethod
	def from_counts(cls, counts):
		total = sum(counts.values())
		cls(counts, total)