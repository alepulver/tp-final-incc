import nltk
from collections import Counter
from . import numeric_indexer as ni
import pandas

class Tokenizer:
	def tokens(self, text):
		raise NotImplementedError()

class BasicTokenizer(Tokenizer):
	def extract_from(self, text):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(text.lower())

		return filter(is_word, tokens)

class Features:
	def extractor(self):
		raise NotImplementedError()
	def indexer(self):
		raise NotImplementedError()
	def as_list(self):
		raise NotImplementedError()
	def as_iter(self):
		raise NotImplementedError()
	def __len__(self):
		raise NotImplementedError()

class NormalizedFeatures:
	pass

class TextFeatures(Features):
	def __init__(self, extractor, indexer, features, token_count):
		self._extractor = extractor
		self._indexer = indexer
		self._features = features
		self._token_count = token_count
	def extractor(self):
		return self._extractor
	def as_dict(self):
		return self._features
	def as_iter(self):
		for k,v in self.as_dict().items():
			yield self._indexer.encode(k), v
	def __len__(self):
		return len(self._indexer)

class FeatureExtractor:
	def extract_from(self, document):
		raise NotImplementedError()

class TextFeatureExtractor(FeatureExtractor):
	def __init__(self, tokenizer, indexer):
		self._tokenizer = tokenizer
		self._indexer = indexer
	def tokens(self, document):
		return (token for token in self._tokenizer.extract_from(document) if self._indexer.can_encode(token))
	def indices(self, document):
		return (self._indexer.encode(token) for token in self._tokenizer.extract_from(document) if self._indexer.can_encode(token))

class WordFrequencyExtractor(TextFeatureExtractor):
	def extract_from(self, document):
		features = Counter()
		total_tokens = 0
		for token in self.tokens(document):
			features[token] += 1
			total_tokens += 1
		for token in features.keys():
			features[token] /= total_tokens
		return TextFeatures(self, self._indexer, features, total_tokens)

class WordEntropyExtractor(TextFeatureExtractor):
	def __init__(self, corpus_frequencies):
		self._corpus_frequencies = corpus_frequencies
		self._frequency_extractor = self._corpus_frequencies.extractor()
		super().__init__(self, self._frequency_extractor._tokenizer, self._frequency_extractor._indexer)
	def extract_from(self, document):
		pass

class PossibleFeatureAnalyzer:
	def __init__(self, counts, total):
		self._counts = counts
		self._total = total

	def extremes(first=10, last=10):
		pass

	def prune_quantiles(self, low=.05, high=0.95):
		assert(low < high and 0 < low < 1 and 0 < high < 1)
		series = pandas.Series(list(self._counts.values()))
		series.sort()
		vmin = series.quantile(low)
		vmax = series.quantile(high)

		# XXX: add dict filterValues function for this
		counts = Counter()
		total = 0
		for k,v in self._counts.items():
			if vmin < v < vmax:
				counts[k] = v
				total += v
		return self.__class__(counts, total)

	# FIXME: this doesn't work because it partially filters data; we must use a reverse index with subtotals
	def prune_quantiles2(self, low=.05, high=0.95):
		assert(low < high and 0 < low < 1 and 0 < high < 1)
		pairs = [(v,k) for (k,v) in self._counts.items()]
		pairs.sort()
		vmin = self._total * low
		vmax = self._total * high

		counts = Counter()
		total = 0
		current = 0
		for v,k in pairs:
			if current >= vmax:
				break
			if vmin < current:
				counts[k] = v
				total += v
			current += v
		return self.__class__(counts, total)

	def as_dataframe(self):
		words = list(self._counts.keys())
		counts = list(self._counts.values())
		frequencies = list(v/self._total for v in self._counts.values())
		return pandas.DataFrame({'Word': words, 'Count': counts, 'Frequency': frequencies})

	def build_indexer(self):
		return ni.NumericIndexer(self._counts.keys())

	@classmethod
	def from_documents(cls, tokenizer, documents):
		counts = Counter()
		total = 0

		for doc in documents:
			for token in tokenizer.extract_from(doc):
				counts[token] += 1
				total += 1
		return cls(counts, total)

	@classmethod
	def from_counts(cls, counts):
		total = sum(counts.values())
		cls(counts, total)