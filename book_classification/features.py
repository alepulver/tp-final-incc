import nltk
from collections import Counter
from . import numeric_indexer as ni

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
	def as_dict(self):
		raise NotImplementedError()

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
	def as_list(self):
		# FIXME: return sparse or dense list, depending on the case!
		result = [(self._indexer.encode(a),b) for (a,b) in self._features.items()]
		result.sort(key = lambda x: x[0])
		return [b for (a,b) in result]

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
	# allow passing all documents, reading some statistics and building features with the previous
	# objects; avoiding to recalculate the numbers; only work with tokens, not numbers
	def __init__(self, tokenizer, documents):
		self._tokenizer = tokenizer
		self._documents = list(documents)
		self._features = Counter()
		self._total_tokens = 0

		for doc in documents:
			self.count_from(doc)

	def count_from(self, doc):
		for token in self._tokenizer.extract_from(doc):
			self._features[token] += 1
			self._total_tokens += 1

	def all_features(self):
		return _features.keys()

	def build_indexer(self):
		return ni.NumericIndexer(self._features.keys())

"""
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

class WordCount(TextFeatures):
	def features(self):
		return len(list(self.words()))

class UniqueWordCount(TextFeatures):
	def features(self):
		return len(set(self.words()))

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
"""