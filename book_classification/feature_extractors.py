from collections import Counter, defaultdict
from .features import AtomicFeatures, SeriesFeatures
import nltk

class SequenceExtractor:
	def extract_from(self, text):
		raise NotImplementedError()

class BasicTokenizer(SequenceExtractor):
	def extract_from(self, text):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(text.lower())

		return filter(is_word, tokens)

class TokenFilter:
	def __init__(self, indexer):
		self._indexer = indexer
	def extract_from(self, sequence):
		for token in sequence:
			if self._indexer.can_encode(token):
				yield token

class FeatureExtractor:
	def extract_from(self, document):
		raise NotImplementedError()

class TokenFeatureExtractor(FeatureExtractor):
	def __init__(self, indexer):
		self._indexer = indexer

def WordFrequencyExtractor(tokenizer, indexer):
	return ComposedFeatureExtractor([tokenizer, TokenFilter(indexer), FrequencyExtractor(indexer)])

class FrequencyExtractor(TokenFeatureExtractor):
	def extract_from(self, sequence):
		features = Counter()
		total_tokens = 0
		for token in sequence:
			features[token] += 1
			total_tokens += 1
		for token in features.keys():
			features[token] /= total_tokens
		return AtomicFeatures(self, features, total_tokens)

class SeriesExtractor(TokenFeatureExtractor):
	def extract_from(self, sequence):
		series = defaultdict(list)
		total_tokens = 0
		for index,token in enumerate(sequence):
			series[token].append(index)
			total_tokens += 1
		return SeriesFeatures(self, series, total_tokens)

class TokenGrouper(TokenFeatureExtractor):
	def __init__(self, indexer, winsize):
		super().__init__(indexer)
		self._winsize = winsize
	def extract_from(self, sequence):
		group = []
		for x in sequence:
			if len(group) >= self._winsize:
				yield group
				group = []
			group.append(x)
		if len(group) > 0:
			yield group

class MovingWindowAggregator:
	def extract_from(self, series):
		for k,v in series.as_dict().items():
			pass

class ComposedFeatureExtractor:
	def __init__(self, extractors):
		self._extractors = list(extractors)
	def extract_from(self, document):
		result = document
		for extractor in self._extractors:
			result = extractor.extract_from(result)
		return result

class SequencedFeatureExtractor:
	def __init__(self, extractor):
		self._extractor = extractor
	def extract_from(self, documents):
		for doc in documents:
			yield self._extractor.extract_from(doc)

"""
class WordEntropyExtractor(TextFeatureExtractor):
	def __init__(self, corpus_frequencies):
		self._corpus_frequencies = corpus_frequencies
		self._frequency_extractor = self._corpus_frequencies.extractor()
		super().__init__(self, self._frequency_extractor._tokenizer, self._frequency_extractor._indexer)
	def extract_from(self, document):
		pass
"""