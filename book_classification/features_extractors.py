from collections import Counter, defaultdict
import math
import book_classification as bc

class Extractor:
	def extract_from(self, book):
		raise NotImplementedError()
	def features_for_vocabulary(self, vocabulary):
		raise NotImplementedError()

class VocabulariesExtractor(Extractor):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer

	def extract_from(self, book):
		data = {}
		for token in self._tokenizer.tokens_from(book):
			data[token] = True
		return bc.TokenVocabularies(self, data)
	
	def features_for_vocabulary(self, vocabulary):
		return vocabulary
		
class FrequenciesExtractor(Extractor):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer

	def extract_from(self, book):
		entries = Counter()
		total = 0
		for token in self._tokenizer.tokens_from(book):
			entries[token] += 1
			total += 1
		for token in entries.keys():
			entries[token] /= total
		return bc.TokenFrequencies(self, entries, total)
	
	def features_for_vocabulary(self, vocabulary):
		return vocabulary

class SeriesExtractor(Extractor):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer

	def extract_from(self, book):
		series = defaultdict(list)
		total_tokens = 0
		for index,token in enumerate(self._tokenizer.tokens_from(book)):
			series[token].append(index)
			total_tokens += 1
		return bc.TokenSeries(self, series, total_tokens)
	
	def features_for_vocabulary(self, vocabulary):
		return vocabulary

class EntropiesExtractor(Extractor):
	def __init__(self, tokenizer, grouper):
		self._tokenizer = tokenizer
		self._grouper = grouper

	def extract_from(self, book):
		parts = self._grouper.parts_from(self._tokenizer.tokens_from(book))
		frequencies_extractor = FrequenciesExtractor(bc.DummySequenceTokenizer())
		frequencies_list = (frequencies_extractor.extract_from(p) for p in parts)

		sum_freqs = Counter()
		sum_freqs_log = Counter()
		total = 0

		for frequencies in frequencies_list:
			for k,v in frequencies.items():
				sum_freqs[k] += v
				sum_freqs_log[k] += v * math.log(v)
			total += 1

		return bc.TokenEntropies(self, sum_freqs, sum_freqs_log, total)
	
	def features_for_vocabulary(self, vocabulary):
		return vocabulary

# TODO: only restrict tokenizer, but return all combinations with zero value as features and put them in matrix
class PairwiseAssociationExtractor(Extractor):
	def __init__(self, tokenizer, window_size, weighting_window):
		self._tokenizer = tokenizer
		self._window_size = weighting_window
		self._weighting_window = weighting_window
		self._grouper = bc.SlidingGrouper(self._window_size)

	def extract_from(self, book):
		entries = Counter()
		total = 0
		
		for window in self._grouper.parts_from(self._tokenizer.tokens_from(book)):
			center = self._weighting_window.center_for(window)
			for element,weight in self._weighting_window.weights_for(window):
				entries[(center,element)] += weight
				total += 1

		return bc.TokenPairwiseAssociation(self, entries, total)

	def features_for_vocabulary(self, vocabulary):
		result = set()
		for word1 in vocabulary:
			for word2 in vocabulary:
				result.add((word1, word2))
		return result