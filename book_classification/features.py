from collections import Counter, defaultdict
import math
import book_classification as bc

class Extractor:
	def extract_from(self, book):
		raise NotImplementedError()

class FixedExtractor(Extractor):
	def __init__(self, extractor, vocabulary):
		self._extractor = extractor
		self._vocabulary = vocabulary
	
	def extract_from(self, book):
		return FixedFeatures(self._extractor.extract_from(book), self._vocabulary)
	#def vocabulary(self):
	#	return self._vocabulary.total().keys()

	@classmethod
	def from_collection(cls, extractor, collection):
		raise NotImplementedError()

class Features:
	@classmethod
	def zero(cls):
		# TODO: implement neutral element for combination
		raise NotImplementedError()
	def combine(self, other):
		raise NotImplementedError()

	# TODO: use dict "mixin" or ABC to avoid boilerplate
	def __len__(self):
		raise NotImplementedError()
	def __getitem__(self, key):
		raise NotImplementedError()
	def total_counts(self):
		raise NotImplementedError()
	def keys(self):
		raise NotImplementedError()
	def values(self):
		return (self[key] for key in self.keys())
	def items(self):
		return ((key, self[key]) for key in self.keys())
	def __contains__(self, key):
		return key in self.keys()

class FixedFeatures(Features):
	def __init__(self, features, vocabulary):
		# IDEA: let vocabulary decide, instead of passing the complete set
		# it only needs to be able to tell its own length, but not which elements
		self._features = features
		self._vocabulary = vocabulary
		raise NotImplementedError()
	# TODO ...

class VocabulariesExtractor(Extractor):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer

	def extract_from(self, book):
		data = {}
		for token in self._tokenizer.tokens_from(book):
			data[token] = True
		return TokenVocabularies(data)

class TokenVocabularies(Features):
	def __init__(self, entries):
		self._entries = entries

	def combine(self, other):
		data = self._entries.copy()
		for k in other._entries.keys():
			data[k] = True
		return self.__class__(data)

	def __len__(self):
		return len(self._entries)
	def __getitem__(self, key):
		return self._entries[key]
	def total_counts(self):
		return len(self)
	def keys(self):
		return self._entries.keys()

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
		return TokenFrequencies(entries, total)

class TokenFrequencies(Features):
	def __init__(self, entries, total):
		self._entries = entries
		self._total = total

	def combine(self, other):
		data = Counter()
		total = self._total + other._total

		for k,v in self._entries.items():
			data[k] += v * self._total/total
		for k,v in other._entries.items():
			data[k] += v * other._total/total
		
		return self.__class__(data, total)

	def __len__(self):
		return len(self._entries)
	def __getitem__(self, key):
		return self._entries[key]
	def total_counts(self):
		return self._total
	def keys(self):
		return self._entries.keys()

class SeriesExtractor(Extractor):
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer

	def extract_from(self, book):
		series = defaultdict(list)
		total_tokens = 0
		for index,token in enumerate(self._tokenizer.tokens_from(book)):
			series[token].append(index)
			total_tokens += 1
		return TokenSeries(series, total_tokens)

class TokenSeries(Features):
	def __init__(self, entries, total):
		self._entries = entries
		self._total = total

	def combine(self, other):
		data = defaultdict(list)
		total = self._total + other._total
		
		for k,v in self._entries.items():
			data[k].extend(v)
		for k,v in other._entries.items():
			data[k].extend(x + self._total for x in v)
		
		return self.__class__(data, total)	

	def __len__(self):
		return len(self._entries)
	def __getitem__(self, key):
		return self._entries[key]
	def total_counts(self):
		return self._total
	def keys(self):
		return self._entries.keys()

class EntropiesExtractor(Extractor):
	def __init__(self, tokenizer, grouper):
		self._tokenizer = tokenizer
		self._grouper = grouper

	def extract_from(self, book):
		parts = self._grouper.parts_from(self._tokenizer.tokens_from(book))
		frequencies_extractor = FrequenciesExtractor(bc.DummyTokenizer())
		frequencies_list = (frequencies_extractor.extract_from(p) for p in parts)

		sum_freqs = Counter()
		sum_freqs_log = Counter()
		total = 0

		for frequencies in frequencies_list:
			for k,v in frequencies.items():
				sum_freqs[k] += v
				sum_freqs_log[k] += v * math.log(v)
			total += 1

		return TokenEntropies(sum_freqs, sum_freqs_log, total)

class TokenEntropies(Features):
	def __init__(self, sum_freqs, sum_freqs_log, total):
		self._sum_freqs = sum_freqs
		self._sum_freqs_log = sum_freqs_log
		self._total = total

	def combine(self, other):
		total = self._total + other._total
		sum_freqs = Counter()
		sum_freqs_log = Counter()
		
		for k in self._sum_freqs.keys():
			sum_freqs[k] += self._sum_freqs[k]
			sum_freqs_log[k] += self._sum_freqs_log[k]
		
		for k in other._sum_freqs.keys():
			sum_freqs[k] += other._sum_freqs[k]
			sum_freqs_log[k] += other._sum_freqs_log[k]
		
		return self.__class__(sum_freqs, sum_freqs_log, total)

	def __len__(self):
		return len(self._sum_freqs)
	def __getitem__(self, key):
		# FIXME: remove word or return 1 instead of adjusting; add test
		coeff = -1 / (math.log(self._total) * self._sum_freqs[key] + 10**-300)
		return coeff * (self._sum_freqs_log[key] - self._sum_freqs[key]*math.log(self._sum_freqs[key]))
	def total_counts(self):
		return self._total
	def keys(self):
		return self._sum_freqs.keys()

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

		return TokenPairwiseAssociation(entries, total)

class TokenPairwiseAssociation(Features):
	def __init__(self, entries, total):
		self._entries = entries
		self._total = total

	def combine(self, other):
		# XXX: is not exact because some information is discarded,
		# but at least it's associative and commutative
		entries = Counter()
		total = 0

		for k,v in self.items():
			entries[k] += v
			total += 1
		for k,v in other.items():
			entries[k] += v
			total += 1

		return TokenPairwiseAssociation(entries, total)

	def __len__(self):
		return len(self._weights)
	def __getitem__(self, key):
		return self._entries[key]
	def total_counts(self):
		return self._total
	def keys(self):
		return self._entries.keys()