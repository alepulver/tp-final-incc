from collections import Counter, defaultdict
import fractions
import math
import functools

class Features:
	# XXX: add zero for combine, neutral addition element
	def combine(self, other):
		raise NotImplementedError()
	def __len__(self):
		raise NotImplementedError()
	def __getitem__(self, key):
		raise NotImplementedError()
	def total_counts(self):
		raise NotImplementedError()
	def items(self):
		raise NotImplementedError()

	@classmethod
	def from_tokens(cls, sequence):
		raise NotImplementedError()
	@classmethod
	def from_windows(cls, sequence):
		raise NotImplementedError()

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
	def items(self):
		return self._entries.items()

	@classmethod
	def from_tokens(cls, sequence):
		entries = Counter()
		total = 0
		for token in sequence:
			entries[token] += 1
			total += 1
		for token in entries.keys():
			entries[token] /= total
		return cls(entries, total)

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
		coeff = -1 / (math.log(self._total) * self._sum_freqs[key])
		return coeff * (self._sum_freqs_log[key] - self._sum_freqs[key]*math.log(self._sum_freqs[key]))
	def total_counts(self):
		return self._total
	def items(self):
		for k in self._sum_freqs.keys():
			yield k, self[k]

	@classmethod
	def from_windows(cls, windows):
		frequencies = (TokenFrequencies.from_tokens(tokens) for tokens in windows)
		return cls.from_frequencies(frequencies)

	@classmethod
	def from_frequencies(cls, sequence):
		sum_freqs = Counter()
		sum_freqs_log = Counter()
		total = 0

		for frequencies in sequence:
			for k,v in frequencies.items():
				sum_freqs[k] += v
				sum_freqs_log[k] += v * math.log(v)
			total += 1

		return cls(sum_freqs, sum_freqs_log, total)

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
	def items(self):
		return self._entries.items()

	@classmethod
	def from_tokens(cls, sequence):
		series = defaultdict(list)
		total_tokens = 0
		for index,token in enumerate(sequence):
			series[token].append(index)
			total_tokens += 1
		return cls(series, total_tokens)

class TokenPairwiseFrequencies(Features):
	# XXX: need to implement windows first
	
	def __init__(self, entries, total):
		self._entries = entries
		self._total = total

	@classmethod
	def from_windows(cls, windows):
		pass