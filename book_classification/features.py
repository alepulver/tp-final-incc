from collections import Counter, defaultdict
from . import numeric_indexer as ni
import pandas
import fractions

class Features:
	def __init__(self, extractor, data, total):
		self._extractor = extractor
		self._data = data
		self._total = total
	def extractor(self):
		return self._extractor
	def as_dict(self):
		return self._data
	def combine(self, other):
		raise NotImplementedError()
	def __len__(self):
		return len(self._data)

class AtomicFeatures(Features):
	def as_iter(self):
		for k,v in self.as_dict().items():
			yield self._extractor._indexer.encode(k), v

	def combine(self, other):
		assert(self._extractor == other._extractor)
		data = Counter()
		gcd = fractions.gcd(self._total, other._total)
		total = self._total * other._total / gcd

		for k,v in self._data.items():
			data[k] += v * self._total/gcd
		for k,v in other._data.items():
			data[k] += v * other._total/gcd
		
		return self.__class__(self._extractor, data, total)

class SeriesFeatures(Features):
	def combine(self, other):
		assert(self._extractor == other._extractor)
		data = defaultdict(list)
		total = self._total + other._total
		
		for k,v in self._data.items():
			data[k].extend(x)
		for k,v in other._data.items():
			data[k].extend(x + self._total for x in v)
		
		return self.__class__(self._extractor, data, total)		

class PossibleFeatureAnalyzer:
	def __init__(self, counts, total):
		self._counts = counts
		self._total = total

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