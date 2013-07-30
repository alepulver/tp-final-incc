from collections import Counter, defaultdict
from . import numeric_indexer as ni
import pandas

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