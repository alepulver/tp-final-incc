from collections import Counter, defaultdict
import book_classification as bc
import pandas
from functools import reduce

# XXX: this is structurally like a collection, in the sense that offers data for each book and author
# but in addition, there is a total
class HierarchialFeatures:
	def __init__(self, by_book, by_author, total):
		self._by_book = by_book
		self._by_author = by_author
		self._total = total

	def by_book(self, book):
		return self._by_book[book]
	def by_author(self, author):
		return self._by_author[author]
	def total(self):
		return self._total

	@classmethod
	def from_book_collection(cls, collection, extractor):
		features_by_book = {}
		for book in collection.books():
			features_by_book[book] = extractor.extract_from(book)
		features_by_author = {}
		for author in collection.authors():
			features = (features_by_book[book] for book in collection.books_by(author))
			features_by_author[author] = reduce(lambda x,y: x.combine(y), features)
		features_total = reduce(lambda x,y: x.combine(y), features_by_author.values())

		return cls(features_by_book, features_by_author, features_total)

	# TODO: move these methods somewhere else
	def dataframe_books(self):
		frames = []
		for book,features in self._by_book.items():
			df = pandas.DataFrame(list(features.values()),
				index = list(features.keys()), columns=[book.title()])
			frames.append(df)
		return pandas.concat(frames)
	
	def dataframe_authors(self):
		frames = []
		for author,features in self._by_author.items():
			df = pandas.DataFrame(list(features.values()),
				index = list(features.keys()), columns=[author])
			frames.append(df)
		return pandas.concat(frames)
	
	def dataframe_total(self):
		return pandas.DataFrame(list(self.total().values()),
				index = list(self.total().keys()), columns=['Value'])

	def dataframe_books_and_authors(self):
		pass

class PossibleFeatureAnalyzer:
	def __init__(self, collection, extraction_env, frequencies, entropies):
		self._collection = collection
		self._extraction_env = extraction_env
		self._frequencies = frequencies
		self._entropies = entropies

	# TODO
	# and make returning entropies lazy, because they can't be patched like frequencies

	def prune_quantiles(self, pairs, low, high):
		assert(low < high)
		assert(0 <= low <= 1)
		assert(0 <= high <= 1)

		pairs.sort()
		current = 0
		words = set()

		for v,k in pairs:
			if current > high:
				break
			if low <= current:
				words.add(k)
			current += v

		extraction_env = self._extraction_env.restrict_vocabulary(words)
		return self.__class__.from_book_collection(self._collection, extraction_env)

	# FIXME: does not work
	#def prune_entropies_quantiles(self, low, high):
	#	pairs = [(v,k) for (k,v) in self._entropies.total().items()]
	#	return self.prune_quantiles(pairs, low, high)
	
	def prune_frequencies_quantiles(self, low, high):
		pairs = [(v,k) for (k,v) in self._frequencies.total().items()]
		return self.prune_quantiles(pairs, low, high)

	def frequencies(self):
		return self._frequencies
	def entropies(self):
		return self._entropies
	def collection(self):
		return self._collection

	@classmethod
	def from_book_collection(cls, collection, extraction_env=None):
		if extraction_env is None:
			extraction_env = ExtractionEnvironment.default()

		frequencies = bc.HierarchialFeatures.from_book_collection(
			collection, lambda x: extraction_env.frequencies(x))
		entropies = bc.HierarchialFeatures.from_book_collection(
			collection, lambda x: extraction_env.entropies(x))
		extraction_env = extraction_env.restrict_vocabulary(frequencies.total().keys())

		return cls(collection, extraction_env, frequencies, entropies)

class CollectionWordPresence:
	# TODO: rewrite with matrix operations on dataframe, may be the same with entropies
	def __init__(self, collection, tokenizer):
		self._collection = collection
		extractor = bc.FrequenciesExtractor(tokenizer)
		self._frequencies = bc.HierarchialFeatures.from_book_collection(
			self._collection, extractor)

		self._from_total = {}
		self._from_author = {}

		for word,freq in self._frequencies.total().items():
			self._from_total[word] = freq
			self._from_author[word] = {}
			
			word_totals = 0
			for author in self._collection.authors():
				by_author = self._frequencies.by_author(author)
				if word in by_author:
					self._from_author[word][author] = by_author[word] * by_author.total_counts()
					word_totals += by_author[word] * by_author.total_counts()
			for author in self._from_author[word].keys():
				self._from_author[word][author] /= word_totals

	def words(self):
		pass
	def from_total(self, word):
		return self._from_total[word]
	def from_author(self, word):
		return self._from_author[word]

	def as_dataframe(self):
		frames = []
		for author in self._collection.authors():
			indices = []
			values = []
			for word,contribs in self._from_author.items():
				if author in contribs:
					indices.append(word)
					values.append(contribs[author])

			df = pandas.DataFrame(values, index = indices, columns=[author])
			frames.append(df)
		return pandas.concat(frames)

class CollectionVocabularyAnalyzer:
	# TODO: rewrite with matrix operations on dataframe
	def __init__(self, collection, tokenizer):
		self._collection = collection
		extractor = bc.VocabularyExtractor(tokenizer)
		self._vocabulary = bc.HierarchialFeatures.from_book_collection(
			self._collection, extractor)

		self._from_book = Counter()
		self._from_author = Counter()

		for word in self._vocabulary.total().keys():
			for author in self._collection.authors():
				if word in self._vocabulary.by_author(author):
					self._from_author[word] += 1

			for book in self._collection.books():
				if word in self._vocabulary.by_book(book):
					self._from_book[word] += 1

	def words_by_author_counts(self):
		counts = []
		for i in range(len(self._collection.authors())):
			current = []
			for k,v in self._from_author.items():
				if v >= (i+1):
					current.append(k)
			counts.append(current)
		return counts

	def words_by_book_counts(self):
		counts = []
		for i in range(len(self._collection.books())):
			current = []
			for k,v in self._from_book.items():
				if v >= (i+1):
					current.append(k)
			counts.append(current)
		return counts