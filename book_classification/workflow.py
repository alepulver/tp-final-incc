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
		return self._by_authors[author]
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

class PossibleVocabularyAnalyzer:
	def __init__(self, collection, tokenizer, vocabularies_by_book, vocabularies_by_author, words_presence):
		self._vocabularies_by_book = vocabularies_by_book
		self._vocabularies_by_author = vocabularies_by_author
		self._words_presence = words_presence

	def xxx(self):
		pass

	def from_book_collection(cls, collection, tokenizer):
		vocabularies_by_book = {}
		vocabularies_by_author = defaultdict(set)
		words_presence_by_book = Counter()
		words_presence_by_author = Counter()

		for book in collection.books():
			vocabularies_by_book[book] = set(tokenizer.tokens_from(book.contents()))
			vocabularies_by_author[book.author()].update(vocabularies_by_book[book])
			for word in vocabularies_by_book[book]:
				words_presence_by_book[word] += 1
		for word in vocabularies_by_author.keys():
			for word in vocabularies_by_book[book]:
				words_presence[word] += 1

		return cls(collection, tokenizer, vocabularies_by_book, vocabularies_by_author, words_presence)