import nltk
from functools import reduce
import pandas

class Tokenizer:
	def tokens_from(self, book):
		raise NotImplementedError()

class DummyTokenizer(Tokenizer):
	def tokens_from(self, sequence):
		return iter(sequence)

class BasicTokenizer(Tokenizer):
	def tokens_from(self, book):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(book.contents().lower())

		return filter(is_word, tokens)

	def vocabulary(self):
		raise TypeError("BasicTokenizer doesn't have a vocabulary, please use 'restrict_vocabulary'")
	def restrict_vocabulary(self, words):
		return FilteringTokenizer(self, words)

# stores each word that it outputs and can answer it
class RememberingTokenizer(Tokenizer):
	pass

# XXX: merging tokenizer, stores how many words were collapsed into one and how
class StemmingTokenizer(Tokenizer):
	pass

class FilteringTokenizer(Tokenizer):
	def __init__(self, tokenizer, vocabulary):
		self._tokenizer = tokenizer
		self._vocabulary = set(vocabulary)

	def tokens_from(self, book):
		return filter(lambda x: x in self._vocabulary, self._tokenizer.tokens_from(book))

	def vocabulary(self):
		return self._vocabulary
	def restrict_vocabulary(self, words):
		return self.__class__(self._tokenizer, self._vocabulary - words)

class Grouper:
	def parts_from(self, sequence):
		raise NotImplementedError()

class DummyGrouper(Grouper):
	def parts_from(self, sequence):
		return iter(sequence)

class FixedGrouper(Grouper):
	def __init__(self, parts_size):
		self._parts_size = parts_size
	def parts_from(self, sequence):
		group = []
		for token in sequence:
			if len(group) >= self._parts_size:
				yield group
				group = []
			group.append(token)
		if len(group) > 0:
			yield group

class SlidingGrouper(Grouper):
	def __init__(self, parts_size):
		self._parts_size = parts_size
	def parts_from(self, sequence):
		window = []
		for element in sequence:
			window.push(element)
			if len(window) >= self._parts_size:
				yield window
				window.pop()

class WeightingWindow:
	def __len__(self):
		raise NotImplementedError()
	def weights_for(self, window):
		raise NotImplementedError()
	def center_for(self, window):
		raise NotImplementedError()

class DenseWeightingWindow(WeightingWindow):
	pass

class SparseWeightingWindow(WeightingWindow):
	pass

class FunctionWeightingWindow(WeightingWindow):
	pass

class WeightingWindow__:
	# TODO: define symmetric ones by growth function, from 1 to n/2 and reverse after half; normalize at the end

	def __init__(self, weights):
		assert(len(weights) > 0 and len(weights) % 2 == 1)
		assert(abs(1 - sum(weights)) < 10**-5)
		self._weights = weights
	def __getitem__(self, key):
		return self._weights[key]
	def __len__(self):
		return len(self._weights)

	@classmethod
	def uniform(cls, size):
		window = [1/size for _ in range(size)]
		return cls(window)

	@classmethod
	def triangular(cls, size, step):
		middle = size / 2
		window = []
		for i in range(1, middle+2):
			window.append(i*step)
		for i in reversed(range(middle+2, size+1)):
			window.append(i*step)
		normalizer = sum(window)
		return cls(w/normalizer for w in window)

	@classmethod
	def gaussian(cls, size, mu, sigma):
		pass

class NumericIndexer:
    def __init__(self, objs):
        self._objects = list(objs)
        self._indices = dict(zip(self._objects, range(len(self._objects))))

    def __len__(self):
        return len(self._objects)

    def can_encode(self, obj):
        return obj in self._indices

    def can_decode(self, index):
        return index < len(self)

    def encode(self, obj):
        return self._indices[obj]

    def decode(self, index):
        return self._objects[index]

    @classmethod
    def from_objects(cls, objs):
        return cls(set(objs))

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