import nltk

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

# stores each word that it outputs and can answer it
class RememberingTokenizer(Tokenizer):
	pass
# merging tokenizer, stores how many words were collapsed into one and how
class StemmingTokenizer(Tokenizer):
	pass
# XXX: alternatively, provide event listener to every tokenizer so they can store things

class FilteringTokenizer(Tokenizer):
	def __init__(self, tokenizer, vocabulary):
		self._tokenizer = tokenizer
		self._vocabulary = set(vocabulary)

	def tokens_from(self, book):
		return filter(lambda x: x in self._vocabulary, self._tokenizer.tokens_from(book))

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
			window.append(element)
			if len(window) >= self._parts_size:
				# need to copy because it is changed later
				yield list(window)
				window.pop(0)

# IDEA: dynamic size window, the one choosing the size is the caller?

class WeightingWindow:
	#def __len__(self):
	#	raise NotImplementedError()
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

class WeightingWindowFactory:
	@classmethod
	def uniform(cls, size):
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
		# add to list without duplicates, but avoid traversing the list
		present = set()
		self._objects = []
		for element in objs:
			if element not in present:
				present.add(element)
				self._objects.append(element)

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