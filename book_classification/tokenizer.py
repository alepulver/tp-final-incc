import nltk

class Tokenizer:
	def tokens_from(self, book):
		raise NotImplementedError()
	def if_vocabulary(self, other):
		raise NotImplementedError()

class MixinTokenizerEvents:
	def add_listener(self, listener):
		self._listeners.add(listener)
	def remove_listener(self, listener):
		self._listeners.remove(listener)
	def broadcast(self, message):
		for listener in self._listeners:
			try:
				message(listener)
			except:
				pass

class DynamicTokenizer(Tokenizer):
	def if_vocabulary(self, other):
		return other.case_dynamic_vocabulary(self)

class FixedTokenizer(Tokenizer):
	def if_vocabulary(self, other):
		return other.case_fixed_vocabulary(self)
	def vocabulary(self):
		raise NotImplementedError()

class TokenizerListener:
	def allow(self, token):
		raise NotImplementedError()
	def discard(self, token):
		raise NotImplementedError()
	def convert(self, token_in, token_out):
		raise NotImplementedError()

class DummyBookTokenizer(DynamicTokenizer):
	def tokens_from(self, book):
		return iter(book.contents())

class DummySequenceTokenizer(DynamicTokenizer):
	def tokens_from(self, sequence):
		return iter(sequence)

class BasicTokenizer(DynamicTokenizer):
	def tokens_from(self, book):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(book.contents().lower())

		return filter(is_word, tokens)

class FilteringTokenizer(FixedTokenizer):
	def __init__(self, tokenizer, vocabulary):
		self._tokenizer = tokenizer
		self._vocabulary = set(vocabulary)

	def tokens_from(self, book):
		return filter(lambda x: x in self._vocabulary, self._tokenizer.tokens_from(book))
	def vocabulary(self):
		return self._vocabulary

class CollapsingTokenizer(MixinTokenizerEvents, FixedTokenizer):
	def __init__(self, tokenizer, vocabulary, fillvalue):
		self._tokenizer = tokenizer
		self._vocabulary = set(vocabulary)
		self._fillvalue = fillvalue
		self._listeners = set()

	def tokens_from(self, book):
		for token in self._tokenizer.tokens_from(book):
			if token in self._vocabulary:
				self.broadcast(lambda x: x.allow(token))
				yield token
			elif token == self._fillvalue:
				raise Exception('fill value occurred at input')
			else:
				self.broadcast(lambda x: x.convert(token, self._fillvalue))
				yield self._fillvalue
	def vocabulary(self):
		return self._vocabulary.union(set([self._fillvalue]))

class StemmingTokenizer(MixinTokenizerEvents, DynamicTokenizer):
	def __init__(self, tokenizer, stemmer):
		self._tokenizer = tokenizer
		self._stemmer = stemmer
		self._listeners = set()

	def tokens_from(self, book):
		for token in self._tokenizer.tokens_from(book):
			result = stemmer(token)
			if token == result:
				self.broadcast(lambda x: x.allow(token))
			else:
				self.broadcast(lambda x: x.convert(token, self._fillvalue))

			yield result

# ..., stores each word that it outputs and can answer it
# ..., stores how many words were collapsed into one and how
class TokenizerEventAnalyzer:
	def __init__(self, tokenizer):
		self._tokenizer = tokenizer
		self._tokenizer.add_listener(self)
	def allow(self, token):
		pass
	def convert(self, token_in, token_out):
		pass
	def discard(self, token):
		pass
	def close(self):
		self._tokenizer.remove_listener(self)