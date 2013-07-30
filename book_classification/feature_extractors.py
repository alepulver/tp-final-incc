import nltk

class BasicTokenizer:
	def extract_from(self, text):
		def is_word(x):
			return x.isalpha() and len(x) > 2
		tokens = nltk.wordpunct_tokenize(text.lower())

		return filter(is_word, tokens)

class FeatureExtractor:
	def __init__(self, tokenizer, indexer):
		self._tokenizer = tokenizer
		self._indexer = indexer
	def extract(self, featureType, text):
		# XXX: should this logic be built into the tokenizer itself? see TokenFilter
		tokens = (t for t in self._tokenizer.extract_from(text) if self._indexer.can_encode(t))
		return featureType.from_tokens(tokens)

class WindowedFeatureExtractor:
	def __init__(self, tokenizer, indexer, windower):
		self._tokenizer = tokenizer
		self._indexer = indexer
		self._windower = windower
	def extract(self, featureType, text):
		tokens = (t for t in self._tokenizer.extract_from(text) if self._indexer.can_encode(t))
		windows = self._windower.extract_from(tokens)
		return featureType.from_windows(windows)

class TokenFilter:
	def __init__(self, indexer):
		self._indexer = indexer
	def extract_from(self, sequence):
		for token in sequence:
			if self._indexer.can_encode(token):
				yield token