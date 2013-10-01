import nltk
from pyhashxx import hashxx


class Tokenizer:
    def tokens_from(self, book):
        raise NotImplementedError()


class DummyBookTokenizer(Tokenizer):
    def tokens_from(self, book):
        return iter(book.contents())


class DummySequenceTokenizer(Tokenizer):
    def tokens_from(self, sequence):
        return iter(sequence)


class BasicTokenizer(Tokenizer):
    def tokens_from(self, book):
        def is_word(x):
            return x.isalpha() and len(x) > 2

        tokens = nltk.wordpunct_tokenize(book.contents().lower())
        return filter(is_word, tokens)


class FilteringTokenizer(Tokenizer):
    def __init__(self, tokenizer, vocabulary):
        self._tokenizer = tokenizer
        self._vocabulary = set(vocabulary)

    def tokens_from(self, book):
        func = lambda x: x in self._vocabulary
        tokens = self._tokenizer.tokens_from(book)
        return filter(func, tokens)

    def vocabulary(self):
        return self._vocabulary


class TransformingTokenizer(Tokenizer):
    def __init__(self, tokenizer, transform):
        self._tokenizer = tokenizer
        self._transform = transform

    def tokens_from(self, book):
        for token in self._tokenizer.tokens_from(book):
            result = self._transform(token)
            yield result


class CollapsingTokenizer(Tokenizer):
    def __init__(self, tokenizer, vocabulary, null='<<<many words removed>>>'):
        self._tokenizer = tokenizer
        self._vocabulary = set(vocabulary)
        self._null = null

        assert(self._null not in self._vocabulary)

    def tokens_from(self, book):
        convert = lambda x: x if x in self._vocabulary else self._null
        return map(convert, self._tokenizer.tokens_from(book))

    def vocabulary(self):
        return self._vocabulary


class HashingTokenizerFilter:
    def __init__(self, tokenizer, seed=1234):
        self._tokenizer = tokenizer
        self._seed = seed

    def tokens_from(self, book):
        tokens = self._tokenizer.tokens_from(book)
        func = lambda x: hashxx(x.encode(), seed=self._seed)
        return map(func, tokens)
