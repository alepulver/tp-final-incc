import nltk
import book_classification as bc


class Tokenizer:
    def tokens_from(self, book):
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

    def uuid(self):
        return bc.digest(self.__class__.__name__)


class FilteringTokenizer(Tokenizer):
    def __init__(self, tokenizer, vocabulary):
        self._tokenizer = tokenizer
        self._vocabulary = set(vocabulary)

    def tokens_from(self, book):
        return filter(lambda x: x in self._vocabulary, self._tokenizer.tokens_from(book))

    def vocabulary(self):
        return self._vocabulary

    # FIXME: use sorted list, set order not determined!
    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, bc.digest(self.vocabulary()))
        return bc.digest(text)


class CollapsingFilteringTokenizer(Tokenizer):
    def __init__(self, tokenizer, vocabulary):
        self._tokenizer = tokenizer
        self._vocabulary = set(vocabulary)
        self._null = '<<<many words removed>>>'
        assert(self._null not in self._vocabulary)

    def tokens_from(self, book):
        def convert(token):
            if token in self._vocabulary:
                return token
            else:
                return self._null

        return map(convert, self._tokenizer.tokens_from(book))

    def vocabulary(self):
        return self._vocabulary.union(set([self._null]))

    def __hash__(self):
        text = "%s(%s)" % (self.__class__.__name__, hash(self.vocabulary()))
        return hash(text)


# FIXME: merge with previous tokenizer
class CollapsingTokenizer(MixinTokenizerEvents, Tokenizer):
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

    def __hash__(self):
        text = "%s(%s)" % (self.__class__.__name__, hash(self.vocabulary()))
        return hash(text)


class StemmingTokenizer(MixinTokenizerEvents, Tokenizer):
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


class TokenizerListener:
    def allow(self, token):
        raise NotImplementedError()

    def discard(self, token):
        raise NotImplementedError()

    def convert(self, token_in, token_out):
        raise NotImplementedError()


# ..., stores each word that it outputs and can answer it
# ..., stores how many words were collapsed into one and how
class TokenizerEventAnalyzer(TokenizerListener):
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
