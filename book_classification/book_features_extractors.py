from collections import Counter, defaultdict
import math
import book_classification as bc
import fcache
import numpy


class Extractor:
    def extract_from(self, book):
        raise NotImplementedError()


def default_cache():
    return fcache.Cache("features", "books_classification")


class CachedExtractor(Extractor):
    def __init__(self, extractor, cache=default_cache()):
        self._extractor = extractor
        self._cache = cache

    def extract_from(self, book):
        element = bc.digest(repr((self._extractor.uuid(), book.uuid())))
        try:
            result = self._cache.get(element)
            #print('reusing %s' % repr(element))
            return result
        except KeyError:
            #print('calculating %s' % repr(element))
            result = self._extractor.extract_from(book)
            self._cache.set(element, result)
            return result


class VocabulariesExtractor(Extractor):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def extract_from(self, book):
        data = {}
        for token in self._tokenizer.tokens_from(book):
            data[token] = True
        return bc.TokenVocabularies(self, data)

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._tokenizer.uuid())
        return bc.digest(text)


class FrequenciesExtractor(Extractor):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def extract_from(self, book):
        entries = Counter()
        total = 0
        for token in self._tokenizer.tokens_from(book):
            entries[token] += 1
            total += 1
        for token in entries.keys():
            entries[token] /= total
        return bc.TokenFrequencies(self, entries, total)

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._tokenizer.uuid())
        return bc.digest(text)


class SeriesExtractor(Extractor):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def extract_from(self, book):
        series = defaultdict(list)
        total_tokens = 0
        for index, token in enumerate(self._tokenizer.tokens_from(book)):
            series[token].append(index)
            total_tokens += 1
        return bc.TokenSeries(self, series, total_tokens)

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._tokenizer.uuid())
        return bc.digest(text)


class EntropiesExtractor(Extractor):
    def __init__(self, tokenizer, grouper):
        self._tokenizer = tokenizer
        self._grouper = grouper

    def extract_from(self, book):
        parts = self._grouper.parts_from(self._tokenizer.tokens_from(book))
        frequencies_extractor = FrequenciesExtractor(bc.DummySequenceTokenizer())
        frequencies_list = (frequencies_extractor.extract_from(p) for p in parts)

        sum_freqs = Counter()
        sum_freqs_log = Counter()
        total = 0

        for frequencies in frequencies_list:
            for k, v in frequencies.items():
                sum_freqs[k] += v
                sum_freqs_log[k] += v * math.log(v)
            total += 1

        return bc.TokenEntropies(self, sum_freqs, sum_freqs_log, total)

    def uuid(self):
        text = "%s(%s,%s)" % (self.__class__.__name__, self._tokenizer.uuid(), self._grouper.uuid())
        return bc.digest(text)


class PairwiseAssociationExtractor(Extractor):
    def __init__(self, tokenizer, grouper, weights, num_features=2**16):
        self._tokenizer = tokenizer
        self._grouper = grouper(len(weights))
        self._weights = weights
        self._num_features = num_features

    def extract_from(self, book):
        entries = numpy.zeros(self._num_features)
        total = 0

        token_stream = self._tokenizer.tokens_from(book)
        for words in self._grouper.parts_from(token_stream):
            words_array = numpy.array(words)
            center = words_array[len(words) // 2]
            indices = (center * 1332711 + words_array) % self._num_features

            for i in range(len(indices)):
                entries[indices[i]] += words_array[i]
                total += 1

        entries /= total
        return bc.TokenPairwiseAssociation(self, entries, total)

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._tokenizer.uuid(), self._grouper.uuid(), bc.digest(repr(weights)))
        return bc.digest(text)

