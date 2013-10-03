from collections import Counter, defaultdict
import math
import book_classification as bc
import numpy
import pyximport; pyximport.install()
from . import optimized
from scipy import sparse


class Extractor:
    def extract_from(self, book):
        raise NotImplementedError()


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
        entries = numpy.zeros(self._num_features, dtype=self._weights.dtype)
        total = 0

        token_stream = self._tokenizer.tokens_from(book)
        for words in self._grouper.parts_from(token_stream):
            optimized.pairwise_association_window(entries, words, self._weights)
            total += 1

        entries /= total
        return bc.TokenPairwiseAssociation(self, entries, total)

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._tokenizer.uuid(), self._grouper.uuid(), bc.digest(repr(weights)))
        return bc.digest(text)


class PairwiseEntropyExtractor(Extractor):
    def __init__(self, tokenizer, grouper, weights, num_features=2**20):
        self._tokenizer = tokenizer
        self._grouper = grouper(len(weights))
        self._weights = weights
        self._num_features = num_features

    def extract_from(self, book):
        # XXX: maybe we should also multiply "center" by something,
        # if we want a result closer to "mutual information"

        #total_entries = sparse.dok_matrix((1, self._num_features))
        total_entries = numpy.zeros(self._num_features)
        total_count = 0

        token_stream = self._tokenizer.tokens_from(book)
        for words in self._grouper.parts_from(token_stream):
            #center = words[len(words) // 2]

            #indices = (center*1664525 + words) % total_entries.shape[1]
            optimized.pairwise_entropy_window(total_entries, words, self._weights)

            total_count += 1

        total_entries /= total_count

        return bc.TokenPairwiseAssociation(self, total_entries, total_count)
