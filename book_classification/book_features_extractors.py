from collections import Counter, defaultdict
import math
import book_classification as bc
import shelve


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


class CachedExtractorWrapper:
    def __init__(self, extractor):
        self._extractor = extractor
        self._cache = {}

    def extract_from(self, book):
        if book not in self._cache:
            self._cache[book] = self._extractor.extract_from(book)

        return self._cache[book]


class PersistentExtractorWrapper:
    def __init__(self, extractor, name):
        self._extractor = extractor
        self._name = name
        self._cache = shelve.open(name)

    def extract_from(self, book):
        key = book.title()
        if key not in self._cache:
            self._cache[key] = self._extractor.extract_from(book)

        return self._cache[key]

    def close(self):
        self._cache.close()


class PersistentExtractorWrapper2:
    def __enter__(self, extractor, name):
        self._extractor = extractor
        self._name = name
        self._cache = shelve.open(name)

    def extract_from(self, book):
        if book not in self._cache:
            self._cache[book] = self._extractor.extract_from(book)

        return self._cache[book]

    def __exit__(self, type, value, traceback):
        self._cache.close()
