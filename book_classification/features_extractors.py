from collections import Counter, defaultdict
import math
import book_classification as bc


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

    def features_for_vocabulary(self, vocabulary):
        return vocabulary


class PairwiseAssociationExtractor(Extractor):
    def __init__(self, tokenizer, grouper, weights):
        self._tokenizer = tokenizer
        self._grouper = grouper(len(weights))
        self._weights = weights

    def extract_from(self, book):
        entries = Counter()
        total = 0

        center = len(self._weights) // 2 + len(self._weights) % 2
        for words in self._grouper.parts_from(self._tokenizer.tokens_from(book)):
            for k, w in zip(words, self._weights):
                entries[(words[center], k)] += w
                total += 1

        return bc.TokenPairwiseAssociation(self, entries, total)
