from collections import Counter, defaultdict
import math
import book_classification as bc


class Features:
    def extractor(self):
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        # TODO: implement neutral element for combination
        raise NotImplementedError()

    def combine(self, other):
        raise NotImplementedError()

    def __getitem__(self, key):
        raise NotImplementedError()

    def total_counts(self):
        raise NotImplementedError()


# TODO: throw away this, and provide encodeAs/etc method interacting with an EncodingStrategy
class MixinFeaturesDict:
    def extractor(self):
        return self._extractor

    def __getitem__(self, key):
        return self._entries[key]

    def total_counts(self):
        return self._total

    def __len__(self):
        return len(self._entries)

    def keys(self):
        return self._entries.keys()

    def values(self):
        return (self[key] for key in self.keys())

    def items(self):
        return ((key, self[key]) for key in self.keys())

    def __contains__(self, key):
        return key in self.keys()

    def __eq__(self, other):
        return list(sorted(self.items())) == list(sorted(other.items()))

    def __ne__(self, other):
        return not (self == other)


class TokenVocabularies(MixinFeaturesDict, Features):
    def __init__(self, extractor, entries):
        self._extractor = extractor
        self._entries = entries
        self._total = len(self._entries)

    def combine(self, other):
        if self._extractor != other._extractor:
            raise TypeError("can not combine features from different extractors")

        data = self._entries.copy()
        for k in other._entries.keys():
            data[k] = True
        return self.__class__(self._extractor, data)


class TokenFrequencies(MixinFeaturesDict, Features):
    def __init__(self, extractor, entries, total):
        self._extractor = extractor
        self._entries = entries
        self._total = total

    def combine(self, other):
        if self._extractor != other._extractor:
            raise TypeError("can not combine features from different extractors")

        data = Counter()
        total = self._total + other._total

        for k, v in self._entries.items():
            data[k] += v * self._total/total
        for k, v in other._entries.items():
            data[k] += v * other._total/total

        return self.__class__(self._extractor, data, total)


class TokenSeries(MixinFeaturesDict, Features):
    def __init__(self, extractor, entries, total):
        self._extractor = extractor
        self._entries = entries
        self._total = total

    def combine(self, other):
        if self._extractor != other._extractor:
            raise TypeError("can not combine features from different extractors")

        data = defaultdict(list)
        total = self._total + other._total

        for k, v in self._entries.items():
            data[k].extend(v)
        for k, v in other._entries.items():
            data[k].extend(x + self._total for x in v)

        return self.__class__(self._extractor, data, total)


class TokenEntropies(MixinFeaturesDict, Features):
    def __init__(self, extractor, sum_freqs, sum_freqs_log, total):
        self._extractor = extractor
        self._sum_freqs = sum_freqs
        self._sum_freqs_log = sum_freqs_log
        self._total = total

        # for compatibility with MixinFeaturesDict methods
        self._entries = self._sum_freqs

    def combine(self, other):
        if self._extractor != other._extractor:
            raise TypeError("can not combine features from different extractors")

        total = self._total + other._total
        sum_freqs = Counter()
        sum_freqs_log = Counter()

        for k in self._sum_freqs.keys():
            sum_freqs[k] += self._sum_freqs[k]
            sum_freqs_log[k] += self._sum_freqs_log[k]

        for k in other._sum_freqs.keys():
            sum_freqs[k] += other._sum_freqs[k]
            sum_freqs_log[k] += other._sum_freqs_log[k]

        return self.__class__(self._extractor, sum_freqs, sum_freqs_log, total)

    def __getitem__(self, key):
        # FIXME: remove word or return 1 instead of adjusting; add test
        coeff = -1 / (math.log(self._total) * self._sum_freqs[key] + 10**-300)
        return coeff * (self._sum_freqs_log[key] - self._sum_freqs[key]*math.log(self._sum_freqs[key]))


class TokenPairwiseAssociation(MixinFeaturesDict, Features):
    def __init__(self, extractor, entries, total):
        self._extractor = extractor
        self._entries = entries
        self._total = total

    def combine(self, other):
        # XXX: is not exact because some information is discarded,
        # but at least it's associative and commutative

        if self._extractor != other._extractor:
            raise TypeError("can not combine features from different extractors")

        entries = Counter()
        total = 0

        for k, v in self.items():
            entries[k] += v
            total += 1
        for k, v in other.items():
            entries[k] += v
            total += 1

        return TokenPairwiseAssociation(self._extractor, entries, total)