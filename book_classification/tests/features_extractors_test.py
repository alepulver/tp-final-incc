import book_classification as bc
from nose.tools import *


def test_CanExtractVocabularies():
    tokenizer = bc.DummySequenceTokenizer()
    extractor = bc.VocabulariesExtractor(tokenizer)

    vocabularies = extractor.extract_from(
        ["one", "two", "one", "three", "three", "three", "three"])
    expected = {'three': 1, 'one': 1, 'two': 1}

    eq_(len(vocabularies), 3)
    eq_(vocabularies.total_counts(), 3)
    eq_(dict(vocabularies.items()), expected)


def test_CanExtractFrequencies():
    tokenizer = bc.DummySequenceTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)

    frequencies = extractor.extract_from(
        ["one", "two", "one", "three", "three", "three", "three"])
    expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}

    eq_(len(frequencies), 3)
    eq_(frequencies.total_counts(), 7)
    eq_(dict(frequencies.items()), expected)


def test_CanExtractSeries():
    tokenizer = bc.DummySequenceTokenizer()
    extractor = bc.SeriesExtractor(tokenizer)

    series = extractor.extract_from(
        ["one", "two", "one", "three", "three", "two", "three"])
    expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}

    eq_(len(series), 3)
    eq_(series.total_counts(), 7)
    eq_(dict(series.items()), expected)


def test_CanExtractEntropies():
    tokenizer = bc.DummySequenceTokenizer()
    grouper = bc.DummyGrouper()
    extractor = bc.EntropiesExtractor(tokenizer, grouper)

    entropies = extractor.extract_from(
        [["one", "two"], ["one", "three"], ["one", "two"], ["one"]])
    #expected = {'one': -0.4535888920010089, 'two': -1.4128711136008072, 'three': -3.1628711136008074}
    expected = {'two': 0.5, 'one': 0.960964047443681, 'three': -0.0}

    eq_(len(entropies), 3)
    eq_(entropies.total_counts(), 4)
    eq_(dict(entropies.items()), expected)


def test_CanExtractPairwiseAssociation():
    tokenizer = bc.DummySequenceTokenizer()
    grouper = bc.SlidingGrouper
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    extractor = bc.PairwiseAssociationExtractor(tokenizer, grouper, weights)

    assocs = extractor.extract_from(
        ["one", "two", "one", "three", "three", "two",
        "three", "one", "two", "three", "one", "one"])
    expected = {
        ('three', 'three'): 1.7000000000000002, ('three', 'one'): 1.1, ('one', 'three'): 0.9,
        ('two', 'three'): 1.0000000000000002, ('one', 'two'): 0.5, ('one', 'one'): 0.6,
        ('two', 'two'): 0.5, ('three', 'two'): 1.2000000000000002, ('two', 'one'): 0.5
    }

    eq_(len(assocs), 9)
    eq_(assocs.total_counts(), 40)
    eq_(dict(assocs.items()), expected)
