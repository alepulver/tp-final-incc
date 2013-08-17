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

def test_VocabulariesFeaturesNames():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.VocabulariesExtractor(tokenizer)
	vocabulary = set(["one", "two", "three"])
	eq_(extractor.features_for_vocabulary(vocabulary), vocabulary)

def test_CanExtractFrequencies():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	
	frequencies = extractor.extract_from(
		["one", "two", "one", "three", "three", "three", "three"])
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	
	eq_(len(frequencies), 3)
	eq_(frequencies.total_counts(), 7)
	eq_(dict(frequencies.items()), expected)

def test_FrequenciesFeaturesNames():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	vocabulary = set(["one", "two", "three"])
	eq_(extractor.features_for_vocabulary(vocabulary), vocabulary)

def test_CanExtractSeries():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.SeriesExtractor(tokenizer)
	
	series = extractor.extract_from(
		["one", "two", "one", "three", "three", "two", "three"])
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	
	eq_(len(series), 3)
	eq_(series.total_counts(), 7)
	eq_(dict(series.items()), expected)

def test_SeriesFeaturesNames():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.SeriesExtractor(tokenizer)
	vocabulary = set(["one", "two", "three"])
	eq_(extractor.features_for_vocabulary(vocabulary), vocabulary)

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

def test_EntropiesFeaturesNames():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.EntropiesExtractor(tokenizer)
	vocabulary = set(["one", "two", "three"])
	eq_(extractor.features_for_vocabulary(vocabulary), vocabulary)

def test_CanExtractPairwiseAssociation():
	tokenizer = bc.DummySequenceTokenizer()
	weighting_window = bc.WeightingWindow.uniform(5)
	extractor = bc.PairwiseAssociationExtractor(tokenizer, weighting_window)
	
	assocs = extractor.extract_from(
		["one", "two", "one", "three", "three", "two",
		"three", "one", "two", "three", "one", "one"])
	expected = {('three', 'one'): 0.6000000000000001, ('three', 'three'): 1.0,
		('three', 'two'): 0.4, ('one', 'two'): 1.2, ('one', 'one'): 0.6000000000000001,
		('one', 'three'): 1.2, ('two', 'one'): 0.2, ('two', 'two'): 0.2,
		('two', 'three'): 0.6000000000000001}
	
	eq_(len(assocs), 5)
	eq_(assocs.total_counts(), 30)
	eq_(dict(assocs.items()), expected)

def test_PairwiseAssociationFeaturesNames():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.PairwiseAssociationExtractor(tokenizer)
	vocabulary = set(["one", "two", "three"])
	expected = set([("one", "one"), ("two", "two"), ("three", "three"),
		("one", "two"), ("two", "one"), ("one", "three"),
		("three", "one"), ("two", "three"), ("three", "two")])
	eq_(extractor.features_for_vocabulary(vocabulary), expected)