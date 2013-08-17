import book_classification as bc
from nose.tools import *

def identicalFeaturesAreEqual(builder):
	tokenizer = bc.DummySequenceTokenizer()
	extractor = builder(tokenizer)
	sequence = ["one", "two", "three", "three"]
	
	resultsOne = extractor.extract_from(sequence)
	resultsTwo = extractor.extract_from(sequence)
	eq_(resultsOne, resultsTwo)

def differentFeaturesAreNotEqual(builder):
	tokenizer = bc.DummySequenceTokenizer()
	extractor = builder(tokenizer)
	sequenceOne = ["one", "two", "three", "three"]
	sequenceTwo = ["bye", "four", "three"]
	
	resultsOne = extractor.extract_from(sequenceOne)
	resultsTwo = extractor.extract_from(sequenceTwo)
	ok_(resultsOne != resultsTwo)

def test_CanCompareVocabularies():
	identicalFeaturesAreEqual(lambda x: bc.VocabulariesExtractor(x))
	differentFeaturesAreNotEqual(lambda x: bc.VocabulariesExtractor(x))

def test_CanCombineVocabularies():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.VocabulariesExtractor(tokenizer)
	
	vocabulariesOne = extractor.extract_from(["one", "two", "three", "three"])
	vocabulariesTwo = extractor.extract_from(["one", "three", "three"])	
	result = vocabulariesOne.combine(vocabulariesTwo)
	expected = {'three': 1, 'one': 1, 'two': 1}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 3)
	eq_(dict(result.items()), expected)

def test_CanCompareFrequencies():
	identicalFeaturesAreEqual(lambda x: bc.FrequenciesExtractor(x))
	differentFeaturesAreNotEqual(lambda x: bc.FrequenciesExtractor(x))

def test_CanCombineFrequencies():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	
	frequenciesOne = extractor.extract_from(["one", "two", "three", "three"])
	frequenciesTwo = extractor.extract_from(["one", "three", "three"])
	result = frequenciesOne.combine(frequenciesTwo)
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanCompareEntropies():
	grouper = bc.DummyGrouper()
	identicalFeaturesAreEqual(lambda x: bc.EntropiesExtractor(x, grouper))
	differentFeaturesAreNotEqual(lambda x: bc.EntropiesExtractor(x, grouper))

def test_CanCombineEntropies():
	tokenizer = bc.DummySequenceTokenizer()
	grouper = bc.DummyGrouper()
	extractor = bc.EntropiesExtractor(tokenizer, grouper)
	
	entropiesOne = extractor.extract_from([["one", "two"], ["one"]])
	entropiesTwo = extractor.extract_from([["one", "three"], ["one", "two"]])
	result = entropiesOne.combine(entropiesTwo)
	#expected = {'one': -0.4535888920010089, 'two': -1.4128711136008072, 'three': -3.1628711136008074}
	expected = {'two': 0.5, 'one': 0.960964047443681, 'three': -0.0}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 4)
	eq_(dict(result.items()), expected)

def test_CanCompareSeries():
	identicalFeaturesAreEqual(lambda x: bc.SeriesExtractor(x))
	differentFeaturesAreNotEqual(lambda x: bc.SeriesExtractor(x))

def test_CanCombineSeries():
	tokenizer = bc.DummySequenceTokenizer()
	extractor = bc.SeriesExtractor(tokenizer)

	seriesOne = extractor.extract_from(["one", "two", "one"])
	seriesTwo = extractor.extract_from(["three", "three", "two", "three"])
	result = seriesOne.combine(seriesTwo)
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanComparePairwiseAssociation():
	return
	weighting_window = bc.WeightingWindow.uniform(5)
	identicalFeaturesAreEqual(lambda x: bc.PairwiseAssociationExtractor(x, weighting_window))
	differentFeaturesAreNotEqual(lambda x: bc.PairwiseAssociationExtractor(x, weighting_window))

def test_CanCombinePairwiseAssociation():
	return
	tokenizer = bc.DummySequenceTokenizer()
	weighting_window = bc.WeightingWindow.uniform(5)
	extractor = bc.PairwiseAssociationExtractor(tokenizer, weighting_window)

	assocsOne = extractor.extract_from(
		["one", "two", "one", "three", "three", "two"])
	assocsTwo = extractor.extract_from(
		["three", "one", "two", "three", "one", "one"])
	result = assocsOne.combine(assocsTwo)
	expected = {('three', 'one'): 0.6000000000000001, ('three', 'three'): 1.0,
		('three', 'two'): 0.4, ('one', 'two'): 1.2, ('one', 'one'): 0.6000000000000001,
		('one', 'three'): 1.2, ('two', 'one'): 0.2, ('two', 'two'): 0.2,
		('two', 'three'): 0.6000000000000001}
	
	eq_(len(assocs), 5)
	eq_(assocs.total_counts(), 30)
	eq_(dict(assocs.items()), expected)