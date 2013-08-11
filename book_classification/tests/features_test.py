import book_classification as bc
from nose.tools import *

def test_CanExtractVocabularies():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.VocabulariesExtractor(tokenizer)
	
	vocabularies = extractor.extract_from(
		["one", "two", "one", "three", "three", "three", "three"])
	expected = {'three': 1, 'one': 1, 'two': 1}
	
	eq_(len(vocabularies), 3)
	eq_(vocabularies.total_counts(), 3)
	eq_(dict(vocabularies.items()), expected)

def test_CanExtractFrequencies():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	
	frequencies = extractor.extract_from(
		["one", "two", "one", "three", "three", "three", "three"])
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	
	eq_(len(frequencies), 3)
	eq_(frequencies.total_counts(), 7)
	eq_(dict(frequencies.items()), expected)

def test_CanExtractSeries():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.SeriesExtractor(tokenizer)
	
	series = extractor.extract_from(
		["one", "two", "one", "three", "three", "two", "three"])
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	
	eq_(len(series), 3)
	eq_(series.total_counts(), 7)
	eq_(dict(series.items()), expected)

def test_CanExtractEntropies():
	tokenizer = bc.DummyTokenizer()
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
	tokenizer = bc.DummyTokenizer()
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

def test_CanCombineVocabularies():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.VocabulariesExtractor(tokenizer)
	
	vocabulariesOne = extractor.extract_from(["one", "two", "three", "three"])
	vocabulariesTwo = extractor.extract_from(["one", "three", "three"])	
	result = vocabulariesOne.combine(vocabulariesTwo)
	expected = {'three': 1, 'one': 1, 'two': 1}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 3)
	eq_(dict(result.items()), expected)

def test_CanCombineFrequencies():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	
	frequenciesOne = extractor.extract_from(["one", "two", "three", "three"])
	frequenciesTwo = extractor.extract_from(["one", "three", "three"])
	result = frequenciesOne.combine(frequenciesTwo)
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanCombineEntropies():
	tokenizer = bc.DummyTokenizer()
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

def test_CanCombineSeries():
	tokenizer = bc.DummyTokenizer()
	extractor = bc.SeriesExtractor(tokenizer)

	seriesOne = extractor.extract_from(["one", "two", "one"])
	seriesTwo = extractor.extract_from(["three", "three", "two", "three"])
	result = seriesOne.combine(seriesTwo)
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanCombinePairwiseAssociation():
	tokenizer = bc.DummyTokenizer()
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

def test_FixedExtractorBehavesTheSameWithFullVocabulary():
	tokens = ["one", "two", "one", "three", "three", "three", "three"]
	tokenizer = bc.DummyTokenizer()
	vocabulary = bc.VocabulariesExtractor(tokenizer).extract_from(tokens)
	extractor = bc.FixedExtractor(bc.VocabulariesExtractor(tokenizer), vocabulary)
	
	vocabularies = extractor.extract_from(tokens)
	expected = {'three': 1, 'one': 1, 'two': 1}
	
	eq_(len(vocabularies), 3)
	eq_(vocabularies.total_counts(), 3)
	eq_(dict(vocabularies.items()), expected)

def test_FixedExtractorOmitsFeatures():
	tokens = ["one", "two", "one", "three", "three", "three", "three"]
	tokenizer = bc.DummyTokenizer()
	vocabulary = ["one", "three"]
	extractor = bc.FixedExtractor(bc.VocabulariesExtractor(tokenizer), vocabulary)
	
	vocabularies = extractor.extract_from(tokens)
	expected = {'three': 1, 'one': 1}
	
	eq_(len(vocabularies), 2)
	eq_(vocabularies.total_counts(), 2)
	eq_(dict(vocabularies.items()), expected)