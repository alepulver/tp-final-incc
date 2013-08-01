import book_classification as bc
from nose.tools import *

def test_CanExtractFrequencies():
	frequencies = bc.TokenFrequencies.from_tokens(
		["one", "two", "one", "three", "three", "three", "three"])
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	eq_(len(frequencies), 3)
	eq_(frequencies.total_counts(), 7)
	eq_(dict(frequencies.items()), expected)

def test_CanExtractSeries():
	series = bc.TokenSeries.from_tokens(
		["one", "two", "one", "three", "three", "two", "three"])
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	eq_(len(series), 3)
	eq_(series.total_counts(), 7)
	eq_(dict(series.items()), expected)

def test_CanExtractEntropies():
	entropies = bc.TokenEntropies.from_parts(
		[["one", "two"], ["one", "three"], ["one", "two"], ["one"]])
	#expected = {'one': -0.4535888920010089, 'two': -1.4128711136008072, 'three': -3.1628711136008074}
	expected = {'two': 0.5, 'one': 0.960964047443681, 'three': -0.0}
	eq_(len(entropies), 3)
	eq_(entropies.total_counts(), 4)
	eq_(dict(entropies.items()), expected)

def test_CanExtractPairwiseAssociation():
	weights = bc.WeightingWindow.uniform(5)
	assocs = bc.TokenPairwiseAssociation.from_tokens(
		["one", "two", "one", "three", "three", "two",
		"three", "one", "two", "three", "one", "one"], weights)
	expected = {('three', 'one'): 0.6000000000000001, ('three', 'three'): 1.0,
		('three', 'two'): 0.4, ('one', 'two'): 1.2, ('one', 'one'): 0.6000000000000001,
		('one', 'three'): 1.2, ('two', 'one'): 0.2, ('two', 'two'): 0.2,
		('two', 'three'): 0.6000000000000001}
	eq_(len(assocs), 5)
	eq_(assocs.total_counts(), 30)
	eq_(dict(assocs.items()), expected)

def test_CanCombineFrequencies():
	frequenciesOne = bc.TokenFrequencies.from_tokens(
		["one", "two", "three", "three"])
	frequenciesTwo = bc.TokenFrequencies.from_tokens(
		["one", "three", "three"])
	result = frequenciesOne.combine(frequenciesTwo)
	expected = {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285}
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanCombineEntropies():
	entropiesOne = bc.TokenEntropies.from_parts(
		[["one", "two"], ["one"]])
	entropiesTwo = bc.TokenEntropies.from_parts(
		[["one", "three"], ["one", "two"]])
	result = entropiesOne.combine(entropiesTwo)
	#expected = {'one': -0.4535888920010089, 'two': -1.4128711136008072, 'three': -3.1628711136008074}
	expected = {'two': 0.5, 'one': 0.960964047443681, 'three': -0.0}
	eq_(len(result), 3)
	eq_(result.total_counts(), 4)
	eq_(dict(result.items()), expected)

def test_CanCombineSeries():
	seriesOne = bc.TokenSeries.from_tokens(
		["one", "two", "one"])
	seriesTwo = bc.TokenSeries.from_tokens(
		["three", "three", "two", "three"])
	result = seriesOne.combine(seriesTwo)
	expected = {'three': [3, 4, 6], 'one': [0, 2], 'two': [1, 5]}
	eq_(len(result), 3)
	eq_(result.total_counts(), 7)
	eq_(dict(result.items()), expected)

def test_CanCombinePairwiseAssociation():
	weights = bc.WeightingWindow.uniform(5)
	assocsOne = bc.TokenPairwiseAssociation.from_tokens(
		["one", "two", "one", "three", "three", "two"], weights)
	assocsTwo = bc.TokenPairwiseAssociation.from_tokens(
		["three", "one", "two", "three", "one", "one"], weights)
	result = assocsOne.combine(assocsTwo)
	expected = {('three', 'one'): 0.6000000000000001, ('three', 'three'): 1.0,
		('three', 'two'): 0.4, ('one', 'two'): 1.2, ('one', 'one'): 0.6000000000000001,
		('one', 'three'): 1.2, ('two', 'one'): 0.2, ('two', 'two'): 0.2,
		('two', 'three'): 0.6000000000000001}
	eq_(len(assocs), 5)
	eq_(assocs.total_counts(), 30)
	eq_(dict(assocs.items()), expected)