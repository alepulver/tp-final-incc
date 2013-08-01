import book_classification as bc
import os
from nose.tools import *

def test_CanExtractFrequenciesFromText():
	text = "one two one three three four four three three"
	indexer = bc.NumericIndexer(["two", "three", "blah","one"])
	extractor = bc.FeatureExtractor(bc.BasicTokenizer(), indexer)
	result = extractor.extract(bc.TokenFrequencies, text)
	eq_(dict(result.items()), {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285})