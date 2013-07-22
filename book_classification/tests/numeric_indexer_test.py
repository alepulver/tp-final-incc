from book_classification import numeric_indexer as ni
from nose.tools import *

def test_ShouldEncodeMany():
	aNumericIndexer = ni.NumericIndexer("ABC")
	result = list(aNumericIndexer.encode_many("AAABBC"))
	eq_(result, [0,0,0,1,1,2])

def test_ShouldDecodeMany():
	aNumericIndexer = ni.NumericIndexer("ABC")
	result = list(aNumericIndexer.decode_many([0,0,0,1,1,2]))
	eq_("".join(result), "AAABBC")