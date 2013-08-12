import book_classification as bc
from nose.tools import *

def test_FixedGrouperCanGroupMultiplesOfSize():
	grouper = bc.FixedGrouper(3)
	result = grouper.parts_from('abcdef')
	eq_(list(result), [['a', 'b', 'c'], ['d', 'e', 'f']])

def test_FixedGrouperCanGroupNonMultiplesOfSize():
	grouper = bc.FixedGrouper(3)
	result = grouper.parts_from('abcdefgh')
	eq_(list(result), [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h']])

def test_SlidingGrouperWithLessThanWindowSize():
	grouper = bc.SlidingGrouper(3)
	result = grouper.parts_from([0, 1])
	eq_(list(result), [])

def test_SlidingGrouperWithExactlyWindowSize():
	grouper = bc.SlidingGrouper(3)
	result = grouper.parts_from([0, 1, 2])
	eq_(list(result), [[0, 1, 2]])

def test_SlidingGrouperWithMoreThanWindowSize():
	grouper = bc.SlidingGrouper(3)
	result = grouper.parts_from([0, 1, 2, 3, 4])
	eq_(list(result), [[0, 1, 2], [1, 2, 3], [2, 3, 4]])

# XXX: weighting window tests

def test_NumericIndexerOnlyRecognizesSomeTokens():
	aNumericIndexer = bc.NumericIndexer(['one', 'two'])
	eq_(aNumericIndexer.can_encode('one'), True)
	eq_(aNumericIndexer.can_decode(1), True)
	eq_(aNumericIndexer.can_encode('three'), False)
	eq_(aNumericIndexer.can_decode(2), False)

def encode_many(indexer, objs):
    return [indexer.encode(o) for o in objs]

def decode_many(indexer, indices):
    return [indexer.decode(i) for i in indices]

def test_NumericIndexerShouldEncode():
	aNumericIndexer = bc.NumericIndexer("ABC")
	result = encode_many(aNumericIndexer, "AAABBC")
	eq_(result, [0,0,0,1,1,2])

def test_NumericIndexerShouldDecode():
	aNumericIndexer = bc.NumericIndexer("ABC")
	result = decode_many(aNumericIndexer, [0,0,0,1,1,2])
	eq_("".join(result), "AAABBC")