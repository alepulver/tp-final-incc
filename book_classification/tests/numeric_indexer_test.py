import book_classification as bc
from nose.tools import *

def encode_many(indexer, objs):
    return [indexer.encode(o) for o in objs]

def decode_many(indexer, indices):
    return [indexer.decode(i) for i in indices]

def test_OnlyRecognizesSomeTokens():
	aNumericIndexer = bc.NumericIndexer(['one', 'two'])
	eq_(aNumericIndexer.can_encode('one'), True)
	eq_(aNumericIndexer.can_decode(1), True)
	eq_(aNumericIndexer.can_encode('three'), False)
	eq_(aNumericIndexer.can_decode(2), False)

def test_ShouldEncode():
	aNumericIndexer = bc.NumericIndexer("ABC")
	result = encode_many(aNumericIndexer, "AAABBC")
	eq_(result, [0,0,0,1,1,2])

def test_ShouldDecode():
	aNumericIndexer = bc.NumericIndexer("ABC")
	result = decode_many(aNumericIndexer, [0,0,0,1,1,2])
	eq_("".join(result), "AAABBC")

def test_ShouldBuildIndexerWhileEncoding():
	aNumericIndexerBuilder = bc.NumericIndexerBuilder()
	
	tokensOne = ['any', 'sentence', 'any']
	for word in tokensOne:
		eq_(aNumericIndexerBuilder.can_encode(word), True)
	eq_(encode_many(aNumericIndexerBuilder, tokensOne), [0, 1, 0])

	tokensTwo = ['could', 'could', 'do', 'sentence']
	for word in tokensTwo:
		eq_(aNumericIndexerBuilder.can_encode(word), True)
	eq_(encode_many(aNumericIndexerBuilder, tokensTwo), [2, 2, 3, 1])

def test_ShouldBuildIndexer():
	aNumericIndexerBuilder = bc.NumericIndexerBuilder()
	tokens = ['any', 'sentence', 'any', 'could', 'could', 'do', 'sentence']
	indices = encode_many(aNumericIndexerBuilder, tokens)

	aNumericIndexer = aNumericIndexerBuilder.build()
	eq_(encode_many(aNumericIndexer, tokens), indices)
	eq_(decode_many(aNumericIndexer, indices), tokens)