import book_classification as bc
from nose.tools import *

def test_BasicTokenizerShouldProcessASentence():
	tokenizer = bc.BasicTokenizer()
	text = "This, I think; is a n1c3.sentence..."
	result = list(tokenizer.tokens_from(text))
	eq_(result, ["this", "think", "sentence"])

def test_FilteringTokenizerShouldRestrictWords():
	tokenizer = bc.FilteringTokenizer(bc.BasicTokenizer(), ['two', 'three'])
	text = "one two one two three one two four"
	result = list(tokenizer.tokens_from(text))
	eq_(result, ["two", "two", "three", "two"])

def test_TokenizersCanRestrictVocabulary():
	raise NotImplementedError()

def test_GrouperCanGroupMultiplesOfSize():
	grouper = bc.BasicGrouper(3)
	result = grouper.parts_from('abcdef')
	eq_(list(result), [['a', 'b', 'c'], ['d', 'e', 'f']])

def test_GrouperCanGroupNonMultiplesOfSize():
	grouper = bc.BasicGrouper(3)
	result = grouper.parts_from('abcdefgh')
	eq_(list(result), [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h']])

@raises(Exception)
def test_WeightingWindowCantHaveEvenSize():
	result = bc.WeightingWindow.uniform(4)
	pass

"""
def test_WeightingWindowCanMakeUniform():
	result = bc.WeightingWindow.uniform(5)
	eq_(result, [])

def test_WeightingWindowCanMakeTriangular():
	result = bc.WeightingWindow.triangular(5, 2)
	eq_(result, [])

def test_WeightingWindowCanMakeGaussian():
	result = bc.WeightingWindow.uniform(5, 0, 3)
	eq_(result, [])

def test_WeightingWindowCantMakeCustomWithInvalidValues():
	result = bc.WeightingWindow.uniform(5, 0, 3)
	eq_(result, [])

def test_WeightingWindowCanMakeCustom():
	result = bc.WeightingWindow.uniform(5, 0, 3)
	eq_(result, [])
"""

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