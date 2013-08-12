import book_classification as bc
from nose.tools import *

def test_BasicTokenizerShouldProcessASentence():
	tokenizer = bc.BasicTokenizer()
	book = bc.DummyBook("This, I think; is a n1c3.sentence...")
	result = list(tokenizer.tokens_from(book))
	eq_(result, ["this", "think", "sentence"])

def test_FilteringTokenizerShouldRestrictWords():
	tokenizer = bc.FilteringTokenizer(bc.BasicTokenizer(), ['two', 'three'])
	book = bc.DummyBook("one two one two three one two four")
	result = list(tokenizer.tokens_from(book))
	eq_(result, ["two", "two", "three", "two"])

def test_CollapsingTokenizerShouldRestrictWords():
	tokenizer = bc.CollapsingTokenizer(bc.BasicTokenizer(), ['two', 'three'], 'blah')
	book = bc.DummyBook("one two one two three one two four")
	result = list(tokenizer.tokens_from(book))
	eq_(result, ["blah", "two", "blah", "two", "three", "blah", "two", "blah"])