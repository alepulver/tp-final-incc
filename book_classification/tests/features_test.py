import book_classification as bc
import os
from nose.tools import *

path_to_books = os.path.dirname(__file__)
aBookPath = os.path.join(path_to_books, "pg1465.txt")
aBook = bc.Book.from_file_path(aBookPath)

def test_TokenizerShouldProcessASentence():
	tokenizer = bc.BasicTokenizer()
	text = "This, I think; is a n1c3.sentence..."
	result = list(tokenizer.extract_from(text))
	eq_(result, ["this", "think", "sentence"])

def test_CanExtractFrequenciesFromText():
	text = "one two one three three four four three three"
	indexer = bc.NumericIndexer(["two", "three", "blah","one"])
	extractor = bc.WordFrequencyExtractor(bc.BasicTokenizer(), indexer)
	result = extractor.extract_from(text)
	eq_(result.as_dict(), {'three': 0.5714285714285714, 'one': 0.2857142857142857, 'two': 0.14285714285714285})
	eq_(len(result), 4)