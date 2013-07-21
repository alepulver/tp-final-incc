import book_classification as bc
import os
from nose.tools import *

path_to_books = os.path.dirname(__file__)
aBookPath = os.path.join(path_to_books, "pg1465.txt")
aBook = bc.Book.from_file_path(aBookPath)

def test_WordFrequenciesShouldMatch():
	aBookFeatures = bc.WordFrequencies(aBook).features()
	eq_(len(aBookFeatures), 2548)
	ok_({"hanging", "positive", "disclaim"} < aBookFeatures.keys())
	# TODO: check that top-5 and bottom-5 match

def test_WordEntropiesShouldMatch():
	pass