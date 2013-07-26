import book_classification as bc
import shelve
from nose.tools import *

my_book_one = bc.Book.from_str("Title: Book One\nAuthor: A\nanimal plant animal plant")
my_book_two = bc.Book.from_str("Title: Book Two\nAuthor: B\ntower oil tower oil")
my_book_three = bc.Book.from_str("Title: Book Three\nAuthor: A\nanimal plant")
my_book_four = bc.Book.from_str("Title: Book Three\nAuthor: B\ntower oil")

my_shelve = shelve.open("storage.db")
aBookCollection = my_shelve['aBookCollection']

def test_CanClassifyAPairOfBooks():
	training_set = bc.BookCollection({my_book_one, my_book_two})
	testing_set = bc.BookCollection({my_book_three, my_book_four})
	experiment = bc.Experiment(training_set, testing_set, bc.WordFrequencies)
	eq_(experiment.results(), {my_book_three: "A", my_book_four: "B"})

def test_CanLoadTestCollection():
	eq_(len(aBookCollection), 1103)

def test_CanClassifyManyBooks():
	subCollection = aBookCollection.only_authors_with_or_more_than(5).sample_authors(10)
	train, test = subCollection.separate_by_at_most_per_author(3)
	# TODO: add method sample_books_within_authors
	test, _ = test.separate_by_at_most_per_author(2)
	eq_(len(train), 30)
	eq_(len(test), 20)

	experiment = bc.Experiment(train, test, bc.WordFrequencies)
	eq_(experiment.results(), {})