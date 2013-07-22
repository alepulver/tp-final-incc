import book_classification as bc
from nose.tools import *

my_book_one = bc.Book.from_str("Title: Book One\nAuthor: A\nanimal plant animal plant")
my_book_two = bc.Book.from_str("Title: Book Two\nAuthor: B\ntower oil tower oil")
my_book_three = bc.Book.from_str("Title: Book Three\nAuthor: A\nanimal plant")
my_book_four = bc.Book.from_str("Title: Book Three\nAuthor: B\ntower oil")

def test_CanClassifyAPairOfBooks():
	training_set = bc.BookCollection({my_book_one, my_book_two})
	testing_set = bc.BookCollection({my_book_three, my_book_four})
	experiment = bc.Experiment(training_set, testing_set, bc.WordFrequencies)
	eq_(experiment.results(), {my_book_three: "A", my_book_four: "B"})