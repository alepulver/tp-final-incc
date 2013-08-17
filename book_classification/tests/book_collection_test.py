import book_classification as bc
import os
from nose.tools import *

my_book_one = bc.Book.from_str("Title: Book One\nAuthor: A\nthe text")
my_book_two = bc.Book.from_str("Title: Book Two\nAuthor: A\nthe text")
my_book_three = bc.Book.from_str("Title: Book Three\nAuthor: A\nthe text")
my_book_four = bc.Book.from_str("Title: Book Four\nAuthor: B\nthe text")
my_book_five = bc.Book.from_str("Title: Book Five\nAuthor: B\nthe text")
my_book_six = bc.Book.from_str("Title: Book Six\nAuthor: C\nthe text")
my_books_all = {my_book_one, my_book_two, my_book_three, my_book_four, my_book_five, my_book_six}

def test_NewCollectionShouldContainBook():
	aBookCollection = bc.BookCollection.from_books({my_book_one})
	eq_(set(aBookCollection.books()), {my_book_one})
	ok_(my_book_one.author() in aBookCollection.authors())
	eq_(aBookCollection.books_by(my_book_one.author()), {my_book_one})
	#eq_(aBookCollection._books_by_author, {my_book_one.author(): {my_book_one}})

def test_NewCollectionShouldSeparateByAuthor():
	aBookCollection = bc.BookCollection.from_books([my_book_one, my_book_four])
	eq_(set(aBookCollection.books()), {my_book_one, my_book_four})
	eq_(aBookCollection._books_by_author,
		{my_book_one.author(): {my_book_one}, my_book_four.author(): {my_book_four}})

def test_BooksPreserveOrderInCollection():
	books = [my_book_one, my_book_two, my_book_three]
	aBookCollection = bc.BookCollection.from_books(books)
	eq_(list(aBookCollection.books()), books)

def test_ShouldFilterByAuthor():
	aBookCollection = bc.BookCollection.from_books(my_books_all)
	anotherBookCollection = aBookCollection.selection().exclude_authors_below(2)
	eq_(set(anotherBookCollection.authors()), {"A", "B"})

def test_ShouldPartitionByAuthorAndQuantity():
	aBookCollection = bc.BookCollection.from_books(my_books_all)
	anotherBookCollectionOne, anotherBookCollectionTwo = aBookCollection.selection().split_per_author_number(2)

	eq_(len(anotherBookCollectionOne.books_by("A")), 2)
	eq_(len(anotherBookCollectionOne.books_by("B")), 2)
	eq_(len(anotherBookCollectionOne.books_by("C")), 1)

	eq_(len(anotherBookCollectionTwo.books_by("A")), 1)
	ok_("B" not in anotherBookCollectionTwo.authors())
	ok_("C" not in anotherBookCollectionTwo.authors())