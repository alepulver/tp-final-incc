from book_classification import book, book_collection
import os
from nose.tools import *

my_book_one = book.Book.from_str("Title: Book One\nAuthor: A\nthe text")
my_book_two = book.Book.from_str("Title: Book Two\nAuthor: A\nthe text")
my_book_three = book.Book.from_str("Title: Book Three\nAuthor: A\nthe text")
my_book_four = book.Book.from_str("Title: Book Four\nAuthor: B\nthe text")
my_book_five = book.Book.from_str("Title: Book Five\nAuthor: B\nthe text")
my_book_six = book.Book.from_str("Title: Book Six\nAuthor: C\nthe text")
my_books_all = {my_book_one, my_book_two, my_book_three, my_book_four, my_book_five, my_book_six}

def test_NewCollectionShouldContainBook():
	aBookCollection = book_collection.BookCollection({my_book_one})
	eq_(aBookCollection.books, {my_book_one})
	eq_(aBookCollection.authors, {my_book_one.author: {my_book_one}})

def test_NewCollectionShouldSeparateByAuthor():
	aBookCollection = book_collection.BookCollection({my_book_one, my_book_four})
	eq_(aBookCollection.books, {my_book_one, my_book_four})
	eq_(aBookCollection.authors, {my_book_one.author: {my_book_one}, my_book_four.author: {my_book_four}})

def test_ShouldFilterByAuthor():
	aBookCollection = book_collection.BookCollection(my_books_all)
	anotherBookCollection = aBookCollection.only_authors_with_or_more_than(2)
	eq_(anotherBookCollection.authors.keys(), {"A", "B"})

def test_ShouldPartitionByAuthorAndQuantity():
	aBookCollection = book_collection.BookCollection(my_books_all)
	anotherBookCollectionOne, anotherBookCollectionTwo = aBookCollection.separate_by_at_most_per_author(2)

	eq_(len(anotherBookCollectionOne.authors["A"]), 2)
	eq_(len(anotherBookCollectionOne.authors["B"]), 2)
	eq_(len(anotherBookCollectionOne.authors["C"]), 1)

	eq_(len(anotherBookCollectionTwo.authors["A"]), 1)
	eq_(len(anotherBookCollectionTwo.authors.get("B", set())), 0)
	eq_(len(anotherBookCollectionTwo.authors.get("B", set())), 0)