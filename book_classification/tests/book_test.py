from book_classification import book
import os
from nose.tools import *

path_to_books = os.path.dirname(__file__)

def equals_mybook(aBook):
	aBookPath = os.path.join(path_to_books, "pg1465.txt")
	contents = open(aBookPath, "rU").read()

	eq_(aBook.author(), "Charles Dickens")
	eq_(aBook.title(), "The Wreck of the Golden Mary")
	eq_(aBook.contents(), contents)

def test_ShouldCreateABookFromFilePath():
	aBookPath = os.path.join(path_to_books, "pg1465.txt")
	aBook = book.Book.from_file_path(aBookPath)
	equals_mybook(aBook)

@raises(Exception)
def test_ShouldFailIfAuthorIsMissing():
	aBookPath = os.path.join(path_to_books, "pg1465_noauthor.txt")
	aBook = book.Book.from_file_path(aBookPath)

def test_ShouldCreateABookFromGzip():
	aBookPathGzip = os.path.join(path_to_books, "pg1465.txt.gz")
	aBook = book.Book.from_file_path(aBookPathGzip)
	equals_mybook(aBook)

def test_ShouldCreateABookFromZip():
	aBookPathZip = os.path.join(path_to_books, "pg1465.zip")
	aBook = book.Book.from_file_path(aBookPathZip)
	equals_mybook(aBook)

def test_BookFromFileShouldHavePath():
	aBookPath = os.path.join(path_to_books, "pg1465.txt")
	aBook = book.Book.from_file_path(aBookPath)
	eq_(aBook.source(), aBookPath)