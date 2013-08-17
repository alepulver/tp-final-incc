import book_classification as bc

book_1 = bc.Book("Myself", "How to model classification",
	"This is a text about how to classify books.")
book_2 = bc.Book("Someone", "Animals of the mountains",
	"A book about how animals survive in extreme environments.")
book_3 = bc.Book("Myself", "Another of my books",
	"To classify a book, try processing the text and doing some math.")
book_4 = bc.Book("Someone", "Animals of the ocean",
	"The best book describing how animals adapt to survive at the bottom of the sea.")

book_5 = bc.Book("Myself", "Yet one more book",
	"For cross-validation, more than two books are required here.")
book_6 = bc.Book("Someone", "Animals of the trees",
	"Monkeys lived in the trees, eating fruits and other dumber monkeys.")

trainingCollection = bc.BookCollection.from_books([book_1, book_2])
testingCollection = bc.BookCollection.from_books([book_4, book_3])
bigCollection = bc.BookCollection.from_books([book_1, book_2, book_3, book_4, book_5, book_6])