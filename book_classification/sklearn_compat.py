import book_classification as bc

class SklExtractor:
	def __init__(self, extractor, output_vocabulary=None):
		self._extractor = extractor
		self._output_vocaulary = output_vocabulary

	def fit(self, books_list, y=None):
		collection = bc.BookCollection.from_books(books_list)
		self._collection_matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
			self._extractor, collection, self._output_vocaulary)
		return self

	def transform(self, books_list):
		collection = bc.BookCollection.from_books(books_list)
		return self._collection_matrix_extractor.extract_from(collection)

class SklPipelineObserver:
	def __init__(self, name):
		self._name = name
	def fit(self, books_list, y):
		print("%s fitting:" % self._name)
		print(books_list)
		print("with:")
		print(y)
		print("")
		return self
	def transform(self, data):
		print("%s transformed:" % self._name)
		print(data)
		print("")
		return data
	def predict_(self, books_list):
		print("%s predicting:" % self._name)
		print(books_list)
		return books_list