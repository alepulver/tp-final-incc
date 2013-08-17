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
	def __init__(self):
		pass
	def fit(self, books_list, y):
		pass
	def transform(self, books_list):
		pass