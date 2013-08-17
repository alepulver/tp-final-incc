import book_classification as bc

class SklExtractor:
	def __init__(self, extractor, output_vocabulary=None):
		self._extractor = extractor
		self._output_vocaulary = output_vocabulary

	def fit(self, collection):
		self._collection_matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
			self._extractor, collection, self._output_vocaulary)

	def transform(self, collection):
		return self._collection_matrix_extractor.extract_from(collection)

# is it really needed? can authors be passed around?
class SklAuthorDecoder:
	def __init__(self):
		pass
	def fit(self, books_list):
		pass
	def transform(self, books_list):
		pass

class SklPipelineObserver:
	def __init__(self):
		pass
	def fit(self, books_list):
		pass
	def transform(self, books_list):
		pass