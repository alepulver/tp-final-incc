import book_classification as bc


class SklExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def fit(self, books_list, y=None):
        collection = bc.BookCollection.from_books(books_list)
        self._matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
            self._extractor, collection)
        return self

    def transform(self, books_list):
        collection = bc.BookCollection.from_books(books_list)
        return self._matrix_extractor.extract_from(collection)


class SklModelAdapter:
    def __init__(self, model):
        self._model = model

    def fit(self, data, labels):
        self._indexer = bc.NumericIndexer(labels)
        return self._model.fit(data, self.encode_labels(labels))

    def predict(self, data):
        labels = self._model.predict(data)
        return self.decode_labels(labels)

    def encode_labels(self, labels):
        return [self._indexer.encode(x) for x in labels]

    def decode_labels(self, labels):
        return [self._indexer.decode(x) for x in labels]

    def __getattr__(self, attr):
        return getattr(attr, self._model)


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
