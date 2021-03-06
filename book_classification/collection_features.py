import book_classification as bc
from scipy import sparse
from functools import reduce
from collections import defaultdict


class CollectionFeatures:
    def __init__(self, collection, collection_extractor, features_by_book):
        self._collection = collection
        self._collection_extractor = collection_extractor
        self._features_by_book = features_by_book

    def collection(self):
        return self._collection

    def by_book(self, book):
        return self._features_by_book[book]

    def select(self, filter_pred):
        features_by_book = defaultdict(dict)
        for book, features in self._features_by_book.items():
            for k, v in features.items():
                if filter_pred(k):
                    features_by_book[book][k] = v

        return self.__class__(self._collection, self._collection_extractor, features_by_book)


class CollectionHierarchialFeatures:
    def __init__(self, extractor, by_book, by_author, total):
        self._extractor = extractor
        self._by_book = by_book
        self._by_author = by_author
        self._total = total

    def by_book(self, book):
        return self._by_book[book]

    def by_author(self, author):
        return self._by_author[author]

    def total(self):
        return self._total

    def authors_features(self):
        # XXX: merge with collectionextractor below, but use books since there is no collection of this

        vocabulary = set()
        for features in self._by_author.values():
            vocabulary.update(features.keys())
        encoder = bc.FeaturesEncoder(vocabulary)

        num_rows = len(self._by_author)
        num_cols = len(vocabulary)
        matrix = sparse.dok_matrix((num_rows, num_cols))

        for i, features in enumerate(self._by_author.values()):
            for j, v in encoder.encode(features):
                matrix[i, j] = v

        return matrix.tocsc()

    @classmethod
    def from_book_collection(cls, collection, extractor):
        features_by_book = {}
        for book in collection.books():
            features_by_book[book] = extractor.extract_from(book)
        features_by_author = {}
        for author in collection.authors():
            features = (features_by_book[book] for book in collection.books_by(author))
            features_by_author[author] = reduce(lambda x,y: x.combine(y), features)
        features_total = reduce(lambda x,y: x.combine(y), features_by_author.values())

        return cls(extractor, features_by_book, features_by_author, features_total)


class FeaturesEncoder:
    def __init__(self, vocabulary):
        self._vocabulary = vocabulary
        self._numeric_indexer = bc.NumericIndexer(self._vocabulary)

    def encode(self, features):
        for k, v in features.items():
            if self._numeric_indexer.can_encode(k):
                yield self._numeric_indexer.encode(k), v

    def decode(self, items):
        for k, v in items:
            yield self._numeric_indexer.decode(k), v

    def vocabulary(self):
        return self._numeric_indexer.vocabulary()


class CollectionFeaturesEncoder:
    def __init__(self, encoder):
        self._encoder = encoder

    def encode(self, features):
        num_rows = len(features.collection())
        num_cols = len(self._encoder.vocabulary())
        matrix = sparse.dok_matrix((num_rows, num_cols))

        for i, book in enumerate(features.collection().books()):
            book_features = features.by_book(book)
            for j, v in self._encoder.encode(book_features):
                matrix[i, j] = v

        return matrix.tocsc()

    def vocabulary(self):
        return self._features_encoder.vocabulary()


class CachedCollectionFeaturesEncoder:
    def __init__(self, encoder, cache=None):
        self._encoder = encoder
        self._cache = cache
        if self._cache is None:
            self._cache = {}

    def encode(self, features):
        num_cols = len(self._encoder.vocabulary())

        rows = []
        for book in features.collection().books():
            if book.title() not in self._cache:
                book_features = features.by_book(book)
                matrix_row = sparse.dok_matrix((1, num_cols))

                for j, v in self._encoder.encode(book_features):
                    matrix_row[0, j] = v

                self._cache[book.title()] = matrix_row.tocsr()

            rows.append(self._cache[book.title()])

        return sparse.vstack(rows, format='csr')

    def vocabulary(self):
        return self._features_encoder.vocabulary()


class CollectionFeaturesMatrixExtractor:
    def __init__(self, extractor, base_collection):
        self._extractor = bc.CollectionFeaturesExtractor(extractor)
        self._training = base_collection
        self._encoder = self._extractor.encoder_for(self._training)

    def extract_from(self, collection):
        features = self._extractor.extract_from(collection)
        return self._encoder.encode(features)
