import book_classification as bc
from scipy import sparse
from functools import reduce
from collections import defaultdict
import multiprocessing


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


class ExtractorClosure:
    def __init__(self, extractor):
        self._extractor = extractor

    def __call__(self, book):
        return self._extractor.extract_from(book)


class CollectionFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        all_books = list(collection.books())
        pool = multiprocessing.Pool()

        func = ExtractorClosure(self._extractor)
        all_features = pool.map(func, all_books)

        result = {}
        for book, features in zip(all_books, all_features):
            result[book] = features
        return CollectionFeatures(collection, self, result)

    def encoder_for(self, collection):
        vocabulary = set()
        features = self.extract_from(collection)

        for book in features.collection().books():
            vocabulary.update(features.by_book(book).keys())

        encoder = FeaturesEncoder(vocabulary)
        return CollectionFeaturesEncoder(encoder)


class SerialCollectionFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        result = {}
        for book in collection.books():
            result[book] = self._extractor.extract_from(book)
        return CollectionFeatures(collection, self, result)

    def encoder_for(self, collection):
        vocabulary = set()
        features = self.extract_from(collection)

        for book in features.collection().books():
            vocabulary.update(features.by_book(book).keys())

        encoder = FeaturesEncoder(vocabulary)
        return CollectionFeaturesEncoder(encoder)


class CollectionFeaturesFilteringExtractor:
    def __init__(self, extractor, filter_predicate):
        self._extractor = extractor
        self._filter_predicate = filter_predicate

    def extract_from(self, collection):
        features = self._extractor.extract_from(collection)
        return features.select(self._filter_predicate)


class CollectionHierarchialFeatures:
    def __init__(self, by_book, by_author, total):
        self._by_book = by_book
        self._by_author = by_author
        self._total = total

    def by_book(self, book):
        return self._by_book[book]

    def by_author(self, author):
        return self._by_author[author]

    def total(self):
        return self._total

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

        return cls(features_by_book, features_by_author, features_total)


class CollectionHierarchialFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        return CollectionHierarchialFeatures.from_book_collection(collection, extractor)

    def encoder_for(self, collection):
        raise NotImplementedError()


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


class CollectionFeaturesMatrixExtractor:
    def __init__(self, extractor, base_collection):
        self._extractor = CollectionFeaturesExtractor(extractor)
        self._training = base_collection
        self._encoder = self._extractor.encoder_for(self._training)

    def extract_from(self, collection):
        features = self._extractor.extract_from(collection)
        return self._encoder.encode(features)
