import book_classification as bc
from scipy import sparse
from functools import reduce


class CollectionFeatures:
    def __init__(self, collection, collection_extractor, features_by_book):
        self._collection = collection
        self._collection_extractor = collection_extractor
        self._features_by_book = features_by_book

    def collection(self):
        return self._collection

    def by_book(self, book):
        return self._features_by_book[book]


class CollectionFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        result = {}
        for book in collection.books():
            result[book] = self._extractor.extract_from(book)
        return CollectionFeatures(collection, self, result)

    def encoder_for(self, collection, vocabulary=None):
        if vocabulary is None:
            input_vocabulary = bc.CollectionHierarchialFeatures.from_book_collection(
                collection, bc.VocabulariesExtractor(self._extractor._tokenizer)).total().keys()
            output_vocabulary = self._extractor.features_for_vocabulary(input_vocabulary)
        else:
            output_vocabulary = self._extractor.features_for_vocabulary(vocabulary)
        encoder = FeaturesEncoder(output_vocabulary)
        return CollectionFeaturesEncoder(encoder)


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

    def encoder_for(self, collection, vocabulary=None):
        raise NotImplementedError()


class FeaturesEncoder:
    def __init__(self, vocabulary):
        self._vocabulary = vocabulary
        self._numeric_indexer = bc.NumericIndexer(self._vocabulary)

    def encode(self, features):
        for k,v in features.items():
            if self._numeric_indexer.can_encode(k):
                yield self._numeric_indexer.encode(k), v

    def decode(self, items):
        for k,v in items:
            yield self._numeric_indexer.decode(k), v

    def vocabulary_size(self):
        return len(self._numeric_indexer)


class CollectionFeaturesEncoder:
    def __init__(self, features_encoder):
        self._features_encoder = features_encoder

    def encode(self, collection_features):
        num_rows = len(collection_features.collection())
        num_cols = self._features_encoder.vocabulary_size()
        matrix = sparse.dok_matrix((num_rows, num_cols))
        for i,book in enumerate(collection_features.collection().books()):
            features = collection_features.by_book(book)
            for j,v in self._features_encoder.encode(features):
                matrix[i, j] = v

        return matrix


class CollectionFeaturesMatrixExtractor:
    def __init__(self, extractor, base_collection, output_vocabulary=None):
        self._extractor = extractor
        self._training = base_collection
        self._output_vocabulary = output_vocabulary

        self._collection_features_extractor = CollectionFeaturesExtractor(self._extractor)
        self._collection_features_encoder = self._collection_features_extractor.encoder_for(
            self._training, self._output_vocabulary)

    def extract_from(self, collection):
        collection_features = self._collection_features_extractor.extract_from(collection)
        return self._collection_features_encoder.encode(collection_features)
