import book_classification as bc
from nose.tools import *



def test_FeaturesToMatrixEncoderWords():
	aBookCollectionSample = aBookCollection.sample_books(5)
	tokenizer = bc.BasicTokenizer()
	fixed_extractor = bc.TruncatedExtractor.from_collection(
		bc.FrequenciesExtractor(tokenizer), aBookCollectionSample)
	encoder = bc.FeaturesToMatrixEncoder(fixed_extractor, aBookCollectionSample.authors())
	
	matrix = encoder.encode_features(aBookCollectionSample)
	eq_(hash(matrix.tocsc()), 8743168762645)

	authors = encoder.encode_authors(aBookCollectionSample)
	eq_(authors, [])