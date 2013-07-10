import re
import zipfile
import gzip

class TextExtractor:
	# add from_file_path cases under hierarchy here
	pass

class Book:
	"""
	...
	"""

	@staticmethod
	def from_file_path(file_name):
		if file_name.endswith(".zip"):
		    with zipfile.ZipFile(file_name) as zf:
		        assert len(zf.namelist()) == 1
		        with zf.open(zf.namelist()[0]) as f:
		            return Book.from_str(str(f.read(), 'utf-8').replace('\r\n', '\n'))
		elif file_name.endswith(".gz"):
		    with gzip.open(file_name) as f:
		        return Book.from_str(str(f.read(), 'utf-8').replace('\r\n', '\n'))
		elif file_name.endswith(".txt"):
			# XXX: or use default case?
		    with open(file_name, "rU") as f:
		        return Book.from_str(f.read())
		else:
		    raise Exception("unknown file extension for %s" % file_name)

	@staticmethod
	def from_str(string):
		text = string

		try:
			author = re.search('Author:\s+(.+)', string).group(1).rstrip()
			title = re.search('Title:\s+(.+)', text).group(1).rstrip()
			return Book(author, title, text)
			
		except AttributeError:
			raise Exception("missing book information")
	
	def __init__(self, author, title, contents):
		self.author = author
		self.title = title
		self.contents = contents