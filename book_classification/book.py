import re
import zipfile
import gzip

class DummyBook:
    def __init__(self, text):
        self._text = text
    def contents(self):
        return self._text

class Book:
    def __init__(self, author, title, contents, source=None):
        self._author = author
        self._title = title
        self._contents = contents
        self._source = source

    def author(self):
        return self._author
    def title(self):
        return self._title
    def contents(self):
        return self._contents
    def source(self):
        return self._source

    @staticmethod
    def from_str(string, source=None):
        text = string

        try:
            author = re.search('Author:\s+(.+)', string).group(1).rstrip()
            title = re.search('Title:\s+(.+)', text).group(1).rstrip()
            return Book(author, title, text, source)

        except AttributeError:
            raise Exception("missing book information")

    @staticmethod
    def from_file_path(file_name):
        # TODO: partial/lazy read, much faster
        if file_name.endswith(".zip"):
            with zipfile.ZipFile(file_name) as zf:
                assert len(zf.namelist()) == 1
                with zf.open(zf.namelist()[0]) as f:
                    return Book.from_str(str(f.read(), 'utf-8').replace('\r\n', '\n'), file_name)
        elif file_name.endswith(".gz"):
            with gzip.open(file_name) as f:
                return Book.from_str(str(f.read(), 'utf-8').replace('\r\n', '\n'), file_name)
        elif file_name.endswith(".txt"):
        # XXX: or use default case?
            with open(file_name, "rU") as f:
                return Book.from_str(f.read(), file_name)
        else:
            raise Exception("unknown file extension for %s" % file_name)