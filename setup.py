from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages, extension
from Cython.Distutils import build_ext

ext_sources = ["book_classification/fast_code.cpp", "book_classification/fast_code_wrap.pyx"]
ext_modules = [
    extension.Extension("fast_code", ext_sources, extra_compile_args=['-O3', '-std=c++11'])
]

setup(
    name = "Book Classification",
    version = "0.1",
    packages = find_packages(),
    #scripts = ['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = [
        'docutils>=0.3',
        'nltk>=3.0a2',
        'scikit-learn>=0.14',
        'pandas',
        'pyhashxx',
        'cython'],
    tests_require = ['nose>=1.0'],
    # package_data = {
    #     # If any package contains *.txt or *.rst files, include them:
    #     '': ['*.txt', '*.rst'],
    #     # And include any *.msg files found in the 'hello' package, too:
    #     'hello': ['*.msg'],
    # },

    # metadata for upload to PyPI
    author = "Alejandro Pulver",
    author_email = "alepulver@gmail.com",
    description = "Learn and predict book authors from words using supervised learning",
    license = "BSD",
    keywords = "",
    url = "https://github.com/alepulver/tp-final-incc",

    # could also include long_description, download_url, classifiers, etc.

    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
)
