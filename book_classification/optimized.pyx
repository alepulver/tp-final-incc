#!python
#cython: boundscheck=False
#cython: wraparound=False

import cython
import numpy as np
cimport numpy as np
from collections import Counter


"""
cdef class BucketCounter:
    cdef int _width_bits
    cdef int _depth
    cdef int [:, :] buckets
    cdef int [:] fill
    cdef int _mask
    cdef int [:] counts

    cpdef __init__(self, int width_bits, int depth):
        assert(width > 0)
        assert(depth > 0)

        self._width_bits = width_bits
        self._depth = depth
        self._buckets = np.matrix((2**width_bits, depth))
        self._fill = np.zero(2**width_bits)
        self._mask = ~(2**width_bits - 1)

    cpdef add(self, int [:] values, int [:] weights):
        for i in range(len(values)):
            position = values[i] & self._mask
            self._buckets[position, self._fill[i]] = values[i]
            self._fill[i] += 1

    cpdef results(self):
        for i in range(self._buckets.shape[0]):
            pass
        return self._counts

    cpdef clear(self):
        for i in range(len(self._fill)):
            self._fill[i] = 0
"""


def pairwise_association_window(double [:] output, long [:] words, double [:] factors):
    cdef long size = len(output)
    cdef long center = words[len(words) // 2]
    cdef long i, index

    for i in range(len(words)):
        #index = (center*1664525 + words[i]) % size
        #output[index] += factors[i]

        index = (center*1664525 + words[i]) % (2*size)
        if index >= size:
            output[index-size] -= factors[i]
        else:
            output[index] += factors[i]


def pairwise_entropy_window(double [:] output, long [:] words, double [:] factors):
    cdef long i, index, pos
    cdef long size = len(output)
    cdef long center = words[len(words) // 2]

    cdef long [:] permutation = np.argsort(words)

    cdef long current = -1
    cdef double accumulated = 0

    for i in range(len(words)):
        pos = permutation[i]

        if words[pos] != current:
            index = (center*1664525 + current) % size
            output[index] += accumulated * np.log(accumulated)

            current = words[pos]
            accumulated = factors[pos]
        else:
            accumulated += factors[pos]

            if i == len(words):
                index = (center*1664525 + current) % size
                output[index] += accumulated * np.log(accumulated)
