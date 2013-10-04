cdef extern void pairwise_entropies_window(double *output, long output_size, const long *words, long words_size, const double *weights)

def pairwise_entropy_window(double [:] output, long [:] words, double [:] factors):
    cdef double * output_i = &output[0]
    cdef const long * words_i = &words[0]
    cdef const double * factors_i = &factors[0]
    pairwise_entropies_window(output_i, len(output), words_i, len(words), factors_i)