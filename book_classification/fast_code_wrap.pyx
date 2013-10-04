cdef extern void pairwise_entropies_window(double *output_sum, double *output_sum_log, long output_size, const long *words, long words_size, const double *weights)

def pairwise_entropy_window(double [:] output_sum, double [:] output_sum_log, long [:] words, double [:] factors):
    cdef double * output_sum_i = &output_sum[0]
    cdef double * output_sum_log_i = &output_sum_log[0]
    cdef const long * words_i = &words[0]
    cdef const double * factors_i = &factors[0]
    pairwise_entropies_window(output_sum_i, output_sum_log_i, len(output_sum), words_i, len(words), factors_i)