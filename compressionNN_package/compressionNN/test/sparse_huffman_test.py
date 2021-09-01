import unittest
import numpy as np
from compressionNN import huffman
from compressionNN import sparse_huffman

class SparseHuffmanTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SparseHuffmanTest, self).__init__(*args, **kwargs)
        n=500
        m=100
        self.input_x = np.random.randint(1000, size=(70,n))
        p = 0.7 #probablità che sia "prunato"
        mask = np.random.choice(a=[False, True], size=(n,m), p=[p, 1-p])
        self.matr = np.random.randint(500, size=(n,m))*(1*mask)

        self.data, self.row_index, self.cum = sparse_huffman.convert_dense_to_csc(self.matr)

        self.d_data, self.d_rev_data, self.d_row_index, self.d_rev_row_index, self.d_cum, self.d_rev_cum = sparse_huffman.huffman_sparse_encoded_dict(self.data, self.row_index, self.cum)

        data_encoded, row_index_encoded, cum_encoded = sparse_huffman.encoded_matrix(self.data, self.d_data, self.d_rev_data, self.row_index, self.d_row_index, self.d_rev_row_index, self.cum, self.d_cum, self.d_rev_cum)

        self.bit_words_machine = 64
        self.int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, self.bit_words_machine))
        self.int_row_index = huffman.convert_bin_to_int(huffman.make_words_list_to_int(row_index_encoded, self.bit_words_machine))
        self.int_cum = huffman.convert_bin_to_int(huffman.make_words_list_to_int(cum_encoded, self.bit_words_machine))

        self.expected_c = len(self.cum)

        self.min_length_encoded_c = huffman.min_len_string_encoded(self.d_rev_cum)
        self.min_length_encoded_d = huffman.min_len_string_encoded(self.d_rev_data)
        self.min_length_encoded_r = huffman.min_len_string_encoded(self.d_rev_row_index)

    def test_csc_to_dense(self):
        #rendo densa la matrice sparsa e controllo che il risultato sia lo stesso di prima della rappresentazione csc
        dense = sparse_huffman.csc_to_dense(self.matr.shape, self.data, self.row_index, self.cum)
        self.assertTrue(np.all(dense == self.matr))

    def test_dot_dense_sparse(self):
        #verifico che la dot tra una matrice densa e una matrice sparsa funzioni correttamente
        dot_sparse = sparse_huffman.dot_sparse_j(self.input_x, self.matr.shape, self.data, self.row_index, self.cum)
        self.assertTrue(np.all(dot_sparse == np.dot(self.input_x,self.matr)))

    def test_code_decode_with_zeros_columns(self):
        n=50
        m=10
        p = 0.7 #probablità che sia "prunato"
        mask = np.random.choice(a=[False, True], size=(n,m), p=[p, 1-p])
        matr = np.random.randint(100, size=(n,m))*(1*mask)
        matr[:,3] = 0
        matr[:,6] = 0

        data, row_index, cum = sparse_huffman.convert_dense_to_csc(matr)
        d_data, d_rev_data, d_row_index, d_rev_row_index, d_cum, d_rev_cum = sparse_huffman.huffman_sparse_encoded_dict(data, row_index, cum)
        data_encoded, row_index_encoded, cum_encoded = sparse_huffman.encoded_matrix(data, d_data, d_rev_data, row_index, d_row_index, d_rev_row_index, cum, d_cum, d_rev_cum)
        int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, self.bit_words_machine))
        int_row_index = huffman.convert_bin_to_int(huffman.make_words_list_to_int(row_index_encoded, self.bit_words_machine))
        int_cum = huffman.convert_bin_to_int(huffman.make_words_list_to_int(cum_encoded, self.bit_words_machine))
        expected_c = len(cum)
        dense = sparse_huffman.sparsed_encoded_to_dense(matr.shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, self.bit_words_machine, expected_c)
        self.assertTrue(np.all(dense == matr))


    def test_encoded_csc_to_dense(self):
        #rendo densa la matrice sparsa codificata con huffman e controllo che il risultato sia lo stesso di prima della rappresentazione csc
        dense = sparse_huffman.sparsed_encoded_to_dense(self.matr.shape, self.int_data, self.int_row_index, self.int_cum, self.d_rev_data, self.d_rev_row_index, self.d_rev_cum, self.bit_words_machine, self.expected_c)
        self.assertTrue(np.all(dense == self.matr))

    def test_dot_dense_sparsed_encoded(self):
        dot_sparse = sparse_huffman.sparsed_encoded_dot(self.input_x, self.matr.shape, self.int_data, self.int_row_index, self.int_cum, self.d_rev_data, self.d_rev_row_index, self.d_rev_cum, self.bit_words_machine, self.expected_c, self.matr.dtype, self.min_length_encoded_d, self.min_length_encoded_r, self.min_length_encoded_c)
        self.assertTrue(np.all(dot_sparse == np.dot(self.input_x,self.matr)))

if __name__ == '__main__':
    unittest.main()
