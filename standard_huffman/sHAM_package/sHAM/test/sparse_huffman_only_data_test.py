import unittest
import numpy as np
from sHAM import huffman
from sHAM import sparse_huffman
from sHAM import sparse_huffman_only_data

class SparseHuffmanOnlyDataTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SparseHuffmanOnlyDataTest, self).__init__(*args, **kwargs)
        n=500
        m=100
        self.input_x = np.random.randint(1000, size=(70,n))
        p = 0.7 #probablità che sia "prunato"
        mask = np.random.choice(a=[False, True], size=(n,m), p=[p, 1-p])
        self.matr = np.random.randint(500, size=(n,m))*(1*mask)

        self.data, self.row_index, self.cum = sparse_huffman.convert_dense_to_csc(self.matr)

        self.d_data, self.d_rev_data  = sparse_huffman_only_data.huffman_sparse_encoded_dict(self.data)

        data_encoded = sparse_huffman_only_data.encoded_matrix(self.data, self.d_data, self.d_rev_data)

        self.bit_words_machine = 64
        self.int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, self.bit_words_machine))

        self.expected_c = len(self.cum)

        self.min_length_encoded = huffman.min_len_string_encoded(self.d_rev_data)



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
        d_data, d_rev_data = sparse_huffman_only_data.huffman_sparse_encoded_dict(data)
        data_encoded = sparse_huffman_only_data.encoded_matrix(data, d_data, d_rev_data)
        int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, self.bit_words_machine))
        expected_c = len(cum)
        dense = sparse_huffman_only_data.sparsed_encoded_to_dense(matr.shape, int_data, d_rev_data, row_index, cum, self.bit_words_machine, expected_c)
        self.assertTrue(np.all(dense == matr))


    def test_encoded_csc_to_dense(self):
        #rendo densa la matrice sparsa codificata con huffman e controllo che il risultato sia lo stesso di prima della rappresentazione csc
        dense = sparse_huffman_only_data.sparsed_encoded_to_dense(self.matr.shape, self.int_data, self.d_rev_data, self.row_index, self.cum, self.bit_words_machine, self.expected_c, self.min_length_encoded)
        self.assertTrue(np.all(dense == self.matr))

    def test_dot_dense_sparsed_encoded(self):
        dot_sparse = sparse_huffman_only_data.sparsed_encoded_dot(self.input_x, self.matr.shape, self.int_data, self.d_rev_data, self.row_index, self.cum, self.bit_words_machine, self.expected_c, self.matr.dtype, self.min_length_encoded)
        self.assertTrue(np.all(dot_sparse == np.dot(self.input_x,self.matr)))

if __name__ == '__main__':
    unittest.main()
