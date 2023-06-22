import unittest
import numpy as np
from sHAM import huffman

class HuffmanTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(HuffmanTest, self).__init__(*args, **kwargs)
        self.input_x = np.random.randint(1000, size=(1000,500))
        self.matr = np.random.randint(500, size=(500,250))

        symb2freq = huffman.dict_elem_freq(self.matr)
        e = huffman.encode(symb2freq)

        self.d_rev = huffman.reverse_elements_list_to_dict(e)
        self.d = dict(e)

        self.encoded = huffman.matrix_with_code(self.matr, self.d, self.d_rev)

        self.bit_words_machine = 64
        self.list_bin = huffman.make_words_list_to_int(self.encoded, self.bit_words_machine)

        self.min_length_encoded = huffman.min_len_string_encoded(self.d_rev)

    def test_encode_decode(self):
        #faccio prima encoding, poi decoding e verifico che questi due passaggi restituiscono una matrice come quella iniziale

        decoded = huffman.decode_matrix(self.matr, self.encoded, self.d_rev)
        self.assertTrue(np.all(decoded == self.matr))


    def test_len_bin_string(self):
        #testo che la lunghezza di ogni elemento della lista di interi sia lunga bit

        assertion = True
        for elem in self.list_bin:
            if len(elem) == self.bit_words_machine:
                assertion = bool(assertion*True)
            else:
                assertion = False
                break
        self.assertTrue(assertion)

    def test_dot_encode(self):
        #confronto la dot con huffman con la dot di numpy

        int_from_string = huffman.convert_bin_to_int(self.list_bin) #creo lista di interi di 64 bit
        dot_encode = huffman.dot_for_col(self.input_x, int_from_string, self.matr, self.d_rev, self.bit_words_machine, self.matr.dtype, self.min_length_encoded)
        numpy_dot = np.dot(self.input_x, self.matr)
        self.assertTrue(np.all(numpy_dot == dot_encode))


if __name__ == '__main__':
    unittest.main()
