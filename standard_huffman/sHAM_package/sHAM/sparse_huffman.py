####### THIS VERSION USE HUFFMAN ON NOT ZERO ELEMENTS, ROW INDICES AND NUMBER OF VALUES FOR EACH COLUMN ##### NOT USED FOR PAPER


from heapq import heappush, heappop, heapify
import numpy as np
from joblib import Parallel, delayed
from sHAM import huffman
from scipy.sparse import csc_matrix
from numba import njit, prange

def do_all_for_me(matr, bit_words_machine):
    """
    It takes the matrix and calls all the functions necessary to compress it
    Args:
        matr: matrix to be compressed
        bit_words_machine: machine word bit number
    returns:
        matr_shape: shape of the matrix that we compress
        int_data, int_row_index, int_cum: lists of integers representing the huffman 
         coding of the vectors of the csc representation (cum contains, for each column,
          the number of non-zero values. usually a cumulative value is used)
        d_rev_data, d_rev_row_index, d_rev_cum: dicts encoded --> element
        expected_c: number of columns in the matrix
        min_length_encoded_d/r/c: minimum length of huffman encodings for each vector
    """
    data, row_index, cum = convert_dense_to_csc(matr)
    d_data, d_rev_data, d_row_index, d_rev_row_index, d_cum, d_rev_cum = huffman_sparse_encoded_dict(data, row_index, cum)
    data_encoded, row_index_encoded, cum_encoded = encoded_matrix(data, d_data, d_rev_data, row_index, d_row_index, d_rev_row_index, cum, d_cum, d_rev_cum)

    int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, bit_words_machine))
    int_row_index = huffman.convert_bin_to_int(huffman.make_words_list_to_int(row_index_encoded, bit_words_machine))
    int_cum = huffman.convert_bin_to_int(huffman.make_words_list_to_int(cum_encoded, bit_words_machine))

    expected_c = len(cum)
    matr_shape = matr.shape

    min_length_encoded_c = huffman.min_len_string_encoded(d_rev_cum)
    min_length_encoded_d = huffman.min_len_string_encoded(d_rev_data)
    min_length_encoded_r = huffman.min_len_string_encoded(d_rev_row_index)

    return matr_shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, expected_c, min_length_encoded_d, min_length_encoded_r, min_length_encoded_c

def convert_dense_to_csc(matrix):
    """
    Create 3 vectors of the matrix representation csc, data contains the non-zero
     elements, row_index represents the row indices where there are non-zero
     elements and cum contains the number of non-zero elements in each column
    Args:
        matrix: original matrix ndarray
    Returns:
        data, row_index, cum: vectors
    """
    csc = csc_matrix(matrix)
    data = csc.data
    row_index = csc.indices
    cum = np.empty_like(csc.indptr[1:])
    indptr = csc.indptr[1:]
    for i in range(cum.shape[0]):
        if i>0:
            cum[i] = indptr[i]-indptr[i-1]
        else:
            cum[i] = indptr[i]

    return data, row_index, cum

def csc_to_dense(matrix_shape, data, row_index, cum):
    """
    Starting from the three vectors of the csc representation,
     i generate the expanded matrix
    Args:
        matrix_shape: shape of original matrix
        data, row_index, cum: vectors of the csc representation
    Returns:
        output: expanded matrix
    """
    output = np.zeros(matrix_shape)
    data_pointer = 0
    row_index_pointer = 0
    column = 0
    for c in cum:
        if c == 0:
            column += 1
        else:
            for e in range(c):
                row = row_index[row_index_pointer]
                row_index_pointer += 1
                data_elem = data[data_pointer]
                data_pointer += 1
                output[row,column] = data_elem
            column += 1
    return output

@njit(parallel=True)
def dot_sparse_j(input_x, matrix_shape, data, row_index, cum):
    """
    performs the dot multiplication between input_x and the matrix represented
     by the three csc vectors
    Args:
        input_x: matrix to multiply (left element of dot)
        matrix_shape: shape of original matrix
        data, row_index, cum: vectors of the csc representation
    Returns:
        output: result of dot
    """
    output = np.zeros((input_x.shape[0],matrix_shape[1]))
    data_pointer = 0
    row_index_pointer = 0
    column = 0
    for c in cum:
        if c == 0:
            column += 1
        else:
            for e in range(c):
                row = row_index[row_index_pointer]
                row_index_pointer += 1
                data_elem = data[data_pointer]
                data_pointer += 1
                for i in prange(input_x.shape[0]):
                    output[i][column] += input_x[i][row]*data_elem
            column += 1
    return output

def huffman_sparse_encoded_dict(data, row_index, cum):
    """
    Generate dictionaries for huffman: element -> encoded and encoded --> element
     for each vector of the csc representation
    Args:
        data, row_index, cum: vectors of the csc representation
    Returns:
        dicts for vectors
    """
    e_data = (huffman.encode(huffman.dict_elem_freq(data)))
    e_row_index = (huffman.encode(huffman.dict_elem_freq(row_index)))
    e_cum = (huffman.encode(huffman.dict_elem_freq(cum)))

    d_data = dict(e_data)
    d_rev_data = huffman.reverse_elements_list_to_dict(e_data)
    d_row_index = dict(e_row_index)
    d_rev_row_index = huffman.reverse_elements_list_to_dict(e_row_index)
    d_cum = dict(e_cum)
    d_rev_cum = huffman.reverse_elements_list_to_dict(e_cum)

    return d_data, d_rev_data, d_row_index, d_rev_row_index, d_cum, d_rev_cum

def encoded_matrix(data, d_data, d_rev_data, row_index, d_row_index, d_rev_row_index, cum, d_cum, d_rev_cum):
    """
    Replaces the elements in the vectors of the csc representation with
     three vectors containing the encodings instead of the elements
    Args:
        data, row_index, cum: vectors of the csc representation
        d_data, d_rev_data: dicts for data
        d_row_index, d_rev_row_index: dicts for row_index
        d_cum, d_rev_cum: dicts for cum
    Returns:
        coded vectors
    """
    data_encoded = huffman.matrix_with_code(data, d_data, d_rev_data)
    row_index_encoded = huffman.matrix_with_code(row_index, d_row_index, d_rev_row_index)
    cum_encoded = huffman.matrix_with_code(cum, d_cum, d_rev_cum)
    return data_encoded, row_index_encoded, cum_encoded


def sparsed_encoded_to_dense(sparsed_shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, bits_for_element, expected_c):
    """
    Starting from three lists of integers (one for each vector of the csc
     representation) it recreates the dense matrix
    Args:
        sparsed_shape: shape of matrix
        int_data, int_row_index, int_cum: lists of integers representing 
         the vectors of the csc representation
        d_rev_data, d_rev_row_index, d_rev_cum: dicts encoded --> element
        bits_for_element: number of bits used for an integer
        expected_c: number of columns in the original matrix
    Returns:
        output: expanded matrix
    """
    len_int_row = len(int_row_index)
    len_data = len(int_data)
    if len_data > len_int_row:
        int_row_index += [0] * (len_data-len_int_row)
    if  len_int_row > len_data:
        int_data += [0] * (len_int_row-len_data)

    current_code_c = ""
    current_code_r = ""
    current_code_d = ""
    list_r = []
    list_d = []
    just_find_r = False
    just_find_d = False
    column = -1
    index_just_watched = 0
    output = np.zeros(sparsed_shape)
    for c in int_cum:
        encoded_text_c = huffman.int_to_bin_string(c, bits_for_element)
        for bit_c in encoded_text_c:
            if expected_c == 0:
                break
            else:
                current_code_c += bit_c
                if(current_code_c in d_rev_cum):
                    column += 1
                    current_c = d_rev_cum[current_code_c]
                    expected_c -= 1
                    expected = current_c
                    current_code_c = ""
                    if index_just_watched == len(int_data):
                        for elem_r, elem_d in zip(list_r[:expected], list_d[:expected]):
                            output[elem_r,column] = elem_d
                        list_r = list_r[expected:]
                        list_d = list_d[expected:]
                    else:
                        for d,r in zip(int_data[index_just_watched:],int_row_index[index_just_watched:]):
                            index_just_watched += 1
                            encoded_text_r = huffman.int_to_bin_string(r, bits_for_element)
                            encoded_text_d = huffman.int_to_bin_string(d, bits_for_element)
                            for bit_r, bit_d in zip(encoded_text_r,encoded_text_d):
                                if just_find_r and just_find_d:
                                    if expected != 0:
                                        expected -= 1
                                        output[list_r.pop(0), column] = list_d.pop(0)
                                        just_find_r = len(list_r) != 0
                                        just_find_d = len(list_d) != 0

                                current_code_r += bit_r
                                if(current_code_r in d_rev_row_index):
                                    just_find_r = True
                                    list_r += [d_rev_row_index[current_code_r]]
                                    current_code_r = ""

                                current_code_d += bit_d
                                if(current_code_d in d_rev_data):
                                    just_find_d = True
                                    list_d += [d_rev_data[current_code_d]]
                                    current_code_d = ""

    return output

def sparsed_encoded_dot(input_x, sparsed_shape, int_data, int_row_index, int_cum, d_rev_data, d_rev_row_index, d_rev_cum, bits_for_element, expected_c, output_type='float32', min_length_encoded_d=1, min_length_encoded_r=1, min_length_encoded_c=1):
    """
    Starting from three lists of integers (one for each vector of the
     csc representation) it executes input_x dot compressed matrix
    Args:
        input_x: expanded matrix, left element of the dot
        sparsed_shape: shape of the compressed matrix
        int_data, int_row_index, int_cum: lists of integers representing 
         the vectors of the csc representation
        d_rev_data, d_rev_row_index, d_rev_cum: dicts encoded --> element
        bits_for_element: number of bits used for an integer
        expected_c: number of columns in the original matrix
        min_length_encoded_d/r/c: minimum length of huffman encodings for each vector
    Returns:
        output: matrice ndarray
    """

    column = -1
    input_x = np.asfortranarray(input_x)
    output = np.zeros((input_x.shape[0],sparsed_shape[1]), order='F', dtype=output_type)

    index_int_c = index_int_d = index_int_r = 0
    index_bit_c = index_bit_d = index_bit_r = 0
    last_int_decoded_c = last_int_decoded_d = last_int_decoded_r = "-1"

    for _ in range(expected_c):
        current_c, index_int_c, index_bit_c, last_int_decoded_c = huffman.find_next(int_cum, index_int_c, index_bit_c, d_rev_cum, bits_for_element, last_int_decoded_c, min_length_encoded_c)
        column += 1
        for _ in range(current_c):
            current_d, index_int_d, index_bit_d, last_int_decoded_d = huffman.find_next(int_data, index_int_d, index_bit_d, d_rev_data, bits_for_element, last_int_decoded_d, min_length_encoded_d)
            current_r, index_int_r, index_bit_r, last_int_decoded_r = huffman.find_next(int_row_index, index_int_r, index_bit_r, d_rev_row_index, bits_for_element, last_int_decoded_r, min_length_encoded_r)
            output = huffman.mult_for_row(input_x, output, current_r, current_d, column)
    return output
