####### sHAM 


from heapq import heappush, heappop, heapify
import numpy as np
from joblib import Parallel, delayed
from sHAM import huffman
from sHAM import sparse_huffman
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
        int_data: list of integers representing the huffman encoding 
         of the vector data of the csc representation
        d_rev_data: dict encoded --> element
        row_index: vector of the row indices of the csc representation
        cum: vector of the number of elements of each column
        expected_c: number of columns in the matrix
        min_length_encoded: minimum length of huffman encodings
    """
    data, row_index, cum = sparse_huffman.convert_dense_to_csc(matr)
    d_data, d_rev_data  = huffman_sparse_encoded_dict(data)
    data_encoded = encoded_matrix(data, d_data, d_rev_data)
    int_data = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, bit_words_machine))
    expected_c = len(cum)
    matr_shape = matr.shape
    min_length_encoded = huffman.min_len_string_encoded(d_rev_data)
    return matr_shape, int_data, d_rev_data, row_index, cum, expected_c, min_length_encoded


def huffman_sparse_encoded_dict(data):
    """
    Generate dicts for huffman: element --> encoded and encoded --> element 
     for the vector data of the csc representation
    Args:
        data: vector of not zero elements of csc representation
    Returns:
        dict for  data
    """
    e_data = (huffman.encode(huffman.dict_elem_freq(data)))

    d_data = dict(e_data)
    d_rev_data = huffman.reverse_elements_list_to_dict(e_data)

    return d_data, d_rev_data

def encoded_matrix(data, d_data, d_rev_data):
    """
    Replaces the elements in the data vector of the csc representation
     with a vector containing the encodings instead of the elements
    Args:
        data: vector data of the csc representation
        d_data, d_rev_data: dict encoded --> element and element --> encoded
    Returns:
        coded vector data
    """
    data_encoded = huffman.matrix_with_code(data, d_data, d_rev_data)
    return data_encoded


def sparsed_encoded_to_dense(sparsed_shape, int_data, d_rev_data, row_index, cum, bits_for_element, expected_c, min_length_encoded=1):
    """
    Starting from the list of integers of of data vector (coded) and 
    from the vectors int_row and cum recreates the dense matrix
    Args:
        sparsed_shape: shape of the compressed matrix
        int_data: list of integers representing the data vector
         of the csc representation
        d_rev_data: dict encoded --> element
        bits_for_element: machine word bit number
        expected_c: number of columns in the original matrix
    Returns:
        output: expanded matrix ndarray
    """

    column = -1
    output = np.zeros((sparsed_shape), order='F')

    index_int_d = 0
    index_bit_d = 0
    last_int_decoded_d = "-1"

    current_c = cum[0]
    row_counter = 0
    cum_counter = 0

    for _ in range(expected_c):
        current_c = cum[cum_counter]
        column += 1
        cum_counter += 1
        for _ in range(current_c):
            current_d, index_int_d, index_bit_d, last_int_decoded_d = huffman.find_next(int_data, index_int_d, index_bit_d, d_rev_data, bits_for_element, last_int_decoded_d, min_length_encoded)
            current_r = row_index[row_counter]
            row_counter += 1
            output[current_r, column] = current_d
    return output


def sparsed_encoded_dot(input_x, sparsed_shape, int_data, d_rev_data, row_index, cum, bits_for_element, expected_c, output_type='float32', min_length_encoded=1):
    """
    Starting from the list of integers of of data vector (coded) and 
    from the vectors int_row and cum perform input_x dot compressed matrix
    Args:
        input_x: expanded matrix, left element of the dot
        sparsed_shape: shape of the compressed matrix
        int_data: list of integers representing the data vector
         of the csc representation
        d_rev_data: dict encoded --> element
        bits_for_element: number of bits used for an integer
        expected_c: number of columns in the original matrix
    Returns:
        output: matrix ndarray
    """

    column = -1
    output = np.zeros((input_x.shape[0],sparsed_shape[1]), order='F', dtype=output_type)
    input_x = np.asfortranarray(input_x)

    index_int_d = 0
    index_bit_d = 0
    last_int_decoded_d = "-1"

    current_c = cum[0]
    row_counter = 0
    cum_counter = 0

    for _ in range(expected_c):
        current_c = cum[cum_counter]
        column += 1
        cum_counter += 1
        for _ in range(current_c):
            current_d, index_int_d, index_bit_d, last_int_decoded_d = huffman.find_next(int_data, index_int_d, index_bit_d, d_rev_data, bits_for_element, last_int_decoded_d, min_length_encoded)
            current_r = row_index[row_counter]
            row_counter += 1
            output = huffman.mult_for_row(input_x, output, current_r, current_d, column)
    return output
