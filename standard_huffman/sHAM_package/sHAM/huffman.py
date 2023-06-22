from heapq import heappush, heappop, heapify
import numpy as np
from joblib import Parallel, delayed
from numba import njit, jit

def do_all_for_me(matr, bit_words_machine):
    """
    It takes the matrix and calls all the functions necessary to compress it
    Args:
        matr: matrix to be compressed
        bit_words_machine: machine word bit number
    returns:
        int_from_string: list of integers representing the elements encoded
         with huffman
        d_rev: dict encoded --> element
        min_length_encoded: minimum length of huffman encodings
    """
    symb2freq = dict_elem_freq(matr)
    e = encode(symb2freq)

    d_rev = reverse_elements_list_to_dict(e)
    d = dict(e)

    encoded = matrix_with_code(matr, d, d_rev)
    list_bin = make_words_list_to_int(encoded, bit_words_machine)
    min_length_encoded = min_len_string_encoded(d_rev)
    int_from_string = convert_bin_to_int(list_bin)

    return int_from_string, matr, d_rev, bit_words_machine, matr.dtype, min_length_encoded

def encode(symb2freq):
    """
    Calculate Huffman encoding
    Args:
        symb2freq: dict element --> frequency
    Returns:
        list of tuples [ (element, huffman_element), ... ]
    """
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def reverse_elements_list_to_dict(output_encoder):
    """
    Args:
        output_encoder: list of tuples [ (element, encoded), ... ]
    Returns:
        dict encoded --> element
    """
    reverse = []
    for a,b in output_encoder:
        reverse += [[b,a]]
    return dict(reverse)

def dict_elem_freq(matrix):
    """
    Calculate frequency of input matrix elements
    Args:
        matrix: ndarray numpy
    Returns:
        dict element --> frequency
    """
    elem, freq = np.unique(matrix, return_counts=True)
    symb2freq = dict(zip(elem,freq))
    return symb2freq

def max_len_string_encoded(d_rev):
    """
    Calculate maximum length of huffman encodings
    Args:
        d_rev: dict encoded --> element
    Returns:
        maximum number of characters
    """
    maxs = 0
    for x in d_rev:
        len_bit = len(x)
        if(len_bit > maxs):
            maxs = len_bit
    return maxs

def min_len_string_encoded(d_rev):
    """
    Calculate minimum length of huffman encodings
    Args:
        d_rev: dict encoded --> element
    Returns:
        minimum number of characters
    """
    min = 1e10
    for x in d_rev:
        len_bit = len(x)
        if(len_bit < min):
            min = len_bit
    return min

def find_next(int_, index_int, index_bit, d_rev, bits_for_element, last_int_decoded, min_length_encoded=1):
    """
    Find the next item in the list of integers that represents
    the various weights encoded with huffman
    Args:
        int_: list of integer representing one or more weights 
         encoded with huffman
        index_int: pointer indicating the position where 
         we are in int_
        index_bit: pointer indicating which bits of the integer
         we are expanding
        d_rev: dict encoded --> element
        bits_for_element: number of bits used for an integer
        last_int_decoded: last integer transformed 
         into binary string analyzed
        min_length_encoded: minimum length of the huffman encoding
         (allows to scroll the integer transformed into a string
          in several bits at a time)
    Returns:
        Next element
    """
    current_code = ""
    while True:
        if last_int_decoded == "-1":
            encoded_text = int_to_bin_string(int_[index_int], bits_for_element)
            last_int_decoded = encoded_text
            index_int += 1
        else:
            encoded_text = last_int_decoded[index_bit:]
        if current_code == "":
            current_code += encoded_text[:min_length_encoded]
            encoded_text = encoded_text[min_length_encoded:]
            index_bit += len(current_code)
            if current_code in d_rev:
                current = d_rev[current_code]
                current_code = ""
        if current_code != "":
            for bit in encoded_text:
                current_code += bit
                index_bit += 1
                if(current_code in d_rev):
                    current = d_rev[current_code]
                    current_code = ""
                    break
        if current_code == "":
            if index_bit == bits_for_element:
                index_bit = 0
                last_int_decoded = "-1"
            break
        else:
            index_bit = 0
            last_int_decoded = "-1"
    return current, index_int, index_bit, last_int_decoded

def matrix_with_code(matrix, d, d_rev):
    """
    Create matrix like original with encodings instead of values
    Args:
        matrix: original matrix ndarray
        d: dict element --> encoded
        d_rev: dict encoded --> element
    Returns:
        matrix ndarray
    """
    
    def multiprocessing_func(matrix, n, d, k):
        n[matrix == k] = d[k]

    num_bits = max_len_string_encoded(d_rev)
    n = np.ndarray(matrix.shape, dtype='U{}'.format(num_bits))

    Parallel(n_jobs=-1, require='sharedmem')(delayed(multiprocessing_func)(matrix, n, d, k) for k in d)

    return n

def decode_matrix(matrix, encoded_matrix, d_rev):
    """
    Create matrix like the original starting from the matrix containing 
     the encodings instead of the elements
    Args:
        matrix: original matrix ndarray
        encoded_matrix: matrix with encodings ndarray
        d_rev: dict encoded --> element
    Returns:
        matrix ndarray
    """
    def multiprocessing_func(matrix, n, d, k):
        n[matrix == k] = d[k]

    n1 = np.empty_like(matrix)
    Parallel(n_jobs=-1, require='sharedmem')(delayed(multiprocessing_func)(encoded_matrix, n1, d_rev, k) for k in d_rev)
    return n1

def make_words_list_to_int(matrix_with_code, bit_words_machine):
    """
    Starting from the matrix with the encodings, column by column,
     create a list of long strings bit_words_machine
    Args:
        matrix_with_code: matrix with encodings ndarray
        bit_words_machine: machine word bit number
    Returns:
        list of strings of length bit_words_machine
    """
    bit = bit_words_machine
    list_string_col =[]
    string = ''
    for x in np.nditer(matrix_with_code, order='F'):
        string += (np.array2string(x)[1:-1])
        if len(string) > bit:
            bit_overflow = len(string)%bit
            list_string_col += [string[:-bit_overflow]]
            string = string[-bit_overflow:]
        elif len(string) == bit:
            list_string_col += [string]
            string = ''
    bit_remain = len(string)
    if bit_remain > 0:
        string += "0"*(bit-bit_remain) #padding di 0 per renderla lunga bit
        list_string_col += [string]
    return list_string_col

def convert_bin_to_int(list_string_col):
    """
    Convert a list of strings to a list of integers
    Args:
        list_string_col: list of strings (characters 0 or 1)
    Returns:
        list of integer
    """
    int_from_string_col = [int(x, 2) for x in list_string_col]
    return int_from_string_col

def int_to_bin_string(x, bits_for_element):
    """
    Convert an integer to a binary string and put initial padding
      to make it long bits_for_element
        x: integer
        bit_for_element: bit length of machine words
    Returns:
        string
    """
    encoded_text = "{0:b}".format(x)
    len_bin = len(encoded_text)
    if len_bin < bits_for_element: #aggiungo gli 0 iniziali che perdo trasformando in int
        encoded_text = "0"*(bits_for_element-len_bin)+encoded_text
    return encoded_text

@njit(["float32[:,:](float32[:,:], float32[:,:], int32, float32, int32)","int64[:,:](int64[:,:], int64[:,:], int32, int64, int32)"],nogil=True, fastmath=True, cache=True)
def mult_for_row(input_x, output, current_row, current_d, column):
    for i in range(input_x.shape[0]):
        output[i][column] += input_x[i][current_row]*current_d
    return output

def dot_for_col(input_x, int_from_string, matrix, d_rev, bits_for_element, output_type='float32', min_length_encoded=1):
    """
    Multiplies input_x dot matrix coded in list of integers.
     Reassemble matrix column by column, when i find an element
      i go to accumulate the product to calculate the dot
    Args:
        input_x: expanded matrix ndarray
        int_from_string: list of integers
        matrix: original matrix ndarray
        d_rev: dict encoded --> element
        bit_for_element: number of bits used for an integer
    Returns:
        matrix ndarray
    """

    output = np.zeros((input_x.shape[0], matrix.shape[1]), order='F', dtype=output_type)
    input_x = np.asfortranarray(input_x)

    index_int = 0
    index_bit = 0
    last_int_decoded = "-1"
    expected_elements = matrix.shape[0] * matrix.shape[1]
    row = 0
    column = 0

    for _ in range(expected_elements):
        current, index_int, index_bit, last_int_decoded = find_next(int_from_string, index_int, index_bit, d_rev, bits_for_element, last_int_decoded, min_length_encoded)
        if current != 0:
            output = mult_for_row(input_x, output, row, current, column)
        row += 1
        if row >= matrix.shape[0]:
            column += 1
            row = 0
    return output


############# FASTER VERSION WITH GENERATE WEIGHTS COLUMN AND USE NUMPY.DOT

#     decoded_text = ""
#     weights = np.ndarray((matrix.shape[0],), dtype=matrix.dtype)
#     output = np.ndarray((input_x.shape[0],matrix.shape[1]), dtype=matrix.dtype)
#     expected_elements = matrix.shape[0] * matrix.shape[1]
#     elements = 0
#     row = 0
#     column = 0
#
#     for x in int_from_string:
#         encoded_text = int_to_bin_string(x, bits_for_element)
#         for bit in encoded_text:
#             if expected_elements == 0:
#                 break
#             current_code += bit
#             if(current_code in d_rev):
#                 weights[row,] = d_rev[current_code]
#                 current_code = ""
#                 expected_elements -= 1
#                 row += 1
#                 if row >= matrix.shape[0]:
#                     output[:,column] = np.dot(input_x,weights)
#                     column += 1
#                     row = 0
#
#     return output


############# MULTICORE VERSION WITHOUT GENERATE WEIGHTS COLUMN
##### STILL SLOWER THAN ORIGINAL dot_for_col

# def dot_for_col(input_x, int_from_string, matrix, d_rev, bits_for_element):
#     @njit
#     def mult_for_row(input_x, output, row, elem, column):
#         for i in range(input_x.shape[0]):
#             output[i][column] += input_x[i][row]*elem
#         return output
#
#
#     current_code = ""
#     decoded_text = ""
#     weights = np.ndarray((matrix.shape[0],), dtype=matrix.dtype)
#     output = np.zeros((input_x.shape[0],matrix.shape[1]), dtype=matrix.dtype)
#     expected_elements = matrix.shape[0] * matrix.shape[1]
#     elements = 0
#     row = 0
#     column = 0
#
#     for x in int_from_string:
#         encoded_text = "{0:b}".format(x)
#         len_bin = len(encoded_text)
#         if len_bin < bits_for_element: #aggiungo gli 0 iniziali che perdo trasformando in int
#             encoded_text = "0"*(bits_for_element-len_bin)+encoded_text
#         for bit in encoded_text:
#             if expected_elements == 0:
#                 break
#             current_code += bit
#             if(current_code in d_rev):
#                 elem = d_rev[current_code]
#                 current_code = ""
#                 expected_elements -= 1
#                 output = mult_for_row(input_x, output, row, elem, column)
#                 row += 1
#                 if row >= matrix.shape[0]:
#                     column += 1
#                     row = 0
#
#     return output
