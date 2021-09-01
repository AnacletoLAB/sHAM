import sys
import math
import numpy as np
import numba
from numba import njit, jit, prange
import timeit
from compressionNN import huffman
from compressionNN import sparse_huffman
from compressionNN import sparse_huffman_only_data
from simplified_func import *
from libmegaDot import dotp_cpp, dotp_cpp_opt, dotp_cpp_sparse, dotp_cpp_sparse_new
from pympler import asizeof
from functools import reduce
from glob import glob
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

#@njit(["float32[:,:](float32[:,:], float32[:,:], int32, float32, int32)","int64[:,:](int64[:,:], int64[:,:], int32, int64, int32)"], nogil=True, fastmath=True, cache=True)
def mult_for_row(input_x, output, current_row, current_d, column):
    for i in range(input_x.shape[0]):
        output[i][column] += input_x[i][current_row]*current_d
    return output

def dotp(inp, tc, d_rev):
    output = np.zeros((inp.shape[0], len(tc)), dtype='float32', order='F')
    for col, t in enumerate(tc):
        row = 0
        curr = ""
        for b in t:
            curr += b
            if curr in d_rev:
                current_d = d_rev[curr]
                if current_d != 0:
                    output = mult_for_row(inp, output, row, current_d, col)
                curr = ""
                row += 1
    return output

def dot_sparse(list_rows, list_data_reduced, d_rev_data, inp):
    output = np.zeros((inp.shape[0], len(list_rows)), order="F", dtype='float32')
    column = -1
    curr = ""
    for c in range(len(list_rows)):
        r = list_rows[c]
        d = list_data_reduced[c]
        column += 1
        row_counter = 0
        for b in d:
            curr += b
            if curr in d_rev_data:
                data = d_rev_data[curr]
                curr = ""
                output = mult_for_row(inp, output, r[row_counter], data, column)
                row_counter += 1
    return output

def dictspace(dict_):
    space_byte = 0
    for key in dict_:
        space_byte += sys.getsizeof(key)-sys.getsizeof("") + 1 # byte fine stringa
        space_byte += 4 # byte per float32    #dict[key]
        space_byte += 8 # byte per struttura dict
    return space_byte

def space_for_row_cum(matr, list_):
    len_ = matr.shape[0]
    if len_ < 2**8:
        return 1 * len(list_)
    elif len_ < 2**16:
        return 2 * len(list_)
    elif len_ < 2**32:
        return 4 * len(list_)
    return 8 * len(list_)

maindir = 'csv_data/'
filelist = sorted(glob(maindir + '*.csv'))

outf = open("./results.txt", "a")
outf.write("filename t_numpy s_numpy t_ham t_ham_cpp_1c t_ham_cpp_opt t_ham_cpp_p s_ham t_gios s_gios t_sham t_sham_p t_sham_new_p s_sham t_csc s_csc t_csr s_csr t_coo s_coo\n")
outf.close()

for fff in filelist:

    ##### Leggo la matrice da file e creo vettore
    filename = fff.strip().split(sep='/')[-1]
    inmat = np.genfromtxt(fff, delimiter=',')
    vvv2  = np.random.rand(8, inmat.shape[0])
    print(filename, end=' ')


    ##### Dot numpy
    t_np = timeit.timeit("np.dot(vvv2, inmat)", number=10, globals=globals()) / 10
    s_np = asizeof.asizeof(inmat)
    print("np ", end='', flush=True)


    ##### Huffmann
    symb2freq = huffman.dict_elem_freq(inmat)
    e = huffman.encode(symb2freq)
    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)
    encoded = huffman.matrix_with_code(inmat, d, d_rev)
    bit_words_machine = 64
    list_string_col, info = make_word_with_info(inmat, encoded, bit_words_machine)
    min_length_encoded = huffman.min_len_string_encoded(d_rev)
    tc = to_convert(info, list_string_col)
    print("ham_enc ", end='', flush=True)

    t_ham = timeit.timeit("dotp(vvv2,tc,d_rev)", number=1, globals=globals())
    print("ham ", end='', flush=True)
    t_ham_cpp_nonPar = timeit.timeit("dotp_cpp(vvv2,tc,d_rev,1)", number=3, globals=globals()) / 3
    print("ham_cpp_1c ", end='', flush=True)
    t_ham_opt = timeit.timeit("dotp_cpp_opt(vvv2,tc,d_rev,1)", number=3, globals=globals()) / 3
    print("ham_opt ", end='', flush=True)
    t_ham_p = timeit.timeit("dotp_cpp(vvv2,tc,d_rev,12)", number=10, globals=globals()) / 10
    print("ham_p ", end='', flush=True)

    list_bin = huffman.make_words_list_to_int(encoded, bit_words_machine)
    min_length_encoded = huffman.min_len_string_encoded(d_rev)
    int_from_strings = huffman.convert_bin_to_int(list_bin)
    s_ham = dictspace(d_rev)
    s_ham += bit_words_machine / 8 * len(int_from_strings)


    ##### Metodo Indici
    vect_weights = np.hstack(inmat).reshape(-1,1)
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
    uniques = np.unique(all_vect_weights)
    dict_ass = {x:i for i,x in enumerate(uniques)}
    num_values = len(dict_ass)
    dense_weights = inmat
    indexes_weights = np.vectorize(lambda x: dict_ass[x])(dense_weights)
    dict_index_centers = {v:k for k,v in dict_ass.items()}
    vect_centers = np.array([dict_index_centers[k] for k in dict_index_centers.keys()]).reshape(-1, 1)
    t_gios = timeit.timeit(lambda:vvv2.dot(vect_centers[indexes_weights].reshape(indexes_weights.shape[0], indexes_weights.shape[1])), number=10, globals=globals()) / 10

    k = 32 # N simboli
    s_gios = (k*32 + math.ceil(np.log2(k)) * inmat.size) / 8
    print("gios ", end='', flush=True)


    ##### Sparse Huffmann
    symb2freq = huffman.dict_elem_freq(inmat[inmat != 0])
    e = huffman.encode(symb2freq)
    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)
    data, row_index, cum = sparse_huffman.convert_dense_to_csc(inmat)
    d_data, d_rev_data  = sparse_huffman_only_data.huffman_sparse_encoded_dict(data)
    data_encoded = sparse_huffman_only_data.encoded_matrix(data, d_data, d_rev_data)

    core = 1
    list_rows = []
    list_data_encoded = []
    last_c = 0
    cumul_c = []
    cumul_c.append(0)
    for c in cum:
        list_rows.append(row_index[last_c:last_c+c])
        list_data_encoded.append(data_encoded[last_c:last_c+c])
        last_c += c
        cumul_c.append(c + cumul_c[-1])
    list_data_reduced = [reduce(lambda x, y: x+y, l, "") for l in list_data_encoded]
    new_list_rows = []
    for k in list_rows:
        new_list_rows.extend(k)
    print("sham_enc ", end='', flush=True)

    t_sham = timeit.timeit("dot_sparse(list_rows, list_data_reduced, d_rev_data, vvv2)", number=1, globals=globals())
    print("sham ", end='', flush=True)
    t_sham_p = timeit.timeit("dotp_cpp_sparse(list_rows, list_data_reduced, d_rev_data, vvv2, 12)", number=10, globals=globals()) / 10
    print("sham_p ", end='', flush=True)
    t_sham_p_new = timeit.timeit("dotp_cpp_sparse_new(new_list_rows, cumul_c, list_data_reduced, d_rev_data, vvv2, 12)", number=10, globals=globals()) / 10
    print("sham_p_new ", end='', flush=True)
    #s_sham = asizeof.asizeof(list_data_reduced) + asizeof.asizeof(d_rev_data)
    list_bin = huffman.make_words_list_to_int(data_encoded, bit_words_machine)
    int_from_strings = huffman.convert_bin_to_int(list_bin)
    s_sham = dictspace(d_rev)
    s_sham += bit_words_machine / 8 * len(int_from_strings)
    s_sham += space_for_row_cum(inmat, cum) + space_for_row_cum(inmat, row_index)


    ##### CSC, CSR, COO
    inmat_csc = csc_matrix(inmat)
    inmat_csr = csr_matrix(inmat)
    inmat_coo = coo_matrix(inmat)

    t_csc = timeit.timeit("inmat_csc.T.dot(vvv2.T)", number=10, globals=globals()) / 10
    t_csr = timeit.timeit("inmat_csr.T.dot(vvv2.T)", number=10, globals=globals()) / 10
    t_coo = timeit.timeit("inmat_coo.T.dot(vvv2.T)", number=10, globals=globals()) / 10
    print("c** ", end='', flush=True)
    s_csc = asizeof.asizeof(inmat_csc)
    s_csr = asizeof.asizeof(inmat_csr)
    s_coo = asizeof.asizeof(inmat_coo)

    
    ##### Stampa finale
    outstring = "{} {:.6f} {:.0f}KB {:.6f} {:.6f} {:.6f} {:.6f} {:.0f}KB {:.6f} {:.0f}KB {:.6f} {:.6f} {:.6f} {:.0f}KB {:.6f} {:.0f}KB {:.6f} {:.0f}KB {:.6f} {:.0f}KB\n".format(
        filename,
        t_np, s_np / 1024,
        t_ham, t_ham_cpp_nonPar, t_ham_opt, t_ham_p, s_ham / 1024,
        t_gios, s_gios / 1024,
        t_sham, t_sham_p, t_sham_p_new, s_sham / 1024, 
        t_csc, s_csc / 1024, t_csr, s_csr / 1024, t_coo, s_coo / 1024,
    )

    outf = open("./results.txt", "a")
    outf.write(outstring)
    outf.close()
    print("saved")

