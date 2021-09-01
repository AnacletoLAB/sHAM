import os
from os import listdir
from os.path import isfile, join
import gc
import sys
from sys import getsizeof
from math import floor
import pickle
import timeit
import getopt
from functools import reduce
from itertools import repeat
from multiprocessing import Pool

import keras
import numba
import numpy as np
import tensorflow as tf
from datahelper_noflag import *
from keras.datasets import mnist, cifar10
from keras.utils import np_utils

from compressionNN import huffman
from compressionNN import sparse_huffman
from compressionNN import sparse_huffman_only_data
from libmegaDot import dotp_cpp, dotp_cpp_sparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

exec(open("../nets/GPU.py").read())

@njit(["float32[:,:](float32[:,:], float32[:,:], int32, float32, int32)","int64[:,:](int64[:,:], int64[:,:], int32, int64, int32)"], nogil=True, fastmath=True, cache=True)
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

def to_convert(info, list_string_col):
    to_conv = []
    for i in range(len(info)-1):
        if info[i][0] == info[i+1][0]:
            to_conv += [list_string_col[info[i][0]][info[i][1]:info[i+1][1]]]
        else:
            if info[i][0] + 1 == info[i+1]:
                to_conv += [list_string_col[info[i][0]][info[i][1]:]+list_string_col[info[i+1][0]][:info[i+1][1]]]
            else:
                elem = ""
                elem += list_string_col[info[i][0]][info[i][1]:]
                for j in range(info[i][0]+1, info[i+1][0]):
                    elem += list_string_col[j]
                if info[i+1][1] != 0:
                    elem += list_string_col[info[i+1][0]][:info[i+1][1]]
                to_conv += [elem]
    return to_conv

def make_word_with_info(matr, encoded, bit_words_machine=64):
    bit = bit_words_machine
    list_string_col =[]
    string = ''
    row = 0
    info = []

    for x in np.nditer(encoded, order='F'):
        if row == 0:
            info += [(len(list_string_col), len(string))]
        string += (np.array2string(x)[1:-1])
        if len(string) > bit:
            bit_overflow = len(string)%bit
            list_string_col += [string[:-bit_overflow]]
            string = string[-bit_overflow:]
        elif len(string) == bit:
            list_string_col += [string]
            string = ''
        row += 1
        if row >= matr.shape[0]:
            row = 0

    bit_remain = len(string)
    info += [(len(list_string_col), bit_remain)]
    if bit_remain > 0:
        string += "0"*(bit-bit_remain) #padding di 0 per renderla lunga bit
        list_string_col += [string]

    return list_string_col, info

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity

def dict_space(dict_):
    space_byte = 0
    for key in dict_:
        space_byte += getsizeof(key)-getsizeof("") + 1 #byte fine stringa
        space_byte += 4 #byte per float32    #dict_[key]
        space_byte += 8 #byte per struttura dict
    return space_byte

def dense_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    return npmatr2d.shape[0]*npmatr2d.shape[1]*byte

def cnn_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    return npmatr2d.size * byte

def list_of_dense_indexes(model):
    index_denses = []
    i = 2
    for layer in model.layers[2:]:
        if type(layer) is tf.keras.layers.Dense:
            index_denses.append(i)
        i += 1
    return index_denses

def make_model_pre_post_dense(model, index_dense):
    submodel = tf.keras.Model(inputs=model.input,
                            outputs=model.layers[index_dense-1].output)
    return submodel

def list_of_dense_weights_indexes(list_weights):
    indexes_denses_weights = []
    i = 2
    for w in list_weights[2:]:
        if len(w.shape) == 2:
            indexes_denses_weights.append(i)
        i += 1
    return indexes_denses_weights

def list_of_cnn_and_dense_weights_indexes(list_weights):
    # Deep DTA
    if len(list_weights) == 22:
        return [2, 4, 6, 8, 10, 12], [14, 16, 18, 20]
    # VGG19
    elif len(list_weights) == 114:
        return [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90], [96, 102, 108]

def make_huffman(model, lodi, lodwi, lw, also_parallel=False):
    core = 24
    times = 5
    bit_words_machine = 64

    vect_weights = [np.hstack(lw[i]).reshape(-1,1) for i in lodwi]
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)

    symb2freq = huffman.dict_elem_freq(all_vect_weights)
    e = huffman.encode(symb2freq)

    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)
    print("lungh_diz", len(d))

    dense_inputs = []

    for i in range(len(lodi)):
        model_pre_dense = make_model_pre_post_dense(model, lodi[i])
        dense_inputs += [model_pre_dense.predict(x_test)]



    encodeds = [huffman.matrix_with_code(lw[l], d, d_rev) for l in lodwi]
    list_bins = [huffman.make_words_list_to_int(encoded, bit_words_machine) for encoded in encodeds]
    min_length_encoded = huffman.min_len_string_encoded(d_rev)
    int_from_strings = [huffman.convert_bin_to_int(list_bin) for list_bin in list_bins]
    t_huff = [timeit.timeit(lambda:huffman.dot_for_col(dense_inputs[i] , int_from_strings[i], lw[lodwi[i]], d_rev, bit_words_machine,
                            lw[lodwi[i]].dtype, min_length_encoded), number=times, globals=globals()) for i in range(len(lodwi))]

    if also_parallel:
        t_print = []
        t_print_cpp = []
        for j, l in enumerate(lodwi):
            encoded = huffman.matrix_with_code(lw[l], d, d_rev)
            list_string_col, info = make_word_with_info(lw[l], encoded)
            tc = to_convert(info, list_string_col)
            splitted_matr = np.array_split(lw[l], core, axis=1)

            num_col_splitted = [s.shape[1] for s in splitted_matr]
            slice_cols = [0]
            count = 0
            for i in range(len(num_col_splitted)):
                slice_cols.append(num_col_splitted[i]+count)
                count += num_col_splitted[i]

            tc_splitted = [tc[slice_cols[i]:slice_cols[i+1]] for i in range(len(slice_cols)-1)]
            inp_f = np.asfortranarray(dense_inputs[j])
            pool = Pool(core)
            t_p_part = timeit.timeit(lambda:pool.starmap(dotp, zip(repeat(inp_f), tc_splitted, repeat(d_rev))), number=times, globals=globals())
            t_print.append(t_p_part)
            pool.close()
            pool.join()

            inf = np.asfortranarray(dense_inputs[j])
            t_p_part_cpp = timeit.timeit(lambda: dotp_cpp(inf, tc, d_rev, core), globals=globals(), number = times)
            t_print_cpp.append(t_p_part_cpp)

        t_last = timeit.timeit(lambda:huffman.dot_for_col(dense_inputs[-1] , int_from_strings[-1], lw[lodwi[-1]], d_rev, bit_words_machine,
                                lw[lodwi[-1]].dtype, min_length_encoded), number=times, globals=globals())

        if t_last < t_print[-1]:
            t_print[-1] = t_last

        t_p = sum(t_print)
        t_p_cpp = sum(t_print_cpp)

    t_np = [timeit.timeit(lambda:dense_inputs[i].dot(lw[lodwi[i]]), number=times, globals=globals()) for i in range(len(lodwi))]

    space_dense = sum([dense_space(lw[i]) for i in lodwi])
    space_huffman = dict_space(d_rev) 
    space_huffman += sum([bit_words_machine/8 * (len(int_from_string)) for int_from_string in int_from_strings]) 

    return space_dense, space_huffman, sum(t_np), sum(t_huff), t_p if also_parallel else 0, t_p_cpp if also_parallel else 0




def space_for_row_cum(matr, list_):
    len_ = matr.shape[0]
    if len_ < 2**8:
        return 1 * len(list_)
    elif len_ < 2**16:
        return 2 * len(list_)
    elif len_ < 2**32:
        return 4 * len(list_)
    return 8 * len(list_)

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

def make_huffman_sparse_par(model, lodi, lodwi, lw, also_parallel=True):
    core = 24
    times = 5
    bit_words_machine = 64

    vect_weights = [np.hstack(lw[i]).reshape(-1,1) for i in lodwi]
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
        
    symb2freq = huffman.dict_elem_freq(all_vect_weights[all_vect_weights != 0])
    e = huffman.encode(symb2freq)

    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)
    print("lungh_diz", len(d))

    dense_inputs = []

    for i in range(len(lodi)):
        model_pre_dense = make_model_pre_post_dense(model, lodi[i])
        dense_inputs += [model_pre_dense.predict(x_test)]
        
    space_dense = 0
    space_sparse_huffman = 0
    for l in lodwi:
        data, row_index, cum = sparse_huffman.convert_dense_to_csc(lw[l])
        #encoded = huffman.matrix_with_code(lw[l], d, d_rev)
        ########list_bin = huffman.make_words_list_to_int(encoded, bit_words_machine)
        d_data, d_rev_data  = sparse_huffman_only_data.huffman_sparse_encoded_dict(data)
        data_encoded = sparse_huffman_only_data.encoded_matrix(data, d_data, d_rev_data)
        list_bin = huffman.make_words_list_to_int(data_encoded, bit_words_machine)
        ########
        int_from_strings = huffman.convert_bin_to_int(list_bin)
        
        space_dense += dense_space(lw[l])
        space_sparse_huffman += dict_space(d_rev)  
        space_sparse_huffman += bit_words_machine/8 * len(int_from_strings) 
        space_sparse_huffman += space_for_row_cum(lw[l], cum) + space_for_row_cum(lw[l], row_index)
    
    t_print = []
    t_print_cpp = []
    for j, l in enumerate(lodwi):
        data, row_index, cum = sparse_huffman.convert_dense_to_csc(lw[l])
        d_data, d_rev_data  = sparse_huffman_only_data.huffman_sparse_encoded_dict(data)
        data_encoded = sparse_huffman_only_data.encoded_matrix(data, d_data, d_rev_data)
        
        list_rows = []
        list_data_encoded = []
        last_c = 0
        for c in cum:
            list_rows.append(row_index[last_c:last_c+c])
            list_data_encoded.append(data_encoded[last_c:last_c+c])
            last_c += c
        
        list_data_reduced = [reduce(lambda x, y: x+y, l, "") for l in list_data_encoded]
        
        inp_f = np.asfortranarray(dense_inputs[j])
        
        splitted_matr = np.array_split(lw[l], core, axis=1)

        num_col_splitted = [s.shape[1] for s in splitted_matr]
        
        slice_cols = [0]
        count = 0
        for i in range(len(num_col_splitted)):
            slice_cols.append(num_col_splitted[i]+count)
            count += num_col_splitted[i]
            
        list_rows_splitted = [list_rows[slice_cols[i]:slice_cols[i+1]] for i in range(len(slice_cols)-1)]
        list_data_splitted = [list_data_reduced[slice_cols[i]:slice_cols[i+1]] for i in range(len(slice_cols)-1)]
            
        pool = Pool(core)
        t_p_part = timeit.timeit(lambda:pool.starmap(dot_sparse, zip(list_rows_splitted, list_data_splitted, repeat(d_rev_data), repeat(inp_f))), number=times, globals=globals())
        t_print.append(t_p_part)
        pool.close()
        pool.join()

        inf = np.asfortranarray(dense_inputs[j])
        t_p_part_cpp = timeit.timeit(lambda: dotp_cpp_sparse(list_rows, list_data_reduced, d_rev_data, inp_f, core), globals=globals(), number = times)
        t_print_cpp.append(t_p_part_cpp)


    matr_shape, int_data, d_rev_data, row_index, cum, expected_c, min_length_encoded = sparse_huffman_only_data.do_all_for_me(lw[lodwi[-1]], bit_words_machine)
    
    
    
    t_last = timeit.timeit(lambda:sparse_huffman_only_data.sparsed_encoded_dot(dense_inputs[-1], matr_shape, int_data, d_rev_data, row_index, cum, bit_words_machine, expected_c, "float32", min_length_encoded), number=times, globals=globals())

    if t_last < t_print[-1]:
        t_print[-1] = t_last

    t_p = sum(t_print)
    t_p_cpp = sum(t_print_cpp)

    t_np = [timeit.timeit(lambda:dense_inputs[i].dot(lw[lodwi[i]]), number=times, globals=globals()) for i in range(len(lodwi))]
        
    return space_dense, space_sparse_huffman, sum(t_np), t_p, t_p_cpp
        



def split_filename(fn):
    c = fn.split(sep="-")
    if len(c) == 2:
        return 0, c[0], c[1]
    elif len(c) == 3:
        return c[0], c[1], c[2]
    elif len(c) == 5:
        return c[0], c[2], c[4]


# Get the arguments from the command-line except the filename
argv = sys.argv[1:]

try:
    string_error = 'usage: testing_time_space.py -t <type of compression> -d <directory of compressed weights> -m <file original keras model>'
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:d:m:s:q:', ['type', 'directory', 'model', 'dataset', 'quant'])
    if len(opts) != 5:
      print (string_error)
      # Iterate the options and get the corresponding values
    else:

        for opt, arg in opts:
            if opt == "-t":
                print("tipo: ", arg)
                type_compr = arg
            elif opt == "-d":
                print("directory: ", arg)
                directory = arg
            elif opt == "-m":
                print("model_file: ", arg)
                model_file=arg
            elif opt == "-q":
                pq = True if arg==1 else False
            elif opt == "-s":
                if arg == "kiba":
                    print(arg)
                    # data loading
                    dataset_path = '../nets/DeepDTA/data_utils/kiba/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1000,
                                          smilen = 100,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 0)

                    XD = np.asarray(XD)
                    XT = np.asarray(XT)
                    Y = np.asarray(Y)

                    test_set, outer_train_sets = dataset.read_sets(dataset_path, 1)

                    flat_list = [item for sublist in outer_train_sets for item in sublist]

                    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
                    trrows = label_row_inds[flat_list]
                    trcol = label_col_inds[flat_list]
                    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
                    trrows = label_row_inds[test_set]
                    trcol = label_col_inds[test_set]
                    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

                    x_train=[np.array(drug), np.array(targ)]
                    y_train=np.array(aff)
                    x_test=[np.array(drug_test), np.array(targ_test)]
                    y_test=np.array(aff_test)

                elif arg == "davis":
                    # data loading
                    dataset_path = '../nets/DeepDTA/data_utils/davis/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1200,
                                          smilen = 85,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 1)

                    XD = np.asarray(XD)
                    XT = np.asarray(XT)
                    Y = np.asarray(Y)

                    test_set, outer_train_sets = dataset.read_sets(dataset_path, 1)

                    flat_list = [item for sublist in outer_train_sets for item in sublist]

                    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
                    trrows = label_row_inds[flat_list]
                    trcol = label_col_inds[flat_list]
                    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
                    trrows = label_row_inds[test_set]
                    trcol = label_col_inds[test_set]
                    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

                    x_train=[np.array(drug), np.array(targ)]
                    y_train=np.array(aff)
                    x_test=[np.array(drug_test), np.array(targ_test)]
                    y_test=np.array(aff_test)
                elif arg == "mnist":
                    print(arg)
                    # data loading
                    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
                    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
                    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
                    x_train = x_train.astype("float32") / 255.0
                    x_test = x_test.astype("float32") / 255.0
                    y_train = np_utils.to_categorical(y_train, 10)
                    y_test = np_utils.to_categorical(y_test, 10)
                elif arg == "cifar10":
                    # data loading
                    num_classes = 10
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    y_train = keras.utils.to_categorical(y_train, num_classes)
                    y_test = keras.utils.to_categorical(y_test, num_classes)
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    # data preprocessing
                    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
                    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
                    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
                    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
                    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
                    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
except getopt.GetoptError:
    # Print something useful
    print (string_error)
    sys.exit(2)

model = tf.keras.models.load_model(model_file)
original_acc = model.evaluate(x_test, y_test)
if isinstance(original_acc, list):
    original_acc = original_acc[1]

print(original_acc)

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

pruning_l_h = []
ws_l_h = []
diff_acc_h = []
space_h = []
space_sh = []
time_h = []
time_h_p = []
time_h_p_cpp = []
nonzero_h = []
perf = []

pruning_l_sh = []
ws_l_sh = []
diff_acc_sh = []
space_sh = []
time_sh = []
nonzero_sh = []

        
for weights in sorted(onlyfiles):
        gc.collect()
        if weights[-3:] == ".h5":
            lw = pickle.load(open(directory+weights, "rb"))
            model.set_weights(lw)

            cnnIdx, denseIdx = list_of_cnn_and_dense_weights_indexes(lw)
            space_expanded_cnn = sum([cnn_space(lw[i]) for i in cnnIdx])
            
            # Estraggo quanti simboli sono usati effettivamente nei liv conv
            vect_weights =[np.hstack(lw[i]).reshape(-1,1) for i in cnnIdx]
            all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
            uniques = np.unique(all_vect_weights)
            num_values = len(uniques)
            space_compr_cnn = (num_values*32 + math.ceil(np.log2(num_values)) * sum([lw[i].size for i in cnnIdx])) / 8

            pr, ws, acc = split_filename(weights)
            ws_acc = float(acc[:-3])
            print("{}% & {} --> {}".format(pr, ws, ws_acc))
            perf += [ws_acc]

            lodi = list_of_dense_indexes(model)
            lodwi = list_of_dense_weights_indexes(lw)
            assert len(lodi) == len(lodwi)

            non_zero = []

            if type_compr == "ham":
                space_dense, space_huffman, t_np, t_huff, t_p, t_p_cpp = make_huffman(model, lodi, lodwi, lw, also_parallel=True)
                time_h.append(floor(t_huff/t_np))
            elif type_compr == "sham":
                space_dense, space_huffman, t_np, t_p, t_p_cpp = make_huffman_sparse_par(model, lodi, lodwi, lw, also_parallel=True)
            elif type_compr == "all":
                 space_dense, space_huffman, t_np, t_huff, t_p, t_p_cpp = make_huffman(model, lodi, lodwi, lw, also_parallel=True)
                 space_dense, space_shuffman, t_np, t_p, t_p_cpp = make_huffman_sparse_par(model, lodi, lodwi, lw, also_parallel=True)
            
            pruning_l_h.append(pr)
            ws_l_h.append(ws)
            diff_acc_h.append(round(ws_acc-original_acc, 5))
            space_h.append(round((space_compr_cnn + space_huffman)/(space_expanded_cnn + space_dense), 5)) # Tengo conto anche di cnn
            space_sh.append(round((space_compr_cnn + space_shuffman)/(space_expanded_cnn + space_dense), 5))
            nonzero_h.append(non_zero)
            ### Commentato per salvare solo i tempi, non i rapporti
            # time_h_p.append(round(t_p/t_np, 5))
            # time_h_p_cpp.append(round(t_p_cpp/t_np, 5))
            time_h.append(round(t_np, 5))
            time_h_p.append(round(t_p, 5))
            time_h_p_cpp.append(round(t_p_cpp, 5))
            ####
            

            if type_compr == "ham":
                print("{} {} acc1, space {}, time {} time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1], time_h_p[-1], time_h_p_cpp[-1]))
            elif type_compr == "sham":
                ### Commentato per salvare solo i tempi, non i rapporti
                # print("{} {} acc1, space {}, time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h_p[-1], time_h_p_cpp[-1]))
                print("{} {} acc1, space {}, time {} time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h[-1], time_h_p[-1], time_h_p_cpp[-1]))
                ####
            elif type_compr == "all":
                print("{} {} acc1, spaceh {}, spacesh {},time {} time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], space_sh[-1], time_h[-1], time_h_p[-1], time_h_p_cpp[-1]))


if type_compr == "ham":
    str_res = "results/huffman_upq.txt" if pq else "results/huffman_pruws.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspace = {}\ntime = {}\ntime_p = {}\ntime_p_cpp = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h, time_h, time_h_p, time_h_p_cpp))

elif type_compr == "sham":
    str_res = "results/huffman_upq.txt" if pq else "results/huffman_pruws_sparse.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspace = {}\ntime_p = {}\ntime_p_cpp = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h, time_h_p, time_h_p_cpp))

if type_compr == "all":
    str_res = "results/all_upq.txt" if pq else "results/all_pruws.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspaceh = {}\nspacesh = {}\ntime_p = {}\ntime_p_cpp = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h, space_sh, time_h_p, time_h_p_cpp))
