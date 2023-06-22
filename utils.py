import numpy as np
import timeit
import copy
import os.path

from scipy.sparse import csc_matrix
from pickle import dump, load
from math import log2, ceil
from functools import reduce

import ctypes
from numpy.ctypeslib import ndpointer

######## CANONICAL HUFFMAN CODE ########

def get_codeword2freq(symb2freq, code, sham=False):
  if sham and 0 in symb2freq:
    symb2freq.pop(0)
  return {code[k]:v for k,v in symb2freq.items()}

def get_symb2freq(matr, sham=False):
  elem, freq = np.unique(matr, return_counts=True)
  symb2freq = dict(zip(elem,freq))
  if sham and 0 in symb2freq:
    symb2freq.pop(0)
  return symb2freq

def get_lengths(freqs, increasing=True):
  n = len(freqs)
  if n == 0:
    return []
  if n == 1:
    return [0]
  if increasing:
    return increasing_lengths(freqs,n)
  else:
    return decreasing_lengths(freqs,n)

def increasing_lengths(decr_freqs, n):
  incr_lens = copy.deepcopy(decr_freqs)
  leaf, root = n-1, n-1
  for next_value in range(n-1, 0, -1):
    if leaf < 0 or (root > next_value and incr_lens[root] < incr_lens[leaf]):
      incr_lens[next_value] = incr_lens[root]
      incr_lens[root] = next_value
      root -= 1
    else:
      incr_lens[next_value] = incr_lens[leaf]
      leaf -= 1
    if leaf < 0 or (root > next_value and incr_lens[root] < incr_lens[leaf]):
      incr_lens[next_value] += incr_lens[root]
      incr_lens[root] = next_value
      root -= 1
    else:
      incr_lens[next_value] += incr_lens[leaf]
      leaf -= 1
  incr_lens[1] = 0
  for next_value in range(2, n):
    incr_lens[next_value] = incr_lens[incr_lens[next_value]] + 1
  avail, used, depth = 1, 0, 0
  root, next_value = 1, 0
  while avail > 0:
    while root < n and incr_lens[root] == depth:
      used += 1
      root += 1
    while avail > used:
      incr_lens[next_value] = depth
      next_value += 1
      avail -= 1
    avail = 2 * used
    depth += 1
    used = 0
  return incr_lens

def decreasing_lengths(incr_freqs, n):
  decr_lens = copy.deepcopy(incr_freqs)
  decr_lens[0] += decr_lens[1]
  root, leaf = 0, 2
  for next_val in range(1, n-1):
    if leaf >= n or decr_lens[root] < decr_lens[leaf]:
      decr_lens[next_val] = decr_lens[root]
      decr_lens[root] = next_val
      root += 1
    else:
      decr_lens[next_val] = decr_lens[leaf]
      leaf += 1
    if leaf >= n or (root < next_val and decr_lens[root] < decr_lens[leaf]):
      decr_lens[next_val] += decr_lens[root]
      decr_lens[root] = next_val
      root += 1
    else:
      decr_lens[next_val] += decr_lens[leaf]
      leaf += 1
  decr_lens[n-2] = 0
  for next_val in range(n-3, -1, -1):
    decr_lens[next_val] = decr_lens[decr_lens[next_val]]+1
  avail, used, depth, root, next_val = 1, 0, 0, n-2, n-1
  while avail > 0:
    while root >= 0 and decr_lens[root] == depth:
      used += 1
      root -= 1
    while avail > used:
      decr_lens[next_val] = depth
      next_val -= 1
      avail -= 1
    avail = 2*used
    depth += 1
    used = 0
  return decr_lens

def get_max_codeword_length(symb2freq, increasing=True, sham=False):
  if sham and 0 in symb2freq.keys():
    symb2freq.pop(0)
  sorted_freqs = list(sorted(symb2freq.values(), reverse=increasing))
  lens = get_lengths(sorted_freqs, increasing=increasing)
  return lens[-1] if increasing else lens[0]

def sum_one(a):
  if not int(a[-1]):
    return a[:-1] + "1"
  if a == "".ljust(len(a), "1"):
    return "1".ljust(len(a)+1, "0")
  j = 2
  while j <= len(a) and int(a[-j]): 
    j += 1
  return (a[:len(a)-j] + "1").ljust(len(a), "0")

def bin_search(a, x):
    for i in range(len(a)-1):
      if a[i] <= x and a[i+1] > x:
        return i, a[i]
    if a[-1] < x:
      raise Exception("Not found.")

def canonical_code(symb2freq, t, incr_lens=True):
  if symb2freq == {}:
    raise Exception("Empty symbols/frequencies dictionary of symbols.")
  sorted_symb2freq = dict(sorted(symb2freq.items(), key=lambda item:item[1], reverse=incr_lens))
  lengths = get_lengths(list(sorted_symb2freq.values()), increasing=incr_lens)
  max_length = lengths[-1] if incr_lens else lengths[0]
  if t > max_length:
    raise ValueError("The size of the partial table cannot exceed the maximum codeword length.")
  symbs_temp = list(sorted_symb2freq.keys())
  if not incr_lens:
    symbs_temp = symbs_temp[::-1]
    lengths = lengths[::-1]
  lengths = dict(zip(symbs_temp, lengths))
  symbols = list(lengths.keys())
  codeword_lengths = {symbols.index(k):v for k,v in lengths.items()}
  code = {}
  for i,(k,v) in enumerate(codeword_lengths.items()):
    if i == 0:
      seq = "".ljust(v, "0")
      code[k] = (v, seq, 0, 0)
    else:
      seq = sum_one(seq)
      if v > old_len:
        seq = seq.ljust(v,"0")
      big_val = int(seq.ljust(max_length,"0"), 2) 
      code[k] = (v, seq, int(seq, 2), big_val)
    old_len = v
  first_symbol = [[] for i in range(max_length+1)]
  for (k,v) in code.items():
    first_symbol[v[0]].append(k)
  for i in range(len(first_symbol)-1, -1, -1):
    if i == 0:
      first_symbol[i] = 0
    elif len(first_symbol[i]) > 0:
      first_symbol[i] = min(first_symbol[i])
    else:
      first_symbol[i] = first_symbol[i+1]
  first_code_r = [code[k][2] for k in first_symbol]
  first_code_r.append(2**max_length)
  first_code_l = [code[k][3] for k in first_symbol]
  first_code_l.append(2**max_length)
  first_symbol.append(0)
  table = []
  for i in range(2**max_length):
    table.append(bin_search(first_code_l, i)[0])
  partial_table = []
  if t != 0:
    for i in range(2**t):
      partial_table.append(bin_search(first_code_l, i)[0])
  explicit_code = {symbols[k]:code[k][1] for k in code.keys()}
  return explicit_code, symbols, first_symbol, first_code_r, first_code_l, table, partial_table

def is_prefix_free(code):
  for _,c1,_,_ in code.values():
    for _,c2,_,_ in code.values():
      if c1 != c2 and c1.startswith(c2):
        return False
  return True

def encode(to_encode, code):
  return ''.join([code[symb] for symb in to_encode])

def decode(enc, fs, symbs, fcl, max_len):
  tot_len = len(enc)
  dec = []
  i, buff, l, used = 0, 0, max_len, 0
  while True:
    if i+l <= tot_len:
      chunk = enc[i:i+l]
    else:
      chunk = enc[i:].ljust(l,'0')
      rem = (i+l)-len(enc[i:])
      if used >= tot_len:
        break
    buff = ((buff << l) & ((1 << max_len) - 1)) + int(chunk, 2)
    i += l
    l, fc = bin_search(fcl, buff)
    used += l
    s = fs[l] + ((buff - fc) >> (max_len - l))
    dec.append(symbs[s])
  return dec

######## MATRICES INFO ########

def avg_k_per_row(matr):
  return np.sum([len(np.unique(row))-1 for row in matr]) / matr.shape[0]

def get_matrix_info(matr):
  n, m, s =  matr.shape[0], matr.shape[1], np.count_nonzero(matr)/matr.size
  info = {}
  info["n"] = n
  info["m"] = m
  info["s"] = s
  k = len(np.unique(matr))
  info["k"] = k
  info["avg_k"] = avg_k_per_row(matr.T)
  return info

######## HAM ########

def get_ham_encoded_matrix(matr, code):
  encode_func = np.vectorize(lambda x: code[x])
  return encode_func(matr)

def get_ham_bitstream(matr, code=None, encoded_matrix=None):
  if encoded_matrix is None:
    encoded_matrix = get_ham_encoded_matrix(matr, code)
  bitstream = []
  for col in encoded_matrix.T:
    bitstream.append(reduce(lambda a,b: a+b, col.tolist()))
  return bitstream

def get_ham_N_D(matrix=None, symb2freq=None, code2freq=None, code=None, t=None, b=32):
  total_len, D = 0, 0
  if code2freq is None:
    if symb2freq is None:
      symb2freq = get_symb2freq(matrix)
    code2freq = get_codeword2freq(symb2freq=symb2freq, code=code)
  total_len, D = 0, 0
  for codeword,freq in code2freq.items():
    length = len(codeword)
    total_len += length * freq
    if length > t:
      D += freq
  N = ceil(total_len / b)
  return N, D

def get_ham_cdot_structures(bitstream, b=32):
  full_bitstream = reduce(lambda a,b : a + b, bitstream)
  bitstream_list = [full_bitstream[i:i+b] for i in range(0, len(full_bitstream), b)]
  int_list = list(map(lambda x:int(x, 2), bitstream_list[:-1]))
  int_list.append(int(bitstream_list[-1].ljust(b, "0"), 2))
  bitstream_len, col_end = 0, []
  for col in bitstream:
    bitstream_len += len(col)
    col_end.append(bitstream_len)
  #col_end = [c-1 for c in col_end]
  return int_list, col_end

######## sHAM #######

def get_sham_encoded_nz(nz, code):
  encode_func = np.vectorize(lambda x: code[x])
  return encode_func(nz)

def get_sham_splitted_nz_ri(encoded_nz=None, nz=None, ri=None, cb=None, code=None):
  if encoded_nz is None:
    encoded_nz = get_sham_encoded_nz(nz, code)
  nz_list, ri_list = [], []
  for i in range(1, len(cb)):
    ri_list.append(ri[cb[i-1]:cb[i]])
    to_append = encoded_nz[cb[i-1]:cb[i]]
    if len(to_append) == 0:     
      to_append = np.array([''])
    nz_list.append(reduce(lambda a,b: a+b, to_append.tolist()))
  return nz_list, ri_list

def get_sham_N_D(t, nz=None, symb2freq=None, code2freq=None, code=None, b=32):
  if symb2freq is None:
    symb2freq = get_symb2freq(nz, True)
  if code2freq is None:
    code2freq = get_codeword2freq(symb2freq=symb2freq, code=code)
  total_len, D = 0, 0
  for k,v in code2freq.items():
    if len(k) > t:
      D += v
    total_len += len(k) * v
  N = ceil(total_len / b)
  return N, D

def get_sham_cdot_structures(nz_list, cb, b=32):
  bitstream = reduce(lambda a,b : a + b, nz_list)
  bitstream_list = [bitstream[i:i+b] for i in range(0, len(bitstream), b)]
  int_list = list(map(lambda x:int(x, 2), bitstream_list[:-1]))
  int_list.append(int(bitstream_list[-1].ljust(b, "0"), 2))
  bitstream_len, col_end = 0, []
  for col in nz_list:
    bitstream_len += len(col)
    col_end.append(bitstream_len)
  #col_end = [c-1 for c in col_end]
  cb_num = [cb[i]-cb[i-1] for i in range(1, len(cb))]
  return int_list, col_end, cb_num

######## CSER ########

def matrix_to_cser(matrix, symb2freq=None):
  matrix = matrix.T
  if symb2freq is None:
    symb2freq = get_symb2freq(matrix)
  O = [k[0] for k in sorted(symb2freq.items(), key=lambda x: x[1], reverse=True)]
  colI = np.array([]).astype("int")
  OI, OPtr, rowPtr = [], [], [0]
  per_row_indices = [[np.argwhere(row==o).flatten() for o in O[1:]] for row in matrix]
  for row in per_row_indices:
    for o_index, o_indices in enumerate(row):
      if o_indices.size:
        OPtr.append(len(colI))
        colI = np.append(colI, o_indices)
        OI.append(o_index + 1)
    rowPtr.append(len(OPtr))
  OPtr.append(len(colI))
  colI = [int(x) for x in list(colI)]
  return {'O':O, 'colI':colI, 'OI':OI, 'OPtr':OPtr, 'rowPtr':rowPtr}

def cser_to_matrix(cser_dict, matr_shape, matr_type="float32"):
  O, colI, OI = cser_dict["O"], cser_dict["colI"], cser_dict["OI"]
  OPtr, rowPtr = cser_dict["OPtr"], cser_dict["rowPtr"]
  matr = np.empty(matr_shape).astype(matr_type)
  matr.fill(O[0])
  O_counter = 0
  for i in range(len(rowPtr)-1):
    start, end = rowPtr[i],rowPtr[i+1]
    ranges = OPtr[start:end+1]
    for j in range(len(ranges)-1):
      start, end = ranges[j], ranges[j+1]
      value = O[OI[O_counter]]
      for col in colI[start:end]:
        matr[i, col] = value
      O_counter += 1
  return matr

######## CSC ########

def get_csc_structure(matrix):
  csc = csc_matrix(matrix)
  nz, ri, cb = csc.data, csc.indices, csc.indptr
  return nz, ri, cb

######## IM ########

def get_im_structure(matr):
  vect_weights = np.hstack(matr).reshape(-1,1)
  all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
  dict_ass = {x:i for i,x in enumerate(np.unique(all_vect_weights))}
  indexes_weights = np.vectorize(lambda x: dict_ass[x])(matr)
  dict_index_centers = {v:k for k,v in dict_ass.items()}
  vect_centers = np.array([dict_index_centers[k] for k in dict_index_centers.keys()]).reshape(-1, 1)
  return indexes_weights, vect_centers

######## SPACE ########

def bit_req(x):
  bit = 8
  while x >= 2**bit:
    bit *= 2
  return bit

def code_space(symbs, fs, fcl, tab, b=32):
  max_length = len(fcl) - 2
  symbs_space = len(symbs) * b
  fs_space = (len(fs)-1) * bit_req(len(symbs))
  fcl_space = len(fcl) * bit_req(2**max_length)
  tab_space = len(tab) * bit_req(max_length)
  return symbs_space, fs_space, fcl_space, tab_space

def code_ub_space(k, b=32):
  fs = (k-1) * bit_req(k)
  fcl = k*bit_req(2**(k-1))
  t = (k-1)*bit_req(k)
  symbs = b * k
  return fs+fcl+t+symbs

def ham_base_space(code2freq, b=32):
  total_len = 0
  for cw, freq in code2freq.items():
    total_len += len(cw) * freq
  return ceil(total_len / b) * b

def ham_psi(n, m, code2freq, symbs, fs, fcl, tab, b=32):
  base = ham_base_space(code2freq=code2freq, b=b)
  symbs_s, fs_s, fcl_s, tab_s = code_space(symbs, fs, fcl, tab, b)
  return (base + symbs_s + fs_s + fcl_s + tab_s) / (n*m)

def ham_ub_psi(n, m, k, b=32, no_code=False):
  B_k = code_ub_space(k, b)
  if no_code:
    return 1 + log2(k)
  return 1 + log2(k) + B_k / (n*m)

def sham_base_space(n, m, code2freq, ri, b=32):
  nz_space = 0
  for cw,freq in code2freq.items():
    nz_space += len(cw) * freq
  nz_space = ceil(nz_space / b) * b
  ri_space = len(ri) * bit_req(n)
  cb_space = m * bit_req(n)
  return nz_space + ri_space + cb_space
  
def sham_psi(n, m, code2freq, ri, symbs, fs, fcl, tab, b=32):
  base = sham_base_space(n=n, m=m, code2freq=code2freq, ri=ri, b=b)
  symbs_s, fs_s, fcl_s, tab_s= code_space(symbs, fs, fcl, tab, b)
  return (base + symbs_s + fs_s + fcl_s + tab_s) / (n*m)

def sham_ub_psi(n, m, s, k, b=32, no_code=False):
  B_k = code_ub_space(k, b)
  if no_code:
    b_I = bit_req(m)
    return s * (1 + log2(k) + b_I) + (b_I * (m+1)) / (n*m)
  return s * (1 + bit_req(k) + bit_req(n)) + (bit_req(n)/n) + (B_k/(n*m))

def cser_psi(n, m, cser_dict, b=32):
  O, colI, OI = cser_dict["O"], cser_dict["colI"], cser_dict["OI"]
  OPtr, rowPtr = cser_dict["OPtr"], cser_dict["rowPtr"]
  space = len(O) * b
  space += len(colI) * bit_req(n) # rows since we transpose the matrix
  space += len(OI) * bit_req(len(O))
  space += len(OPtr) * bit_req(len(colI))
  space += len(rowPtr) * bit_req(len(OPtr))
  return space / (n*m)

def csc_psi(n, m, s, b=32):
  return s * (b + bit_req(n)) + bit_req(n) / n

def im_psi(n, m, k, b=32):
  return bit_req(k) + (k * b) / (n*m)

######## ENERGY ########

energy_costs = {}
energy_costs['s'] = {8:0.2, 16:0.4, 32:0.9}
energy_costs['m'] = {8:0.6, 16:1.1, 32:3.7}
energy_costs['rw'] = {}
energy_costs['rw'][8] = {8:1.25, 16:2.5, 32:5.0}
energy_costs['rw'][32] = {8:2.5, 16:5.0, 32:10.0}
energy_costs['rw'][1024] = {8:12.5, 16:25.0, 32:50.0}
energy_costs['rw'][1025] = {8:250.0, 16:500.0, 32:1000.0}

def get_rw_entry(length, bit_requir):
  temp = (length * bit_requir) / (8 * 1024)
  for key in energy_costs['rw'].keys():
    if temp <= key:
      return energy_costs['rw'][key][bit_requir]
  return energy_costs['rw'][1025][bit_requir]  

def ham_per_elem_energy(n, m, s, k, N, D, t, max_length, b=32):
  b_k, b_x, b_l = bit_req(k), b, bit_req(max_length)
  nm = n*m
  res = get_rw_entry(k, b) + nm + get_rw_entry(N, b) * N
  if t > 0:
    res += get_rw_entry(2**t, bit_req(max_length)) * nm
  res += get_rw_entry(max_length+2, b_l) * (nm + D*log2(max_length))
  res += (energy_costs["s"][b] + energy_costs["m"][b] + get_rw_entry(n, b_x)) * s*nm
  res += get_rw_entry(m, b) * m
  return res / (nm)

def sham_per_elem_energy(n, m, s, k, N1, D1, t, max_length, b=32):
  b_x, b_k, b_I, b_l = b, bit_req(k), bit_req(m), bit_req(max_length)
  snm = s*n*m
  res = get_rw_entry(k, b) * snm + get_rw_entry(N1, b) * N1
  res += get_rw_entry(max_length+2, b_l) * (snm + D1*log2(max_length))
  if t > 0:
    res += get_rw_entry(2**t, bit_req(max_length)) * snm
  res += (energy_costs["s"][b] + energy_costs["m"][b] + get_rw_entry(n, b_x)) * snm
  res += get_rw_entry(snm, b_I) * snm
  res += get_rw_entry(m, b) * m
  return res / (n*m)

def cser_per_elem_energy(n, m, s, k, avg_k, cser_dict, b=32): # call this function passing n and then m
  O, colI, OI = cser_dict["O"], cser_dict["colI"], cser_dict["OI"]
  OPtr, rowPtr = cser_dict["OPtr"], cser_dict["rowPtr"]
  k = len(O)
  b_x, b_I, b_k = b, bit_req(n), bit_req(k)
  snm = s*n*m
  res = (energy_costs["s"][b_x] + get_rw_entry(n, b_x) + get_rw_entry(snm, b_I)) * snm
  res += (energy_costs["s"][b] + energy_costs["m"][b] + get_rw_entry(len(OPtr), bit_req(snm))) * m*avg_k
  res += get_rw_entry(k, b_k) * m*avg_k
  res += (get_rw_entry(m, b) + get_rw_entry(len(rowPtr), bit_req(len(OPtr)))) * m
  return res / (n*m)

def csc_per_elem_energy(n, m, s, b=32):
  b_x, b_I = b, bit_req(n)
  snm = s*n*m
  res = (energy_costs["s"][b] + energy_costs["m"][b]) * snm
  res += (get_rw_entry(snm, b_I) + get_rw_entry(snm, b) + get_rw_entry(n, b_x)) * snm
  res += get_rw_entry(snm, b_I) * n*m
  res += get_rw_entry(m, b) * n*m
  return res / (n*m)

def im_per_elem_energy(n, m, s, k, b=32):
  b_k, b_x = bit_req(k), b
  nm = n*m
  res = (get_rw_entry(k, b) + get_rw_entry(nm, b_k)) * nm
  res += (energy_costs["s"][b] + energy_costs["m"][b] + get_rw_entry(n, b_x)) * s*nm
  res += get_rw_entry(m, b) * m
  return res / (nm)

######## LOAD STUFF ########

def load_if_exists(filename):
  if not os.path.exists(filename):
    raise FileNotFoundError("The file does not exist.")
  with open(filename, "rb") as file:
    return load(file)

######## CDOT ########

def compute_cdot(n, m, input_x, ham_or_sham, code, cdot_structure, input_row_num=8, variant="partial", num=1, nthread=8, cdot_path="../c_dot/", seed=0, b=32, matrix_type="float32"):
  if ham_or_sham not in ["ham", "sham"]:
    raise ValueError("Only ham and sham formats are admitted.")
  if variant not in ["no-table", "partial", "full"]:
    raise ValueError("The variants for the canonical Huffman code must be one among 'no-table', 'partial' or 'full'.")
  MATROW, MATCOL = n, m
  WORDSIZE = np.uint8(b)
  thread_count = np.uint8(nthread)
  clib = ctypes.CDLL(cdot_path + ham_or_sham + "_dot_" + variant + ".so")
  col_end = np.asarray(cdot_structure["col_end"], dtype=np.uint32)
  symbols = np.asarray(code["symbs"], dtype=np.float32)
  ND_POINTER_float32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
  first_code_l = np.asarray(code["fcl"], dtype=np.uint32)
  ND_POINTER_uint = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C")
  first_symbol = np.asarray(code["fs"], dtype=np.uint16)
  ND_POINTER_usint = np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C")
  if variant == "partial":
    partial_table = np.asarray(code["par_tab"], dtype = np.uint32)
    T = np.uint16(code["t"])
  elif variant == "full":
    table = np.asarray(code["tab"], dtype = np.uint32)
  LMAX = np.uint16(code["lmax"])
  CHAM = np.asarray(cdot_structure["int_list"], dtype=np.uint32)
  K = np.uint16(len(symbols)) 
  fcl_length = np.uint16(len(first_code_l))
  CHAM_DIM = np.uint32(len(CHAM))
  if ham_or_sham == "sham":
    ri = np.asarray(cdot_structure["ri"], dtype=np.uint32)
    cb = np.asarray(cdot_structure["cb_num"], dtype=np.uint32)
  input_x_reshaped = input_x.reshape(MATROW,input_row_num).astype(matrix_type)
  left_mat = np.asarray(input_x_reshaped, dtype=np.float32).reshape(MATROW,input_row_num, order="C")
  resmat = np.zeros((MATCOL, input_row_num), dtype=np.float32).reshape((MATCOL, input_row_num), order="C")
  ND_POINTER2_float32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")
  clib.dotMAT.restype = None
  if variant == "no-table":
    if ham_or_sham == "ham": 
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat), number=num, globals=globals()) / num
    elif ham_or_sham == "sham":
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32, ND_POINTER_uint, ND_POINTER_uint]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat, ri, cb), number=num, globals=globals()) / num 
  elif variant == "partial":
    if ham_or_sham == "ham":
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32, ND_POINTER_uint, ctypes.c_ushort]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat, partial_table, T), number=num, globals=globals()) / num
    elif ham_or_sham == "sham":
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_uint, ctypes.c_ushort]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat, ri, cb, partial_table, T), number=num, globals=globals()) / num
  elif variabt == "full":
    if ham_or_sham == "ham":
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32, ND_POINTER_uint]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat, table), number=num, globals=globals()) / num
    if ham_or_sham == "sham":
      clib.dotMAT.argtypes = [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_uint, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_usint, ND_POINTER_uint, ND_POINTER_float32, ND_POINTER2_float32, ND_POINTER2_float32, ND_POINTER_uint, ND_POINTER_uint, ND_POINTER_uint]
      dot_time = timeit.timeit(lambda:clib.dotMAT(K, fcl_length, MATROW, MATCOL, CHAM_DIM, LMAX, thread_count, WORDSIZE, input_row_num, first_code_l, CHAM, first_symbol, col_end, symbols, resmat, left_mat, ri, cb, table), number=num, globals=globals()) / num     
  return dot_time, resmat

def compute_dot_times(n, m, ham_code, ham_cdot_structure, sham_code, sham_cdot_structure, variant="partial", sparse_matr=None, indexes_weights=None, vect_centers=None, sham_row_index=None, input_row_num=8, rep=25, num=1, nthread=8, cdot_path="../c_dot/", seed=0, b=32, matrix_type="float32"):
  times = {"HAM":[], "sHAM":[], "CSC":[], "IM":[]}
  np.random.seed(seed)
  for _ in range(rep):
    input_x_values = np.random.rand(n*input_row_num)
    input_x = input_x_values.reshape(input_row_num, n).astype(matrix_type)
    ham_cdot_time, ham_cdot_res = compute_cdot(n=n, m=m, input_x=input_x_values, ham_or_sham="ham", code=ham_code, cdot_structure=ham_cdot_structure, variant=variant, num=num, nthread=nthread, cdot_path=cdot_path, b=b, matrix_type=matrix_type)
    times["HAM"].append(ham_cdot_time)
    sham_cdot_time, sham_cdot_res = compute_cdot(n=n, m=m, input_x=input_x_values, ham_or_sham="sham", code=sham_code, cdot_structure=sham_cdot_structure, input_row_num=input_row_num, variant=variant, num=num, nthread=nthread, cdot_path=cdot_path, b=b, matrix_type=matrix_type)
    times["sHAM"].append(sham_cdot_time)
    times["IM"].append(timeit.timeit(lambda:input_x.dot(vect_centers[indexes_weights].reshape(indexes_weights.shape[0], indexes_weights.shape[1])), number=num, globals=globals()) / num)
    times["CSC"].append(timeit.timeit(lambda:sparse_matr.dot(input_x.T), number=num, globals=globals()) / num)
  return {key:np.mean([c/input_row_num for c in value]) for key,value in times.items()}