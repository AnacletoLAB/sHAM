import sys
sys.path.append('../')
from utils import *

import warnings
import numpy as np

from glob import glob
from pickle import dump, load

def load_nn_matrix(name, matrix_path="data/matrices/", matrix_type="float32"):
  matrix = np.loadtxt(matrix_path + name + ".csv", delimiter=',').astype(matrix_type)
  if len(matrix.shape) == 1:
    matrix = matrix.reshape(-1,1)
  return matrix

def get_unique_names(nn="vgg", matrices_path="data/matrices/", reverse=False):
  if nn != "vgg" and nn != "dta":
    raise ValueError('Only "vgg" and "dta" are available as neural networks to test.')
  return sorted(list(set([filename.replace(matrices_path, "")[:-8] for filename in sorted(glob(matrices_path + nn + "_*"))])), reverse=reverse)

def get_matrices_names_from_unique_name(nn_name, matrices_path="data/matrices/"):
  return [filename.replace(matrices_path, "").replace(".csv", "") for filename in sorted(glob(matrices_path + nn_name + "_*"))]

def get_nn_symb2freq(nn_name, matrices_path="data/matrices/", sham=False, save=False, save_path="data/symb2freq/"):
  filename = save_path + nn_name + "_symb2freq.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  nn_symb2freq = {}
  for matrix_name in get_matrices_names_from_unique_name(nn_name=nn_name, matrices_path=matrices_path):
    matrix = load_nn_matrix(name=matrix_name, matrix_path="data/matrices/")
    symb2freq = get_symb2freq(matrix, sham=sham)
    for symb,freq in symb2freq.items():
      if symb in nn_symb2freq:
        nn_symb2freq[symb] += freq
      else:
        nn_symb2freq[symb] = freq
  if sham:
    nn_symb2freq.pop(0)
  if save:
    if sham:
      warnings.warn("You are saving a symb2freq dictionary without 0. You will not be able to use it for the HAM format.")
    with open(filename, "wb") as file:
      dump(nn_symb2freq, file)
  return nn_symb2freq

def get_nn_huffman_code(nn_name, t=None, symb2freq=None, symb2freq_path="data/symb2freq/", sham=False, save=False, save_path="data/"):
  ham_sham = "sham" if sham else "ham"
  filename = save_path + ham_sham + "/huffman_code/" + nn_name + "_code.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if symb2freq is None:
    symb2freq = get_nn_symb2freq(nn_name)
  if sham and 0 in symb2freq:
    symb2freq.pop(0)
  max_length = get_max_codeword_length(symb2freq)
  if t is None:
    t = ceil((log2(max_length)))
  code, symbs, fs, fcr, fcl, tab, par_tab = canonical_code(symb2freq, t)
  huffman_code = {"code":code, "symbs":symbs, "fs":fs, "fcr":fcr, "fcl":fcl, "tab":tab, "par_tab":par_tab, "t":t, "lmax":max_length}
  if save:
    with open(filename, "wb") as file:
      dump(huffman_code, file)
  return huffman_code

def get_nn_matrix_info(name, matrix=None, matrix_path="data/matrices/", save=False, save_path="/data/matrices/info/"):
  filename = save_path + "/" + name + "_info.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_nn_matrix(name, matrix_path=matrix_path)
  info = get_matrix_info(matrix)
  if save:
    with open(filename, "wb") as file:
      dump(info, file)            
  return info

def get_nn_cser_structure(name, matrix=None, matrix_symb2freq=None, matrix_path="data/matrices/", save=False, save_path="data/cser/"):
  filename = save_path + name + "_cser.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_nn_matrix(name, matrix_path=matrix_path)
  cser_dict = matrix_to_cser(matrix=matrix, symb2freq=matrix_symb2freq)
  if save:
    with open(filename, "wb") as file:
      dump(cser_dict, file)
  return cser_dict

def get_nn_ham_bitstream(name, matrix=None, ham_code=None, save=False, save_path="data/ham/"):
  filename = save_path + name + "_bitstream.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_nn_matrix(name)
  if ham_code is None:
    ham_code = get_nn_huffman_code(name)
  encoded_matrix = get_ham_encoded_matrix(matrix, ham_code["code"])
  bitstream = get_ham_bitstream(matrix, ham_code["code"], encoded_matrix=encoded_matrix)
  if save:
    with open(filename, "wb") as file:
      dump(bitstream, file)
  return bitstream

def get_nn_ham_N_D(name, matrix=None, matrix_symb2freq=None, ham_code=None, b=32, save=False, save_path="data/ham/"):
  filename = save_path + name + "_N_D.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_nn_matrix(name)
  matrix_symb2freq = get_symb2freq(matrix)
  code2freq_temp = get_codeword2freq(symb2freq=matrix_symb2freq, code=ham_code["code"])
  N, D = get_ham_N_D(code2freq=code2freq_temp, code=ham_code["code"], t=ham_code['t'], b=b)
  N_D = {"N":N, "D":D}
  if save:
    with open(filename, "wb") as file:
      dump(N_D, file)
  return N_D

def get_nn_ham_cdot_structure(name, bitstream=None, matrix=None, ham_code=None, b=32, save=False, save_path="data/ham/cdot_structure/"):
  filename = save_path + name + "_cdot.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if bitstream is None:
    bitstream = get_nn_ham_bitstream(name, matrix=matrix, ham_code=ham_code)
  int_list, col_end = get_ham_cdot_structures(bitstream, b=b)
  cdot_structure = {"int_list":int_list, "col_end":col_end}
  if save:
    with open(filename, "wb") as file:
      dump(cdot_structure, file)
  return cdot_structure

def get_nn_sham_structure(name, sham_code=None, matrix=None, save=False, save_path="data/sham/"):
  filename = save_path + name + "_structure.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_nn_matrix(name)
  nz, ri, cb = get_csc_structure(matrix)
  encoded_nz = get_sham_encoded_nz(nz, sham_code["code"])
  structure = {"enc_nz":encoded_nz, "nz":nz, "ri":ri, "cb":cb}
  if save:
    with open(filename, "wb") as file:
      dump(structure, file)
  return structure

def get_nn_sham_N_D(name, sham_code, nz=None, matrix_symb2freq=None, matrix=None, b=32, save=False, save_path="data/sham/"):
  filename = save_path + name + "_N_D.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if nz is None:
    if matrix is None:
      matrix = load_nn_matrix(name)
    nz = csc_matrix(matrix).data
  matrix_symb2freq = get_symb2freq(nz, sham=True)
  code2freq_temp = get_codeword2freq(symb2freq=matrix_symb2freq, code=sham_code["code"])
  N, D = get_sham_N_D(code2freq=code2freq_temp, code=sham_code['code'], t=sham_code['t'], b=b)
  N_D = {"N":N, "D":D}
  if save:
    with open(filename, "wb") as file:
      dump(N_D, file)
  return N_D

def get_nn_sham_cdot_structure(name, encoded_nz=None, ri=None, cb=None, matrix=None, sham_code=None, b=32, save=False, save_path="data/sham/cdot_structure/"):
  filename = save_path + name + "_cdot.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if encoded_nz is None or ri is None or cb is None:
    sham_structure = get_nn_sham_structure(name, sham_code=sham_code, matrix=matrix)
    encoded_nz, ri, cb = sham_structure["enc_nz"], sham_structure["ri"], sham_structure["cb"]
  nz_list, _ = get_sham_splitted_nz_ri(encoded_nz=encoded_nz, ri=ri, cb=cb, code=sham_code["code"])
  int_list, col_end, cb_num = get_sham_cdot_structures(nz_list=nz_list, cb=cb, b=b)
  cdot_structure = {"int_list":int_list, "col_end":col_end, "ri":ri, "cb_num":cb_num}
  if save:
    with open(filename, "wb") as file:
      dump(cdot_structure, file)
  return cdot_structure

def compute_nn_spaces(nn_name, ham_code, sham_code, b=32):
  space = {"HAM":0, "HAM ub":0, "sHAM":0, "sHAM ub":0, "CSER":0, "CSC":0, "IM":0}
  ham_symbs_space, ham_fs_space, ham_fcl_space, ham_tab_space = code_space(symbs=ham_code["symbs"], fs=ham_code["fs"], fcl=ham_code["fcl"], tab=ham_code["par_tab"], b=b)
  space["HAM"] = ham_symbs_space + ham_fs_space + ham_fcl_space + ham_tab_space
  space["HAM ub"] = code_ub_space(k=len(ham_code["symbs"]), b=b)
  sham_symbs_space, sham_fs_space, sham_fcl_space, sham_tab_space = code_space(symbs=sham_code["symbs"], fs=sham_code["fs"], fcl=sham_code["fcl"], tab=sham_code["par_tab"], b=b)
  space["sHAM"] = sham_symbs_space + sham_fs_space + sham_fcl_space + sham_tab_space
  space["sHAM ub"] = code_ub_space(k=len(sham_code["symbs"]), b=b)
  total_entries = 0
  for matrix_name in get_matrices_names_from_unique_name(nn_name):
    matrix = load_nn_matrix(matrix_name)
    info = get_nn_matrix_info(name=matrix_name, matrix=matrix)
    n,m,s,k = info['n'], info['m'], info['s'], info['k']
    nm = n*m
    cser_dict = get_nn_cser_structure(name=matrix_name, matrix=matrix)
    sham_structure = get_nn_sham_structure(matrix_name, matrix=matrix, sham_code=sham_code)
    sham_nz, sham_ri = sham_structure['enc_nz'], sham_structure['ri']
    total_entries += nm
    matrix_symb2freq = get_symb2freq(matrix)
    matrix_code2freq = get_codeword2freq(symb2freq=matrix_symb2freq, code=ham_code['code'])
    space["HAM"] += ham_base_space(matrix_code2freq, b=b)
    space["HAM ub"] += ham_ub_psi(n=n, m=m, k=len(ham_code["symbs"]), b=b, no_code=True) * nm
    matrix_code2freq = get_codeword2freq(symb2freq=matrix_symb2freq, code=sham_code['code'], sham=True)        
    space["sHAM"] += sham_base_space(n, m, matrix_code2freq, sham_ri, b=b)
    space["sHAM ub"] += sham_ub_psi(n=n, m=m, s=s, k=len(sham_code["symbs"]), b=b, no_code=True) * nm
    space["CSER"] += cser_psi(n=n, m=m, cser_dict=cser_dict, b=b) * nm
    space["CSC"] += csc_psi(n=n, m=m, s=s, b=b) * nm
    space["IM"] += im_psi(n=n, m=m, k=k, b=b) * nm
  return {key:val/total_entries for key,val in space.items()}

def compute_nn_energies(nn_name, ham_code, sham_code, b=32):
  energy = {"HAM":0, "sHAM":0, "CSER":0, "CSC":0, "IM":0}
  total_entries = 0
  for matrix_name in get_matrices_names_from_unique_name(nn_name):
    matrix = load_nn_matrix(name=matrix_name)
    info = get_nn_matrix_info(name=matrix_name, matrix=matrix)
    n,m,s,k = info['n'], info['m'], info['s'], info['k']
    nm = n*m
    total_entries += nm
    matrix_symb2freq = get_symb2freq(matrix)
    ham_N_D = get_nn_ham_N_D(name=matrix_name, ham_code=ham_code, matrix_symb2freq=matrix_symb2freq, matrix=matrix)
    cser_dict = get_nn_cser_structure(name=matrix_name, matrix_symb2freq=matrix_symb2freq, matrix=matrix)
    if 0 in matrix_symb2freq.keys():
      matrix_symb2freq.pop(0)
    sham_N_D = get_nn_sham_N_D(name=matrix_name, sham_code=sham_code, matrix_symb2freq=matrix_symb2freq, matrix=matrix)
    energy["HAM"] += ham_per_elem_energy(n=n, m=m, s=s, k=len(ham_code["symbs"]), N=ham_N_D["N"], D=ham_N_D["D"], t=ham_code['t'], max_length=ham_code["lmax"], b=b) * nm
    energy["sHAM"] += sham_per_elem_energy(n=n, m=m, s=s, k=len(sham_code["symbs"]), N1=sham_N_D["N"], D1=sham_N_D["D"], t=sham_code['t'], max_length=sham_code["lmax"], b=b) * nm
    energy["CSER"] += cser_per_elem_energy(n=n, m=m, s=s, k=len(cser_dict["O"]), avg_k=info['avg_k'], cser_dict=cser_dict, b=b) * nm
    energy["CSC"] += csc_per_elem_energy(n=n, m=m, s=s, b=b) * nm
    energy["IM"] += im_per_elem_energy(n=n, m=m, s=s, k=k, b=b) * nm
  return {key:val/total_entries for key,val in energy.items()}

def compute_nn_times(nn_name, ham_code, sham_code, variant="partial", matrix=None, input_row_num=8, rep=25, num=1, nthread=8, cdot_path="../c_dot/", seed=0, b=32, matrix_type="float32"):
  times = {"HAM":0, "sHAM":0, "CSC":0, "IM":0}
  for matrix_name in get_matrices_names_from_unique_name(nn_name):
    if matrix is None:
      matrix = load_nn_matrix(matrix_name)
    n, m = matrix.shape
    csc_structure = csc_matrix(matrix)
    nz = csc_structure.data
    encoded_nz = get_sham_encoded_nz(nz, sham_code["code"])
    ri = csc_structure.indices
    cb = csc_structure.indptr
    ham_cdot_structure = get_nn_ham_cdot_structure(matrix_name, matrix=matrix, ham_code=ham_code, save=True)
    sham_cdot_structure = get_nn_sham_cdot_structure(matrix_name, matrix=matrix, encoded_nz=encoded_nz, ri=ri, cb=cb, sham_code=sham_code, save=True)
    ind_weights, vect_centers = get_im_structure(matrix)
    times_temp = compute_dot_times(n=n, m=m, ham_code=ham_code, ham_cdot_structure=ham_cdot_structure, sham_code=sham_code, sham_cdot_structure=sham_cdot_structure, variant=variant, sparse_matr=csc_matrix(matrix.T), indexes_weights=ind_weights, vect_centers=vect_centers, sham_row_index=csc_structure.indices, input_row_num=input_row_num, rep=rep, num=num, nthread=nthread, cdot_path=cdot_path, seed=seed, b=b, matrix_type=matrix_type)
    times["HAM"] += times_temp["HAM"]
    times["sHAM"] += times_temp["sHAM"]
    times["CSC"] += times_temp["CSC"]
    times["IM"] += times_temp["IM"]
  return times