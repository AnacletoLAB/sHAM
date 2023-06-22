import sys
sys.path.append('../')
from utils import *

import warnings
import numpy as np

from scipy.io import mmread
from glob import glob
from pickle import dump, load

def load_benchmark_matrix(name, matrix_path="data/matrices/", matrix_type="float32"):
  for filename in (glob(matrix_path + "*.*")):
    if name in filename:
      ext = filename.split(".")[-1]
      if ext == "mtx":
        sparse_matrix = mmread(filename)
        return np.array(sparse_matrix.todense()).astype(matrix_type)
      if ext in ["csv", "txt"]:
        return np.loadtxt(filename, delimiter=",").astype(matrix_type)
  raise FileNotFoundError("There is no matrix with that name.")

def get_benchmark_symb2freq(name, matrix=None, matrices_path="data/matrices/", sham=False, save=False, save_path="data/symb2freq/"):
  filename = save_path + name + "_symb2freq.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name)
  symb2freq = get_symb2freq(matrix)
  if sham and 0 in symb2freq:
    symb2freq.pop(0)
  if save:
    if sham:
      warnings.warn("You are saving a symb2freq dictionary without 0. You will not be able to use it for the HAM format.")
    with open(filename, "wb") as file:
      dump(symb2freq, file)
  return symb2freq

def get_benchmark_huffman_code(name, t=None, symb2freq=None, symb2freq_path="data/symb2freq/", sham=False, save=False, save_path="data/"):
  ham_sham = "sham" if sham else "ham"
  filename = save_path + ham_sham + "/huffman_code/" + name + "_code.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if symb2freq is None:
    symb2freq = get_benchmark_symb2freq(name, sham=sham)
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

def get_benchmark_matrix_info(name, matrix=None, matrix_path="data/matrices/", save=False, save_path="/data/matrices/info/"):
  filename = save_path + "/" + name + "_info.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name, matrix_path=matrix_path)
  info = get_matrix_info(matrix)
  if save:
    with open(filename, "wb") as file:
      dump(info, file)            
  return info

def get_benchmark_cser_structure(name, matrix=None, symb2freq=None, matrix_path="data/matrices/", save=False, save_path="data/cser/"):
  filename = save_path + name + "_cser.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name, matrix_path=matrix_path)
  if symb2freq is None:
    symb2freq = get_benchmark_symb2freq(name, matrix=matrix)
  cser_dict = matrix_to_cser(matrix=matrix, symb2freq=symb2freq)
  if save:
    with open(filename, "wb") as file:
      dump(cser_dict, file)
  return cser_dict

def get_benchmark_ham_bitstream(name, matrix=None, ham_code=None, save=False, save_path="data/ham/"):
  filename = save_path + name + "_bitstream.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name)
  if ham_code is None:
    ham_code = get_benchmark_huffman_code(name)
  encoded_matrix = get_ham_encoded_matrix(matrix, ham_code["code"])
  bitstream = get_ham_bitstream(matrix, ham_code["code"], encoded_matrix=encoded_matrix)
  if save:
    with open(filename, "wb") as file:
      dump(bitstream, file)
  return bitstream

def get_benchmark_ham_N_D(name, matrix=None, symb2freq=None, code2freq=None, ham_code=None, b=32, save=False, save_path="data/ham/"):
  filename = save_path + name + "_N_D.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if symb2freq is None:
    symb2freq = get_benchmark_symb2freq(name)
  N, D = get_ham_N_D(symb2freq=symb2freq, code2freq=code2freq, code=ham_code["code"], t=ham_code["t"], b=b)
  N_D = {"N":N, "D":D}
  if save:
    with open(filename, "wb") as file:
      dump(N_D, file)
  return N_D

def get_benchmark_ham_cdot_structure(name, bitstream=None, matrix=None, ham_code=None, b=32, save=False, save_path="data/ham/cdot_structure/"):
  filename = save_path + name + "_cdot.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if bitstream is None:
    bitstream = get_benchmark_ham_bitstream(name, matrix=matrix, ham_code=ham_code)
  int_list, col_end = get_ham_cdot_structures(bitstream, b=b)
  cdot_structure = {"int_list":int_list, "col_end":col_end}
  if save:
    with open(filename, "wb") as file:
      dump(cdot_structure, file)
  return cdot_structure

def get_benchmark_sham_structure(name, sham_code=None, matrix=None, save=False, save_path="data/sham/"):
  filename = save_path + name + "_structure.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name)
  nz, ri, cb = get_csc_structure(matrix)
  encoded_nz = get_sham_encoded_nz(nz, sham_code["code"])
  structure = {"enc_nz":encoded_nz, "nz":nz, "ri":ri, "cb":cb}
  if save:
    with open(filename, "wb") as file:
      dump(structure, file)
  return structure

def get_benchmark_sham_N_D(name, sham_code, matrix=None, symb2freq=None, code2freq=None, encoded_nz=None, b=32, save=False, save_path="data/sham/"):
  filename = save_path + name + "_N_D.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if matrix is None:
    matrix = load_benchmark_matrix(name)
  if symb2freq is None:
    symb2freq = get_benchmark_symb2freq(name, matrix, sham=True)
  if 0 in symb2freq:
    symb2freq.pop(0)
  N, D = get_sham_N_D(t=sham_code["t"], symb2freq=symb2freq, code2freq=code2freq, code=sham_code["code"], b=b)
  N_D = {"N":N, "D":D}
  if save:
    with open(filename, "wb") as file:
      dump(N_D, file)
  return N_D

def get_benchmark_sham_cdot_structure(name, encoded_nz=None, ri=None, cb=None, matrix=None, sham_code=None, b=32, save=False, save_path="data/sham/cdot_structure/"):
  filename = save_path + name + "_cdot.pkl"
  if os.path.exists(filename):
    return load_if_exists(filename)
  if encoded_nz is None or ri is None or cb is None:
    sham_structure = get_benchmark_sham_structure(name, sham_code=sham_code, matrix=matrix)
    encoded_nz, ri, cb = sham_structure["enc_nz"], sham_structure["ri"], sham_structure["cb"]
  nz_list, _ = get_sham_splitted_nz_ri(encoded_nz=encoded_nz, ri=ri, cb=cb, code=sham_code["code"])
  int_list, col_end, cb_num = get_sham_cdot_structures(nz_list=nz_list, cb=cb, b=b)
  cdot_structure = {"int_list":int_list, "col_end":col_end, "ri":ri, "cb_num":cb_num}
  if save:
    with open(filename, "wb") as file:
      dump(cdot_structure, file)
  return cdot_structure

def compute_benchmark_spaces(name, matrix=None, info=None, symb2freq=None, ham_code=None, sham_code=None, ri=None, cser_dict=None, b=32):
  space = {"HAM":0, "HAM ub":0, "sHAM":0, "sHAM ub":0, "CSER":0, "CSC":0, "IM":0}
  if matrix is None:
    matrix = load_matrix(name)
  if info is None:
    info = get_benchmark_matrix_info(name=name, matrix=matrix)
  n,m,s,k = info['n'], info['m'], info['s'], info['k']
  if symb2freq is None:
    symb2freq = get_benchmark_symb2freq(name, matrix)
  if ham_code is None:
    ham_code = get_benchmark_huffman_code(name=name, matrix=matrix)
  if cser_dict is None:
    cser_dict = get_benchmark_cser_structure(name=name, matrix=matrix)
  if sham_code is None:
    sham_code = get_benchmark_huffman_code(name=name, matrix=matrix, sham=True)
  if ri is None:
    ri = get_benchmark_sham_structure(name, matrix=matrix, sham_code=sham_code)["ri"]
  code2freq = get_codeword2freq(symb2freq=symb2freq, code=ham_code["code"])
  space["HAM"] = ham_psi(n=n, m=m, code2freq=code2freq, symbs=ham_code["symbs"], fs=ham_code["fs"], fcl=ham_code["fcl"], tab=ham_code["par_tab"], b=b)
  space["HAM ub"] = ham_ub_psi(n=n, m=m, k=k, b=b)
  code2freq = get_codeword2freq(symb2freq=symb2freq, code=sham_code["code"], sham=True)
  space["sHAM"] = sham_psi(n=n, m=m, code2freq=code2freq, ri=ri, symbs=sham_code["symbs"], fs=sham_code["fs"], fcl=sham_code["fcl"], tab=sham_code["par_tab"], b=b)
  space["sHAM ub"] = sham_ub_psi(n=n, m=m, s=s, k=k, b=b)
  space["CSER"] = cser_psi(n=n, m=m, cser_dict=cser_dict, b=b)
  space["CSC"] = csc_psi(n=n, m=m, s=s, b=b)
  space["IM"] = im_psi(n=n, m=m, k=k, b=b)
  return space

def compute_benchmark_energies(name, matrix=None, info=None, ham_code=None, sham_code=None, cser_dict=None, b=32):
  energy = {"HAM":0, "sHAM":0, "CSER":0, "CSC":0, "IM":0}
  if matrix is None:
    matrix = load_matrix(name)
  if info is None:
    info = get_benchmark_matrix_info(name=name, matrix=matrix)
  n,m,s,k = info['n'], info['m'], info['s'], info['k']
  if ham_code is None:
    ham_code = get_benchmark_huffman_code(name=name, matrix=matrix)
  if sham_code is None:
    sham_code = get_benchmark_huffman_code(name=name, matrix=matrix, sham=True)
  if cser_dict is None:
    cser_dict = get_benchmark_cser_structure(name=name, matrix=matrix)
  ham_N_D = get_benchmark_ham_N_D(name=name, matrix=matrix, ham_code=ham_code)
  sham_N_D = get_benchmark_sham_N_D(name=name, matrix=matrix, sham_code=sham_code)
  energy["HAM"] = ham_per_elem_energy(n=n, m=m, s=s, k=k, N=ham_N_D["N"], D=ham_N_D["D"], t=ham_code['t'], max_length=ham_code["lmax"], b=b)
  energy["sHAM"] = sham_per_elem_energy(n=n, m=m, s=s, k=len(sham_code["symbs"]), N1=sham_N_D["N"], D1=sham_N_D["D"], t=sham_code['t'], max_length=sham_code["lmax"], b=b)
  energy["CSER"] = cser_per_elem_energy(n=n, m=m, s=s, k=len(cser_dict["O"]), avg_k=info['avg_k'], cser_dict=cser_dict, b=b)
  energy["CSC"] = csc_per_elem_energy(n=n, m=m, s=s, b=b)
  energy["IM"] = im_per_elem_energy(n=n, m=m, s=s, k=k, b=b)
  return energy

def compute_benchmark_times(name, ham_cdot_structure, sham_cdot_structure, matrix=None, ham_code=None, sham_code=None, variant="partial", input_row_num=8, rep=25, num=1, nthread=8, cdot_path="../c_dot/", seed=0, b=32, matrix_type="float32"):
  if matrix is None:
    matrix = load_benchmark_matrix(name)
  if ham_code is None:
    ham_code = get_benchmark_huffman_code(name)
  if sham_code is None:
    sham_code = get_benchmark_huffman_code(name, sham=True)
  n, m = matrix.shape
  ind_weights, vect_centers = get_im_structure(matrix)
  return compute_dot_times(n, m, ham_code=ham_code, ham_cdot_structure=ham_cdot_structure, sham_code=sham_code, sham_cdot_structure=sham_cdot_structure, variant=variant, sparse_matr=csc_matrix(matrix.T), indexes_weights=ind_weights, vect_centers=vect_centers, sham_row_index=sham_cdot_structure["ri"], input_row_num=input_row_num, rep=rep, num=num, nthread=nthread, cdot_path=cdot_path, seed=seed, b=b, matrix_type=matrix_type)