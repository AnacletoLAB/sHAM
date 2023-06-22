import sys
sys.path.append('../')
from utils import *
from benchmark_utils import *

import click
import pandas as pd

# If you run the script without arguments, all the benchmark matrices will be tested.
# Since the encoding of 'census' and 'imagenet' matrices might take some time and memory, you can run the test on the other three benchmark matrices using the following command:
## 'python3 experiments.py --compute_space --compute_energy --compute_time --matrices orsreg_1 --matrices SiNa --matrices covtype'
@click.command()
@click.option('--matrices', multiple=True, default=["orsreg_1", "SiNa", "covtype", "census", "imagenet"], help="allows to choose the benchmark matrices on which the test will be run; repeat for each matrix")
@click.option('--compute_space', is_flag=True, default=False, help="use this option to compute the space occupancy of the selected matrices")
@click.option('--compute_energy', is_flag=True, default=False, help="use this option to compute the energy of the selected matrices")
@click.option('--compute_time', is_flag=True, default=False, help="use this option to compute the dot time of the selected matrices")
def main(matrices, compute_space, compute_energy, compute_time):
  
  b = 32
  matrix_type = "float" + str(b)
  input_row_num = 8
  rep = 25
  num = 1
  nthread = 8
  seed = 0

  save_path = "results/"

  benchmark_spaces, benchmark_energies, benchmark_times = {}, {}, {}
  for name in matrices:
    print(name)
    matrix = load_benchmark_matrix(name)
    info = get_benchmark_matrix_info(name, matrix=matrix)

    symb2freq = get_benchmark_symb2freq(name, matrix=matrix)
    ham_code = get_benchmark_huffman_code(name, symb2freq=symb2freq, sham=False, save=True)
    sham_code = get_benchmark_huffman_code(name, symb2freq=symb2freq, sham=True, save=True)

    cser_dict = get_benchmark_cser_structure(name, matrix=matrix, symb2freq=symb2freq, save=True)

    if compute_space:
      print("\tComputing space occupancy...")
      benchmark_spaces[name] = compute_benchmark_spaces(name=name, matrix=matrix, info=info, ham_code=ham_code, sham_code=sham_code, cser_dict=cser_dict, b=b)
      benchmark_spaces_df = pd.DataFrame(benchmark_spaces).T
      benchmark_spaces_df.to_csv(save_path + "space.csv")

    if compute_energy:
      print("\tComputing dot energy...")
      benchmark_energies[name] = compute_benchmark_energies(name, matrix=matrix, info=info, ham_code=ham_code, sham_code=sham_code, cser_dict=cser_dict, b=b)
      benchmark_energies_df = pd.DataFrame(benchmark_energies).T
      benchmark_energies_df.to_csv(save_path + "energy.csv")
      
    if compute_time:
      print("\tComputing dot time...")          
      ham_cdot_structure = get_benchmark_ham_cdot_structure(name, matrix=matrix, ham_code=ham_code, b=b, save=True)
      sham_cdot_structure = get_benchmark_sham_cdot_structure(name, matrix=matrix, sham_code=sham_code, b=b, save=True)
      benchmark_times[name] = compute_benchmark_times(name, ham_code=ham_code, ham_cdot_structure=ham_cdot_structure, sham_code=sham_code, sham_cdot_structure=sham_cdot_structure, variant="partial", input_row_num=input_row_num, rep=rep, num=num, nthread=nthread, seed=seed, b=b, matrix_type="float32")
      benchmark_times_df = pd.DataFrame(benchmark_times).T
      benchmark_times_df.to_csv(save_path + "time.csv")

    print()

if __name__ == '__main__':
  main()
