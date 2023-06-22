import sys
sys.path.append('../')
from utils import *
from nns_utils import *

import click
import pandas as pd

@click.command()
@click.option('--matrices', multiple=True, default=["dta", "vgg"], help="allows to choose the neural networks matrices on which the test will be run")
@click.option('--compute_space', is_flag=True, default=False, help="use this option to compute the space occupancy of the selected matrices")
@click.option('--compute_energy', is_flag=True, default=False, help="use this option to compute the energy of the selected matrices")
@click.option('--compute_time', is_flag=True, default=False, help="use this option to compute the dot time of the selected matrices")
def main(matrices, compute_space, compute_energy, compute_time):
  
  b = 32
  input_row_num = 8
  rep = 25
  num = 1
  nthread = 8
  seed = 0

  save_path = "results/"

  nn_spaces, nn_energies, nn_times = {}, {}, {}
  for vgg_or_dta in matrices:
    nn_names = get_unique_names(vgg_or_dta, reverse=False)
    for nn_name in nn_names:
      print(nn_name)
      symb2freq = get_nn_symb2freq(nn_name)
      ham_code = get_nn_huffman_code(nn_name, symb2freq=symb2freq, sham=False, save=True)
      sham_code = get_nn_huffman_code(nn_name, symb2freq=symb2freq, sham=True, save=True)

      if compute_space:
        print("\tComputing space occupancy...")
        nn_spaces[nn_name] = compute_nn_spaces(nn_name=nn_name, ham_code=ham_code, sham_code=sham_code, b=b)
        nn_space_df = pd.DataFrame(nn_spaces).T
        nn_space_df.to_csv(save_path + "space.csv")

      if compute_energy:
        print("\tComputing dot energy...")
        nn_energies[nn_name] = compute_nn_energies(nn_name, ham_code=ham_code, sham_code=sham_code, b=b)
        nn_energies_df = pd.DataFrame(nn_energies).T
        nn_energies_df.to_csv(save_path + "energy.csv")
        
      if compute_time:
        print("\tComputing dot time...")
        nn_times[nn_name] = compute_nn_times(nn_name, ham_code=ham_code, sham_code=sham_code, variant="partial", input_row_num=input_row_num, rep=rep, num=num, nthread=nthread, seed=seed, b=b, matrix_type="float32")
        nn_times_df = pd.DataFrame(nn_times).T
        nn_times_df.to_csv(save_path + "time.csv")

      print()

if __name__ == '__main__':
  main()
