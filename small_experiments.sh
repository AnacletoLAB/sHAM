#!/bin/bash 

# run the experiments on the smaller deep neural networks
cd nns
python3 experiments.py --matrices dta --compute_space --compute_energy --compute_time
# results are stored in nns/results

# run the experiments on the two smaller benchmark matrices
cd ../benchmark
python3 experiments.py --matrices orsreg_1 --matrices SiNa --compute_space --compute_energy --compute_time
# results are stored in benchmark/results

cd ..

