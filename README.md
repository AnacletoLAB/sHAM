# Efficient and Compact Representations of Deep Neural Networks via Entropy Coding

## HAM and sHAM
We have improved our HAM and sHAM compressed matrices formats by using a Canonical Huffman code that does not require to access the Huffman tree to perform decoding.

## Experiments
We have tested the space occupancy, dot time and energy requirements of HAM and sHAM and compared them to other state of the art techniques:
* Compressed Sparse Column (CSC)
* Index Map (IM)
* Compressed Shared Elements Row (CSER)

## Data
The tests have been performed on the dense layers of two deep neural networks and on five benchmark matrices (check the paper for the references).
#### DNNs (with different combinations of pruning and quantization):
* **VGG19**, trained on the MNIST dataset.
* **DeepDTA**, trained on the DAVIS dataset.
#### Benchmark matrices:
| **Matrix** |  **rows**  | **columns** |   **1-sparsity**  | **distinct values** |
|:----------:|:-------:|:-----:|:--------:|:-----:|
|  **orsreg_1**  |   2205  |  2205 | 2.907e−3 |  111  |
|    **SiNa**    |   5743  |  5743 | 6.027e−6 | 24317 |
|   **Covtype**  |  581012 |   54  | 2.200e−1 |  6682 |
|   **Census**   | 2458285 |   68  | 5.697e−1 |   45  |
|  **ImageNet**  | 1262102 |  900  | 3.099e−1 |  824  |

## Getting started

### Requirements and installation
* Install `python3`, `python3-pip` and `python3-venv`.
* Make sure that `python --version` starts by 3 or execute `alias python='python3'` in the shell.
* Create a virtual environment and activate it: 
  ```
  python3 -m venv /path/to/new/virtual/environment
  source /path/to/new/virtual/environment/bin/activate
  ```

* Install the required dependencies:
  ```
  pip install -r ./requirements.txt
  ```

## Repository content description
The python script *utils.py* contains all the primitives used to build the compressed structures and to compute the associated space occupancy, matrix-vector multiplication time and energy requirements.

The main folder contains three directories:
* *c_dot*, containing the source code written in C language and the executable files to perform the matrix-vector multiplication of our sparse formats.

* *nns*, containing a Python script (*experiments.py*) to reproduce the experiments shown in the paper, with all the required data; the results in *.csv* format are stored in the *result* folder.
* *benchmark*, organized as the *nns* directory.

In the main folder, there are also two *.sh* scripts to run the experiments (detailed below, in the *Usage* section).

## Usage
We provide two simple *.sh* scripts to run the experiments:
* *small_experiments.sh* computes space occupancy, dot time and energy requirements for *DeepDTA*, *orsreg_1* and *SiNa*
* *full_experiments.sh* computes space occupancy, dot time and energy requirements for all the matrices

**Warning:** *Census* and *ImageNet* might take some time and memory to build the compressed structures.

It is advisable to compile the *.c* files for the dot product, by running the following:
  ```
  gcc -w -fPIC -shared -o ham_dot_partial.so ham_dot_partial.c -pedantic -Wall -pthread
gcc -w -fPIC -shared -o sham_dot_partial.so sham_dot_partial.c -pedantic -Wall -pthread
  ```


## Downloading the matrices
In order to reproduce the experiments, you should download [this](https://www.mediafire.com/file/m0cjv959w4melbu/sHAM_data.tar.gz/file) *.tar.gz* file, containing all the matrices, and merge the repository with the downloaded data.




