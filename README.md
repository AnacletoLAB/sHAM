# Compression strategies and space-conscious representations for deep neural networks
This repository contains the code allowing to reproduce the results described in G. Marinò et al.,
_Compact representations of convolutional neural networks via weight pruning and quantization_, 
the extended and revised work of the original paper presented at [ICPR](https://www.micc.unifi.it/icpr2020/) 
conference. The original contribution is [available](ICPR2020_sHAM.pdf) for reviewing purposes of 
a companion paper currently submitted to the [RRPR](https://rrpr2020.sciencesconf.org/) conference.
Also, the original package is available on [this repository](https://github.com/giosumarin/ICPR2020_sHAM).

This package introduces new compression quantization strategies; also, their application is no
longer limited to Fully Connected layers, as now Convolutional Layers are supported.

We also introduced a new optimized dot procedure written in C++ which reduces the overal execution
time for training and testing of HAM and sHAM compressed models.

The experiments folder contins the basic scripts used for performing the tests presented in the
forthcoming paper.


## Getting Started

### Prerequisites

* Install `python3`, `python3-pip` and `python3-venv` (Debian 10.6)
* Make sure that `python --version` starts by 3 or execute `alias python='pyhton3'` in the shell before executing `runner.sh`.
* For CUDA configuration (if GPU is available) follow https://www.tensorflow.org/install/gpu.

### Compiling megaDot
At present time, we do not provide automatic compile for the C++ dot. However, we use
cmake for automatically download and compile all the dependencies of the c++ code.
For this reason, the only requirements are the installation of `make`, `cmake` and of a relatively
new C++ compiler (code was tested with various releases of g++ > 5.4.0)

From the root of this repository:
* create a build directory:  `mkdir build & cd build`
* call cmake:  `cmake ../megaDot`
* compile:  `make`

This procedure generates the `libmegaDot.so` library file. A pre-compiled version for Linux x86-64 is present
in the experiments/time_space directory


### Additional data
The trained models - as well as data required by DeepDTA - are rather big, so they are not versioned. Rather, 
they are available for [download](https://mega.nz/file/jkcmyJAB#XHIRNpGP7_iaK9Y_6ZjMk_5RhtnZ4I0FId9R6mjy7KY).
Once downloaded, move the `sHAM_data.zip` to the root of this repository and unzip it: `unzip sHAM_data.zip`.
By doing so, the trained models and all the supporting data will be copied to the correct locations inside
the directory tree

## Usage
Pruning + quantization and network compression are separately executed in two stages.
1. To apply pruning and/or quantization to a model, we provide the `compress.py` script in the
experiments/performance_eval directory. These script are customized for VGG and DeepDTA networks
and a minimal runner script is contained in each network sub-directory
2. To compress a trained network with either HAM or sHAM we provide the `uws_testing_time_space.py`
example script in the experiments/time_space directory, as well with a sample runner script.

The directory experiments/SOTA_comparison contains a script which compares HAM and sHAM to several
other compression methods from the literature. Comparison takes into account compression ratios and
the execution time of a dot product on the Fully Connected layers of a trained VGG model.
To use the script:
* copy a trained VGG .h5 model file into the experiments/SOTA_comparison/csv_data directory
* run the generate_csv.py script
* run the compare_VGG.py script
