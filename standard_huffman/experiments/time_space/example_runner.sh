#!/bin/bash

# This example evaluates the compression rate of HAM and sHAM on all the VGG19 models
# trained on the MNIST dataset which underwent pruning and quantization via uCWS
python uws_testing_time_space.py -t all -d ../performance_eval/VGG19/VGG19-MNIST/pruCWS/ \
-m ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 -s mnist -q 0


# This example evaluates the compression rate of HAM on all the VGG19 models
# trained on the CIFAR10 dataset which underwent quantization via uECSQ
python uws_testing_time_space.py -t ham -d ../performance_eval/VGG19/VGG19-CIFAR10/uECSQ/ \
-m ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 -s cifar10 -q 0


# This example evaluates the compression rate of sHAM on all the DeepDTA models
# trained on the CDavis dataset which underwent quantization via uPWS
python uws_testing_time_space.py -t sham -d ../performance_eval/DeepDTA/deepDTA_davis/uPWS/ \
-m ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 -s davis -q 0
