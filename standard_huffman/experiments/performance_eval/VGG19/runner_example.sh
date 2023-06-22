#!/bin/bash

# This example applies pruning and uECSQ quantization to the VGG19 model trained on the MNIST dataset.
# Pruning is only applied to fully connected layers (percentage 90%), while clustering is applied to
# CNN and FC layers with 64 clusters
python compression.py --compression pruECSQ --net original_nets/VGG19-MNIST.h5 --dataset MNIST --clusterfc 64 --clustercnn 64 --prfc 90

# This example applies pruning and uCWS quantization to the VGG19 model trained on the CIFAR10 dataset.
# Compression is only applied to convolutional layers, with pruning percentage ranging from 60% to 99%,
# while clustering fixed to 256 clusters
for p in 60 70 80 90 95 96 97 98 99
do
	python compression.py --compression pruCWS --net original_nets/VGG19-CIFAR10.h5 --dataset CIFAR10 --clustercnn 256 --prcnn $p
done

