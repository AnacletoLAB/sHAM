#!/bin/bash

# This example applies uECQS quantization to the DeepDTA model trained on the Davis dataset.
# Clustering is applied to CNN (128 clusters) and FC (16 clusters) layer.
python compression.py --compression uECSQ --net original_nets/deepDTA_davis.h5 --dataset DAVIS --clusterfc 16 --clustercnn 128

# This example applies pruning and uCWS quantization to the DeepDTA model trained on the Kiba dataset.
# Pruning is only applied to fully connected layers (percentage 60%), while clustering is applied to 
# fully connected (k=32) and convolutional (k ranging from 16 to 128) layers
for k in 16 32 64 96 128
do
	python compression.py --compression pruCWS --net original_nets/deepDTA_kiba.h5 --dataset KIBA --clustercnn $k --clusterfc 32 --prfc 60
done

