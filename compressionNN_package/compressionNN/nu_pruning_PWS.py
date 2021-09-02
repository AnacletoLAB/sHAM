import numpy as np
from compressionNN import nu_pruning_CWS
from compressionNN import nu_PWS

from scipy import ndimage

import gc

from numpy.random import seed
seed(1)

def find_index_first_dense(list_weights):
    i = 0
    for w in list_weights:
        if len(w.shape)==2:
            return i
        i += 1

def centroid_gradient_matrix_combined(idx_matrix,gradient,cluster,mask):
    gradient += 0.0000000001
    gradient[np.logical_not(mask)] = 0
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)

class nu_pruning_PWS(nu_pruning_CWS.nu_pruning_CWS, nu_PWS.nu_PWS):
    def __init__(self, model, perc_prun_for_dense, bits_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.bits = bits_for_dense_layers
        self.clusters = [ 2**i for i in bits_for_dense_layers]
        self.index_first_dense = index_first_dense
        self.perc_prun_for_dense = perc_prun_for_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_pruning_stochastic(self, list_trainable=None, untrainable_per_layers=None):
        self.apply_pruning(list_trainable, untrainable_per_layers) #crea masks e setta pesi prunati nel modello
        self.apply_stochastic(list_trainable, untrainable_per_layers) #crea centri e matrici degli indici e setta pesi ws nel modello
        gc.collect()

    def get_weightsharing_weigths(self):
        return self.model.get_weights()
