import numpy as np
from numpy import errstate, inf

from sHAM import weightsharing, uweightsharing, stochastic

import gc

def ECSQ(weights_to_quantize, k, wanted_clusters, lambd=0.5, tr=0.001):
    J_last = inf
    stacked = [np.hstack(w) for w in weights_to_quantize]
    vect_weights = np.concatenate(stacked, axis=None)
    dim = len(vect_weights)
    w_split = np.array_split(np.sort(vect_weights), k)

    c = np.array([np.mean(w) for w in w_split]).reshape(-1,1)
    p = np.array([w.size/dim for w in w_split]).reshape(-1,1)

    dim_weights = [w.shape for w in weights_to_quantize]

    idx_layers = [np.zeros_like(w, dtype='int16') for w in weights_to_quantize]
    stacked_idx = [np.hstack(idx) for idx in idx_layers]
    vect_idx = np.concatenate(stacked_idx, axis=None)

    shift = 0
    for i in range(len(w_split)):
        vect_idx[shift:shift+len(w_split[i])] = i
        shift += len(w_split[i])


    while True:
        J = 0
        for i, elem in enumerate(vect_weights):
            with errstate(divide='ignore'):
                j_t = np.square(np.abs(elem-c)) - lambd*np.log(p)
            l = np.argmin(j_t)
            vect_idx[i] = l

            J += j_t[l]/dim
        for i in range(len(c)):
            c[i] = np.mean(vect_weights[vect_idx == i]) if len(vect_idx[vect_idx == i]) != 0 else -inf
            p[i] = len(vect_idx[vect_idx == i])/dim

        if J_last - J <= tr or wanted_clusters >= len(c[c!=-inf]):
            break
        J_last = J

    new_vect_idx = np.copy(vect_idx)
    for i_c in range(len(c)):
        if c[i_c] == -inf:
            new_vect_idx[vect_idx >= i_c] -= 1

    c = (c[(c != -inf)].reshape(-1,1))

    print(len(c))

    idx_layers = []
    for row, col in dim_weights:
        idx_layers.append((new_vect_idx[:row*col]).reshape(row,col))
        new_vect_idx = new_vect_idx[row*col:]

    return c, idx_layers


class uECQS_NN(uweightsharing.uWeightsharing_NN):
    def __init__(self, model, clusters_for_dense_layers, wanted_clusters, index_first_dense, tr=0.001, lamb=0.5, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        self.lamb = lamb
        self.tr = tr
        self.wanted_clusters = wanted_clusters
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_uECQS(self, list_trainable=None, untrainable_per_layers=None):

        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense

        weights_to_quantize = [list_weights[i] for i in range (d, len(list_weights), self.div)]

        self.centers, self.idx_layers = ECSQ(weights_to_quantize, self.clusters, lambd=self.lamb, tr=self.tr, wanted_clusters=self.wanted_clusters)

        self.clusters = len(self.centers)

        if not list_trainable:
            self.untrainable_per_layers = 0
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            self.untrainable_per_layers = untrainable_per_layers
            self.model.set_weights(self.recompose_weight(list_weights, True, untrainable_per_layers))
        gc.collect()
