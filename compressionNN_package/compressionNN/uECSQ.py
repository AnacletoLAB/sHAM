import os
import time
from functools import reduce

import numpy as np
from numpy import errstate, inf
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense

from compressionNN import uCWS


def ECSQ(weights_to_quantize, k, wanted_clusters, lambd=0.5, tr=0.001):
    J_last = inf
    stacked = [np.vstack(w) for w in weights_to_quantize]
    vect_weights = np.concatenate(stacked, axis=None)
    dim = len(vect_weights)
    w_split = np.array_split(np.sort(vect_weights), k)

    c = np.array([np.mean(w) for w in w_split]).reshape(-1,1)
    p = np.array([w.size/dim for w in w_split]).reshape(-1,1)

    dim_weights = [w.shape for w in weights_to_quantize]

    idx_layers = [np.zeros_like(w, dtype='int16') for w in weights_to_quantize]
    stacked_idx = [np.vstack(idx) for idx in idx_layers]
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
    for s in dim_weights:
        idx_layers.append((new_vect_idx[:reduce(lambda a,b: a*b, s)]).reshape(s))
        new_vect_idx = new_vect_idx[reduce(lambda a,b: a*b, s):]

    return c, idx_layers



class uECSQ(uCWS.uCWS):
    def __init__(self, model, clusters_for_conv_layers, clusters_for_dense_layers, wanted_clusters_cnn, wanted_clusters_fc, tr=0.001, lamb=0.5):
        self.model = model
        self.clusters_fc = clusters_for_dense_layers # 0 disables Fully Connected clustering
        self.clusters_cnn = clusters_for_conv_layers # 0 disables CNN clustering
        self.wanted_clusters_fc = wanted_clusters_fc
        self.wanted_clusters_cnn = wanted_clusters_cnn
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
        self.lamb_fc = lamb
        self.lamb_cnn = lamb
        self.tr = tr
    
    def __del__(self):
        if os.path.exists(self.timestamped_filename):
            os.remove(self.timestamped_filename)


    def apply_uECSQ(self):
        # Dense layers
        if self.clusters_fc > 0:
            self.idx_layers_fc, self.centers_fc, self.clusters_fc = self.apply_private_uecsq(Dense, self.clusters_fc, self.wanted_clusters_fc, self.lamb_fc)
        # Convolutional layers
        if self.clusters_cnn > 0:
            self.idx_layers_cnn, self.centers_cnn, self.clusters_cnn = self.apply_private_uecsq((Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.wanted_clusters_cnn, self.lamb_cnn)


    def apply_private_uecsq(self, instan, perc, wanted_clusters, lamb):
        massive_weight_list = self.extract_weights(instan, perc)
        centers, idx_layers = ECSQ(massive_weight_list, k=perc, wanted_clusters=wanted_clusters, lambd=lamb, tr=self.tr)
        perc = len(centers)
        self.recompose_weight(instan, perc, centers, idx_layers)

        return idx_layers, centers, perc


    def tune_lambda(self, lambdaList):
        if self.clusters_fc > 0:
            print("Tuning FC lambda")
            self.lamb_fc = self.tune_lambda_private_uecsq(lambdaList, Dense, self.clusters_fc, self.wanted_clusters_fc)
        if self.clusters_cnn > 0:
            print("Tuning CNN lambda")
            self.lamb_cnn = self.tune_lambda_private_uecsq(lambdaList, (Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.wanted_clusters_cnn)

    def tune_lambda_private_uecsq(self, lambdaList, instan, perc, wanted_clusters):
        final_lambd = 0.
        abs_distance = 10000
        for i, lam in enumerate(lambdaList):
            print(lam, end=' ')
            massive_weight_list = self.extract_weights(instan, perc)
            c, _ = ECSQ(massive_weight_list, k=3*wanted_clusters, wanted_clusters=wanted_clusters, lambd=lam)
            # print(len(c))
            if len(c) >= wanted_clusters:
                final_lambd = lambdaList[i] if (abs(len(c) - wanted_clusters) <= abs_distance) else lambdaList[i-1]
                print("best", final_lambd)
                break
            abs_distance = abs(len(c) - wanted_clusters)
        if final_lambd == 0:
            final_lambd = lambdaList[-1]
        return final_lambd