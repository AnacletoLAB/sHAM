import numpy as np
from numpy import errstate, inf

from sHAM import weightsharing, uweightsharing, stochastic

import gc


def unif_quant(value, delta, b):
    return delta * round((value+b)/delta)-b

class uUniformQuant_NN(uweightsharing.uWeightsharing_NN):
    def __init__(self, model, index_first_dense, delta=0, b=0., apply_compression_bias=False, clusters_for_dense_layers=None, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        self.b = b
        self.delta = delta
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_uUniformQuant(self, list_trainable=None, untrainable_per_layers=None):

        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense

        uniform_quantization = np.vectorize(unif_quant)


        weights_to_quantize = [list_weights[i] for i in range (d, len(list_weights), self.div)]
        stacked = [np.hstack(w) for w in weights_to_quantize]
        vect_weights = np.concatenate(stacked, axis=None)

        if not self.clusters:
            uniformed = uniform_quantization(vect_weights, self.delta, self.b)
            self.centers = np.unique(uniformed).reshape(-1,1)
            self.clusters = len(self.centers)
            self.idx_layers = [weightsharing.redefine_weights(list_weights[i], self.centers) for i in range (d, len(list_weights), self.div)]
        else:
            dd = 10**6
            ok = False
            for delta in (range(100, dd+500, 500)):#[0.000001, 0.00001, 0.000015, 0.00002, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.5, 1]:
                uniformed = uniform_quantization(vect_weights, delta/dd, self.b)
                unique = np.unique(uniformed)
                print(delta/dd, len(unique))
                if len(unique) <= self.clusters:
                    ok = True
                    print("used ",delta/dd, len(unique))
                    self.centers = unique.reshape(-1,1)
                    self.clusters = len(unique)
                    self.idx_layers = [weightsharing.redefine_weights(list_weights[i], self.centers) for i in range (d, len(list_weights), self.div)]
                    break
            if not ok:
                uniformed = uniform_quantization(vect_weights, 1, self.b)
                unique = np.unique(uniformed)
                self.centers = unique.reshape(-1,1)
                self.clusters = len(unique)
                self.idx_layers = [weightsharing.redefine_weights(list_weights[i], self.centers) for i in range (d, len(list_weights), self.div)]

        if not list_trainable:
            self.untrainable_per_layers = 0
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            self.untrainable_per_layers = untrainable_per_layers
            self.model.set_weights(self.recompose_weight(list_weights, True, untrainable_per_layers))
        gc.collect()
