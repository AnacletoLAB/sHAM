import numpy as np

from sHAM import weightsharing, uweightsharing, stochastic

import gc


class uStochastic_NN(uweightsharing.uWeightsharing_NN):
    def __init__(self, model, clusters_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_ustochastic(self, list_trainable=None, untrainable_per_layers=None):

        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense

        dtype = "uint8" if self.clusters <= 255 else "uint16"

        vect_weights = [np.hstack(list_weights[i]).reshape(-1,1) for i in range (d, len(list_weights), self.div)]
        all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
        n_intervals = self.clusters - 1
        values_dict, intervals = stochastic.generate_intervals(all_vect_weights, n_intervals)
        indices_dict = {v: k for k,v in values_dict.items()}

        vect_bin = np.vectorize(stochastic.binarize)
        vect_bin.excluded.add(1)
        vect_bin.excluded.add(2)
        self.idx_layers = [vect_bin(list_weights[i], intervals, indices_dict).astype(dtype) for i in range (d, len(list_weights), self.div)]

        self.centers = np.zeros(shape=(self.clusters, 1), dtype="float32")

        for k, v in values_dict.items():
            self.centers[k] = v

        if not list_trainable:
            self.untrainable_per_layers = 0
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            self.untrainable_per_layers = untrainable_per_layers
            self.model.set_weights(self.recompose_weight(list_weights, True, untrainable_per_layers))
        gc.collect()
