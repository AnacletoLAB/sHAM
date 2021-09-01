import numpy as np
from scipy import ndimage

from compressionNN import nu_CWS

import gc


def find_index_first_dense(list_weights):
    i = 0
    for w in list_weights:
        if len(w.shape)==2:
            return i
        i += 1

def idx_matrix_to_matrix(idx_matrix,centers):
    return centers[idx_matrix.reshape(-1,1)].reshape(idx_matrix.shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)

#STOCHASTIC COMPRESSION FUNCTIONS
def generate_intervals(W, n_intervals):
    intervals = []
    values_dict = {}
    for i in range(n_intervals):
        lower_extreme = np.quantile(W, i/n_intervals)
        upper_extreme = np.quantile(W, (i+1)/n_intervals)
        intervals.append((lower_extreme, upper_extreme))
        values_dict[i] = lower_extreme
    #The last extreme must also be included
    values_dict[len(values_dict)]= intervals[-1][1]
    return values_dict , intervals

def get_interval(w, intervals):
    interval = None
    for i in intervals:
        if w >= i[0] and w < i[1]:
            interval = i
            break
    if not interval:
        interval = intervals[-1]
    return interval

def binarize(w, intervals, indices_dict):
    [v,V] = get_interval(w, intervals)
    return indices_dict[V] if np.random.uniform() <= (w-v)/(V-v) else indices_dict[v]

def stochastic_compression(W, b, dtype=np.uint8):
    n_intervals = (2**b) - 1
    values_dict, intervals = generate_intervals(W, n_intervals)
    indices_dict = {v: k for k,v in values_dict.items()}
    vect_bin = np.vectorize(binarize)
    vect_bin.excluded.add(1)
    vect_bin.excluded.add(2)
    return values_dict, vect_bin(W, intervals, indices_dict).astype(dtype)
#END STOCHASTIC COMPRESSION FUNCTIONS

class nu_PWS(nu_CWS.nu_CWS):
    def __init__(self, model, bits_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.bits = bits_for_dense_layers
        self.clusters = [ 2**i for i in bits_for_dense_layers]
        self.index_first_dense = index_first_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_stochastic(self, list_trainable=None, untrainable_per_layers=None):

        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense
        
        dtypelist = [ "uint8" if i <= 8 else "uint16" for i in self.bits]
        result = [stochastic_compression(list_weights[i], self.bits[(i-d)//self.div], dtypelist[(i-d)//self.div]) for i in range (d, len(list_weights), self.div)]
        
        values = [ v for (v , _) in result]
        self.centers = []
        i = 0
        for d in values:
            vect = np.zeros(shape=(self.clusters[i], 1), dtype="float32")
            for key, v in d.items():
                vect[key] = v
            self.centers.append(vect)
            i = i+1
        
        self.idx_layers = [ m for (_ , m) in result]
        
        if not list_trainable:
            self.untrainable_per_layers = 0
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            self.untrainable_per_layers = untrainable_per_layers
            self.model.set_weights(self.recompose_weight(list_weights, True, untrainable_per_layers))
        gc.collect()

    def recompose_weight(self, list_weights, trainable_vars=False, untrainable_per_layers=None):
        if not trainable_vars:
            d = self.index_first_dense
            return list_weights[:d]+[(idx_matrix_to_matrix(self.idx_layers[(i-d)//self.div], self.centers[(i-d)//self.div])) if i%self.div==0 else (list_weights[i]) for i in range(d,len(list_weights))]
        else:
            div = self.div + untrainable_per_layers
            list_weights = self.trainable_to_weights(self.model.get_weights(), list_weights, untrainable_per_layers)
            d = find_index_first_dense(list_weights)
            return list_weights[:d]+[(idx_matrix_to_matrix(self.idx_layers[(i-d)//div], self.centers[(i-d)//div])) if i%div==0 else (list_weights[i]) for i in range(d,len(list_weights))]
