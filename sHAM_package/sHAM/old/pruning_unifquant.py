from keras import backend as k
from keras.utils import np_utils
from tensorflow import keras
from sHAM import pruning_uweightsharing
from sHAM import unifquant


def find_index_first_dense(list_weights):
    i = 0
    for w in list_weights:
        if len(w.shape)==2:
            return i
        i += 1

class PruninguUniformQuant_NN(pruning_uweightsharing.PruninguWeightsharing_NN, unifquant.uUniformQuant_NN):
    def __init__(self, model, perc_prun_for_dense, index_first_dense, delta=0, b=0., apply_compression_bias=False, clusters_for_dense_layers=None, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        self.perc_prun_for_dense = perc_prun_for_dense
        self.b = b
        self.delta = delta
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_pr_uunifquant(self, list_trainable=None, untrainable_per_layers=None):
        self.apply_pruning(list_trainable, untrainable_per_layers) #crea masks e setta pesi prunati nel modello
        self.apply_uUniformQuant(list_trainable, untrainable_per_layers) #crea centri e matrici degli indici e setta pesi ws nel modello

    def get_weightsharing_weigths(self):
        return self.model.get_weights()
