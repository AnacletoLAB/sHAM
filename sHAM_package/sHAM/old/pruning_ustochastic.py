from sHAM import pruning_uweightsharing
from sHAM import ustochastic

def find_index_first_dense(list_weights):
    i = 0
    for w in list_weights:
        if len(w.shape)==2:
            return i
        i += 1

class PruninguStochastic_NN(pruning_uweightsharing.PruninguWeightsharing_NN, ustochastic.uStochastic_NN):
    def __init__(self, model, perc_prun_for_dense, clusters_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        self.perc_prun_for_dense = perc_prun_for_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_pr_ustochastic(self, list_trainable=None, untrainable_per_layers=None):
        self.apply_pruning(list_trainable, untrainable_per_layers) #crea masks e setta pesi prunati nel modello
        self.apply_ustochastic(list_trainable, untrainable_per_layers) #crea centri e matrici degli indici e setta pesi ws nel modello

    def get_weightsharing_weigths(self):
        return self.model.get_weights()
