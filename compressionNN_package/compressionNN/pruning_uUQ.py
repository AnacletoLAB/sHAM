import time

from compressionNN import pruning_uCWS
from compressionNN import uUQ

class pruning_uUQ(pruning_uCWS.pruning_uCWS, uUQ.uUQ):
    def __init__(self, model, perc_prun_for_dense, perc_prun_for_cnn, clusters_for_dense_layers, clusters_for_conv_layers, delta=0, b=0.):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense  # 0 disabilita pruning per livelli densi
        self.perc_prun_for_cnn = perc_prun_for_cnn      # 0 disabilita pruning per livelli convoluzionali
        self.clusters_fc = clusters_for_dense_layers
        self.clusters_cnn = clusters_for_conv_layers
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
        self.b = b
        self.delta = delta

    def apply_pr_uUQ(self):
        self.apply_pruning() #crea masks e setta pesi prunati nel modello
        self.apply_uUQ() #crea centri e matrici degli indici e setta pesi ws nel modello
