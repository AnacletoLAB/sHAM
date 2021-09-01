import os
import time

import numpy as np
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense

from compressionNN import nu_PWS, uCWS

class uPWS(uCWS.uCWS):
    def __init__(self, model, clusters_for_conv_layers, clusters_for_dense_layers):
        self.model = model
        self.clusters_fc = clusters_for_dense_layers # 0 disables Fully Connected clustering
        self.clusters_cnn = clusters_for_conv_layers # 0 disables CNN clustering
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
    
    def __del__(self):
        if os.path.exists(self.timestamped_filename):
            os.remove(self.timestamped_filename)

    def apply_uPWS(self):
        # Dense layers
        if self.clusters_fc > 0:
            self.idx_layers_fc, self.centers_fc = self.apply_private_upq(Dense, self.clusters_fc)
        # Convolutional layers
        if self.clusters_cnn > 0:
            self.idx_layers_cnn, self.centers_cnn = self.apply_private_upq((Conv1D, Conv2D, Conv3D), self.clusters_cnn)

    def apply_private_upq(self, instan, perc):
        dtype = "uint8" if perc <= 255 else "uint16"
        massive_weight_list = self.extract_weights(instan, perc)
        all_vect_weights = np.concatenate([x.reshape(-1,1) for x in massive_weight_list], axis=None).reshape(-1,1)

        n_intervals = perc - 1
        values_dict, intervals = nu_PWS.generate_intervals(all_vect_weights, n_intervals)
        indices_dict = {v: k for k,v in values_dict.items()}

        vect_bin = np.vectorize(nu_PWS.binarize)
        vect_bin.excluded.add(1)
        vect_bin.excluded.add(2)
        idx_layers = [vect_bin(massive_weight_list[i], intervals, indices_dict).astype(dtype) for i in range(len(massive_weight_list))]
        
        centers = np.zeros(shape=(perc, 1), dtype="float32")
        for k, v in values_dict.items():
            centers[k] = v

        self.recompose_weight(instan, perc, centers, idx_layers)

        return idx_layers, centers
