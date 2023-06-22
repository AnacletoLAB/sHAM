import os
import time

import numpy as np
from numpy import errstate, inf
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense

from sHAM import uCWS

def unif_quant(value, delta, b):
    return delta * round((value+b)/delta)-b

class uUQ(uCWS.uCWS):
    def __init__(self, model, clusters_for_conv_layers, clusters_for_dense_layers, delta=0, b=0.):
        self.model = model
        self.clusters_fc = clusters_for_dense_layers # 0 disables Fully Connected clustering; -1 for default
        self.clusters_cnn = clusters_for_conv_layers # 0 disables CNN clustering; -1 for default
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
        self.b = b
        self.delta = delta

    def __del__(self):
        if os.path.exists(self.timestamped_filename):
            os.remove(self.timestamped_filename)

    def apply_uUQ(self):
        # Dense layers
        if (self.clusters_fc > 0) or (self.clusters_fc == -1):
            self.idx_layers_fc, self.centers_fc, self.clusters_fc = self.apply_private_uuq(Dense, self.clusters_fc)
        # Convolutional layers
        if (self.clusters_cnn > 0) or (self.clusters_cnn == -1):
            self.idx_layers_cnn, self.centers_cnn, self.clusters_cnn = self.apply_private_uuq((Conv1D, Conv2D, Conv3D), self.clusters_cnn)


    def apply_private_uuq(self, instan, perc):
        uniform_quantization = np.vectorize(unif_quant)

        if perc > 0:
            massive_weight_list = self.extract_weights(instan, perc)
            all_vect_weights = np.concatenate([x.reshape(-1,1) for x in massive_weight_list], axis=None).reshape(-1,1)
            
            dd = 10**6
            ok = False
            for delta in (range(100, dd+500, 500)):#[0.000001, 0.00001, 0.000015, 0.00002, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1, 0.5, 1]:
                uniformed = uniform_quantization(all_vect_weights, delta/dd, self.b)
                unique = np.unique(uniformed)
                #print(delta/dd, len(unique))
                if len(unique) <= perc:
                    ok = True
                    #print("used ",delta/dd, len(unique))
                    centers = unique.reshape(-1,1)
                    perc = len(unique)
                    idx_layers = [uCWS.redefine_weights(w, centers) for w in massive_weight_list]
                    break
            if not ok:
                uniformed = uniform_quantization(all_vect_weights, 1, self.b)
                unique = np.unique(uniformed)
                centers = unique.reshape(-1,1)
                perc = len(unique)
                idx_layers = [uCWS.redefine_weights(w, centers) for w in massive_weight_list]
            
            self.recompose_weight(instan, perc, centers, idx_layers)

        elif perc == -1:
            massive_weight_list = self.extract_weights(instan, perc)
            all_vect_weights = np.concatenate([x.reshape(-1,1) for x in massive_weight_list], axis=None).reshape(-1,1)

            uniformed = uniform_quantization(all_vect_weights, self.delta, self.b)
            centers = np.unique(uniformed).reshape(-1,1)
            perc = len(self.centers)
            idx_layers = [uCWS.redefine_weights(w, centers) for w in massive_weight_list]

            self.recompose_weight(instan, perc, centers, idx_layers)

        return idx_layers, centers, perc




        




 
        
