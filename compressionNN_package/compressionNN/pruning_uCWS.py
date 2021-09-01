import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense
from scipy import ndimage

from compressionNN import pruning
from compressionNN import uCWS


def centroid_gradient_matrix_combined(idx_matrix,gradient,cluster,mask):
    gradient += 0.0000000001
    gradient[np.logical_not(mask)] = 0
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)


class pruning_uCWS(pruning.pruning, uCWS.uCWS):
    def __init__(self, model, perc_prun_for_dense, perc_prun_for_cnn, clusters_for_dense_layers, clusters_for_conv_layers):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense  # 0 disabilita pruning per livelli densi
        self.perc_prun_for_cnn = perc_prun_for_cnn      # 0 disabilita pruning per livelli convoluzionali
        self.clusters_fc = clusters_for_dense_layers
        self.clusters_cnn = clusters_for_conv_layers
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"


    def apply_pr_uCWS(self, mbkmeans=True):
        self.apply_pruning() #crea masks e setta pesi prunati nel modello
        self.apply_uCWS(mbkmeans) #crea centri e matrici degli indici e setta pesi ws nel modello

    def update_centers_and_recompose(self, list_weights_before, lr, instan, perc, centers, idx_layers, masks):
        list_weights = self.extract_weights(instan, perc)
        centers_upd = [(centroid_gradient_matrix_combined(idx_layers[i], list_weights[i]-list_weights_before[i], perc, masks[i])) for i in range(len(list_weights))]
        
        for c_u in centers_upd:
            centers = centers + lr * c_u

        self.recompose_weight(instan, perc, centers, idx_layers)

    def train_ws(self, epochs, lr, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1, best_model=True, min_is_better=True, threshold=0.0001):
        comp_lmbd = (lambda a,b: a<=b) if min_is_better else (lambda a,b: a>=b)
        with tf.device('gpu:0'):
            self.patience = patience
            self.acc_train = []
            self.acc_test = []
            STOP = False
            for epoch in range(epochs):
                if STOP == True:
                    break
                for (batch, (images, labels)) in enumerate(dataset):
                    list_weights_before_cnn = self.extract_weights((Conv1D, Conv2D, Conv3D), self.clusters_cnn)
                    list_weights_before_fc = self.extract_weights(Dense, self.clusters_fc)
                    self.train_step_pr(images, labels)  # TODO: qui si potrebbe fare una cosa per evitare di copiare tutta la f di training
                    if self.clusters_cnn > 0:
                        self.update_centers_and_recompose(list_weights_before_cnn, lr, (Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.centers_cnn, self.idx_layers_cnn, self.masks_cnn)
                    if self.clusters_fc > 0:
                        self.update_centers_and_recompose(list_weights_before_fc, lr, Dense, self.clusters_fc, self.centers_fc, self.idx_layers_fc, self.masks_fc)

                    if step_per_epoch:
                        if batch == step_per_epoch:
                            break
                train_acc_epoch = self.evaluate_internal(X_train, y_train)
                if self.patience >= 0:
                    if len(self.acc_train) != 0:
                        if comp_lmbd(self.acc_train[-1] - train_acc_epoch, threshold):
                            if self.patience == 0:
                                STOP = True
                            else:
                                self.patience -= 1
                        else:
                            self.patience = patience
                            if best_model:
                                self.model.save_weights(self.timestamped_filename)
                    elif best_model:
                        self.model.save_weights(self.timestamped_filename)
                test_acc_epoch = self.evaluate_internal(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train: {}'.format(epoch, train_acc_epoch))
            if best_model:
                self.model.load_weights(self.timestamped_filename)
                test_acc_epoch = self.evaluate_internal(X_test, y_test)
                self.acc_test.append(test_acc_epoch)

            print ('Epoch {} --> test: {}'.format(epoch, test_acc_epoch))

        if os.path.exists(self.timestamped_filename):
            os.remove(self.timestamped_filename)
