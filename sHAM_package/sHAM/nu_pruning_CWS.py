import numpy as np
from scipy import ndimage
import tensorflow as tf

from sHAM import nu_pruning
from sHAM import nu_CWS

import gc


def find_index_first_dense(list_weights):
    i = 2
    for w in list_weights[2:]:
        if len(w.shape)==2:
            return i
        i += 1

def centroid_gradient_matrix_combined(idx_matrix,gradient,cluster,mask):
    gradient += 0.0000000001
    gradient[np.logical_not(mask)] = 0
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)

def idx_matrix_to_matrix(idx_matrix,centers):
    return centers[idx_matrix.reshape(-1,1)].reshape(idx_matrix.shape)


class nu_pruning_CWS(nu_pruning.nu_pruning, nu_CWS.nu_CWS):
    def __init__(self, model, perc_prun_for_dense, clusters_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        self.perc_prun_for_dense = perc_prun_for_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2


    def apply_pruning_ws(self, list_trainable=None, untrainable_per_layers=None, mbkmeans=True):
        self.apply_pruning(list_trainable, untrainable_per_layers) #crea masks e setta pesi prunati nel modello
        self.apply_ws(list_trainable, untrainable_per_layers, mbkmeans) #crea centri e matrici degli indici e setta pesi ws nel modello
        gc.collect()

    def recompose_weight(self, list_weights, trainable_vars=False, untrainable_per_layers=None):
        masks_for_centers = [self.masks[i] for i in range(0, len(self.masks), self.div)]
        if not trainable_vars:
            d = self.index_first_dense
            return list_weights[:d]+[(idx_matrix_to_matrix(self.idx_layers[(i-d)//self.div], self.centers[(i-d)//self.div])*masks_for_centers[(i-d)//self.div]) if i%self.div==0 else (list_weights[i]) for i in range(d,len(list_weights))]
        else:
            div = self.div + untrainable_per_layers
            list_weights = self.trainable_to_weights(self.model.get_weights(), list_weights, untrainable_per_layers)
            d = find_index_first_dense(list_weights)
            return list_weights[:d]+[(idx_matrix_to_matrix(self.idx_layers[(i-d)//div], self.centers[(i-d)//div])*masks_for_centers[(i-d)//div]) if i%div==0 else (list_weights[i]) for i in range(d,len(list_weights))]


    def update_centers_and_recompose_comb(self, list_weights_before, lr):
        list_weights = self.model.get_weights()
        div = self.div + self.untrainable_per_layers
        d = find_index_first_dense(list_weights)

        masks_for_centers = [self.masks[i] for i in range(0, len(self.masks), self.div)]
        centers_upd = [(centroid_gradient_matrix_combined(self.idx_layers[(i-d)//div], list_weights[i] - list_weights_before[i], self.clusters[(i-d)//div], masks_for_centers[(i-d)//div])) for i in range(d,len(list_weights), div)]
        self.centers = [self.centers[i] + lr * centers_upd[i] for i in range(len(self.centers))]
        if  len(list_weights) == len(self.model.trainable_weights):
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            trainable=[]
            for w in (self.model.trainable_weights):
                trainable.append(w.numpy())
            self.model.set_weights(self.recompose_weight(trainable, True, self.untrainable_per_layers))

    #@tf.function
    def train_pr_ws(self, epochs, lr, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1):
        with tf.device('gpu:0'):
            self.patience = patience
            d = self.index_first_dense
            self.acc_train = []
            self.acc_test = []
            STOP = False
            for epoch in range(epochs):
                if STOP == True:
                    break
                for (batch, (images, labels)) in enumerate(dataset):
                    list_weights_before = self.model.get_weights()
                    self.train_step_pr(images, labels)
                    self.update_centers_and_recompose_comb(list_weights_before, lr)
                    if step_per_epoch:
                        if batch == step_per_epoch:
                            break
                train_acc_epoch = self.accuracy(X_train, y_train)
                if self.patience >= 0:
                    if len(self.acc_train) != 0:
                        if train_acc_epoch - self.acc_train[-1] <= 0.1:
                            if self.patience == 0:
                                STOP = True
                            else:
                                self.patience -= 1
                        else:
                            self.patience = patience

                test_acc_epoch = self.accuracy(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train accuracy: {}'.format(epoch, train_acc_epoch))
            print ('Epoch {} --> test accuracy: {}'.format(epoch, test_acc_epoch))

    def train_ws_deepdta(self, epochs, lr, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1):
        with tf.device('gpu:0'):
            self.patience = patience
            self.acc_train = []
            self.acc_test = []
            STOP = False
            for epoch in range(epochs):
                if STOP == True:
                    break
                for (batch, (images, labels)) in enumerate(dataset):
                    list_weights_before = self.model.get_weights()
                    self.train_step_pr(images, labels)
                    self.update_centers_and_recompose_comb(list_weights_before, lr)
                    if step_per_epoch:
                        if batch == step_per_epoch:
                            break
                train_acc_epoch = self.model.evaluate(X_train, y_train)
                if self.patience >= 0:
                    if len(self.acc_train) != 0:
                        if train_acc_epoch >= self.acc_train[-1]:
                            if self.patience == 0:
                                STOP = True
                            else:
                                self.patience -= 1
                        else:
                            self.patience = patience

                test_acc_epoch = self.model.evaluate(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train MSE: {}'.format(epoch, train_acc_epoch))
            print ('Epoch {} --> test MSE: {}'.format(epoch, test_acc_epoch))



    def get_weightsharing_weigths(self):
        return self.model.get_weights()
