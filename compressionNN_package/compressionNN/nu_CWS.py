import time

import numpy as np
import tensorflow as tf
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

from compressionNN import compressed_nn

import gc

def find_index_first_dense(list_weights):
    i = 2
    for w in list_weights[2:]:
        if len(w.shape)==2:
            return i
        i += 1

def build_clusters(cluster,weights, mbkmeans=True):
    if mbkmeans:
        kmeans = MiniBatchKMeans(n_clusters=cluster, batch_size=100000, init_size=3*cluster, random_state=42)
    else:
        kmeans = KMeans(n_clusters=cluster, random_state=42, max_iter=100, n_jobs=1)
    kmeans.fit(np.hstack(weights).reshape(-1,1))
    return kmeans.cluster_centers_.astype('float32')

def nearest_centroid_index(centers,value):
    centers = np.asarray(centers)
    idx = (np.abs(centers - value)).argmin()
    return idx

def redefine_weights(weights,centers):
    if len(centers) <= 256:
        arr_ret = np.empty_like(weights).astype(np.uint8)
    else:
        arr_ret = np.empty_like(weights).astype(np.uint16)

    if len(weights.shape) == 2:
        for i, row in enumerate(weights):
            for j, _ in enumerate(row):
                arr_ret[i,j] = nearest_centroid_index(centers,weights[i,j])
    else:
        for i in range(weights.shape[0]):
            arr_ret[i] = nearest_centroid_index(centers,weights[i])
    return arr_ret

def idx_matrix_to_matrix(idx_matrix, centers):
    return centers[idx_matrix.reshape(-1,1)].reshape(idx_matrix.shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)


class nu_CWS(compressed_nn.Compressed_NN):
    def __init__(self, model, clusters_for_dense_layers, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.clusters = clusters_for_dense_layers
        self.index_first_dense = index_first_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_ws(self, list_trainable=None, untrainable_per_layers=None, mbkmeans=True):
        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense
        self.centers = [build_clusters(weights=list_weights[i], cluster=self.clusters[(i-d)//self.div], mbkmeans=mbkmeans) for i in range (d, len(list_weights), self.div)]
        self.idx_layers = [redefine_weights(list_weights[i], self.centers[(i-d)//self.div]) for i in range (d, len(list_weights), self.div)]

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


    @tf.function
    def train_step_ws(self, images, labels):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss_value = self.loss_object(labels, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


    def update_centers_and_recompose(self, list_weights_before, lr):
        list_weights = self.model.get_weights()
        div = self.div + self.untrainable_per_layers
        d = find_index_first_dense(list_weights)
        centers_upd = [(centroid_gradient_matrix(self.idx_layers[(i-d)//div], list_weights[i]-list_weights_before[i], self.clusters[(i-d)//div])) for i in range(d,len(list_weights), div)]
        self.centers = [self.centers[i] + lr * centers_upd[i] for i in range(len(self.centers))]
        if  len(list_weights) == len(self.model.trainable_weights):
            self.model.set_weights(self.recompose_weight(list_weights))
        else:
            trainable=[]
            for w in (self.model.trainable_weights):
                trainable.append(w.numpy())
            self.model.set_weights(self.recompose_weight(trainable, True, self.untrainable_per_layers))


    def train_ws(self, epochs, lr, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1, best_model=True):
        timestamped_filename = str(time.time()) + "_check.h5"
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
                    self.train_step_ws(images, labels)
                    self.update_centers_and_recompose(list_weights_before, lr)
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
                            if best_model:
                                self.model.save_weights(timestamped_filename)
                    elif best_model:
                        self.model.save_weights(timestamped_filename)


                test_acc_epoch = self.accuracy(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train accuracy: {}'.format(epoch, train_acc_epoch))
            if best_model:
                self.model.load_weights(timestamped_filename)
                test_acc_epoch = self.accuracy(X_test, y_test)
                self.acc_test.append(test_acc_epoch)

            print ('Epoch {} --> test accuracy: {}'.format(epoch, test_acc_epoch))

    def train_ws_deepdta(self, epochs, lr, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1, best_model=True):
        timestamped_filename = str(time.time()) + "_check.h5"
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
                    self.train_step_ws(images, labels)
                    self.update_centers_and_recompose(list_weights_before, lr)
                    if step_per_epoch:
                        if batch == step_per_epoch:
                            break
                train_acc_epoch = self.model.evaluate(X_train, y_train)
                if self.patience >= 0:
                    if len(self.acc_train) != 0:
                        if self.acc_train[-1] - train_acc_epoch  <= 0.0001:
                            if self.patience == 0:
                                STOP = True
                            else:
                                self.patience -= 1
                        else:
                            self.patience = patience
                            if best_model:
                                self.model.save_weights(timestamped_filename)
                    elif best_model:
                        self.model.save_weights(timestamped_filename)
                test_acc_epoch = self.model.evaluate(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train MSE: {}'.format(epoch, train_acc_epoch))
            if best_model:
                self.model.load_weights(timestamped_filename)
                test_acc_epoch = self.model.evaluate(X_test, y_test)
                self.acc_test.append(test_acc_epoch)

            print ('Epoch {} --> test MSE: {}'.format(epoch, test_acc_epoch))




    def get_weightsharing_weigths(self):
        return self.model.get_weights()
