import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy import ndimage

from compressionNN import compressed_nn

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
    vinc_p = lambda x: nearest_centroid_index(centers,x)
    return np.vectorize(vinc_p)(weights).astype(np.uint8 if len(centers) <= 256 else np.uint16)

def idx_matrix_to_matrix(idx_matrix, centers):
    return centers[idx_matrix.reshape(-1,1)].reshape(idx_matrix.shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return ndimage.sum(gradient,idx_matrix,index=range(cluster)).reshape(cluster,1)

class uCWS(compressed_nn.Compressed_NN):
    def __init__(self, model, clusters_for_conv_layers, clusters_for_dense_layers = None):
        self.model = model
        self.clusters_fc = clusters_for_dense_layers # 0 disables Fully Connected clustering
        self.clusters_cnn = clusters_for_conv_layers # 0 disables CNN clustering
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
    
    def list_layer_idx(self):
        outList = []
        convKern_re = re.compile(".*conv.*kernel.*")
        denseKern_re = re.compile("((fc)|(dense)|(predictions)).*kernel.*")
        lista_handle = [x._handle_name for x in self.model.trainable_weights]
        for i in range(len(lista_handle)):
            if (self.clusters_cnn) and (convKern_re.match(lista_handle[i])):
                outList.append(i)
            elif (self.clusters_fc) and (denseKern_re.match(lista_handle[i])):
                outList.append(i)
        return outList

    def recompose_weight(self, instan, perc, centers, idx_layers):
        index_i = 0
        for layer in self.model.layers:
            if (isinstance(layer,instan) and perc > 0):
                if len(layer.get_weights()) > 1:
                    layer.set_weights([idx_matrix_to_matrix(idx_layers[index_i], centers), layer.get_weights()[1]])
                else:
                    layer.set_weights([idx_matrix_to_matrix(idx_layers[index_i], centers)])
                index_i += 1

    def apply_uCWS(self, mbkmeans=True):
        # Dense layers
        if self.clusters_fc > 0:
            massive_weight_list = self.extract_weights(Dense, self.clusters_fc)
            all_vect_weights = np.concatenate([x.reshape(-1,1) for x in massive_weight_list], axis=None).reshape(-1,1) # ultimo reshape x trasposiz da vett riga a vett colonna
            self.centers_fc = build_clusters(weights=all_vect_weights, cluster=self.clusters_fc, mbkmeans=mbkmeans)
            self.idx_layers_fc = [redefine_weights(w, self.centers_fc) for w in massive_weight_list]
            self.recompose_weight(Dense, self.clusters_fc, self.centers_fc, self.idx_layers_fc)
        # Convolutional layers
        if self.clusters_cnn > 0:
            massive_weight_list = self.extract_weights((Conv1D, Conv2D, Conv3D), self.clusters_cnn)
            all_vect_weights = np.concatenate([x.reshape(-1,1) for x in massive_weight_list], axis=None).reshape(-1,1) # ultimo reshape x trasposiz da vett riga a vett colonna
            self.centers_cnn = build_clusters(weights=all_vect_weights, cluster=self.clusters_cnn, mbkmeans=mbkmeans)
            self.idx_layers_cnn = [redefine_weights(w, self.centers_cnn) for w in massive_weight_list]
            self.recompose_weight((Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.centers_cnn, self.idx_layers_cnn)


    def extract_weights(self, instan, perc):
        to_be_returned = []
        for layer in self.model.layers:
            if (isinstance(layer,instan) and perc > 0):
                to_be_returned.append(layer.get_weights()[0])
        return to_be_returned

    def update_centers_and_recompose(self, list_weights_before, lr, instan, perc, centers, idx_layers):
        list_weights = self.extract_weights(instan, perc)
        centers_upd = [(centroid_gradient_matrix(idx_layers[i], list_weights[i]-list_weights_before[i], perc)) for i in range(len(list_weights))]
        
        for c_u in centers_upd:
            centers = centers + lr * c_u

        self.recompose_weight(instan, perc, centers, idx_layers)

    @tf.function
    def train_step_ws(self, images, labels):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss_value = self.loss_object(labels, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def evaluate_internal(self, x, y):
        res = self.model.evaluate(x, y)
        return res if isinstance(res, float) else res[-1]

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
                    self.train_step_ws(images, labels)
                    if self.clusters_cnn > 0:
                        self.update_centers_and_recompose(list_weights_before_cnn, lr, (Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.centers_cnn, self.idx_layers_cnn)
                    if self.clusters_fc > 0:
                        self.update_centers_and_recompose(list_weights_before_fc, lr, Dense, self.clusters_fc, self.centers_fc, self.idx_layers_fc)
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
