#from keras import backend as k
import re
import numpy as np
from sHAM import compressed_nn

import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras.layers.core import Dense

### Tipi di layer:
# tf.python.keras.layers.convolutional.Conv2D
# tf.python.keras.layers.convolutional.Conv1D
# tf.python.keras.layers.convolutional.Conv3D
# tf.python.keras.layers.core.Dense

# tf.python.keras.layers.core.Activation
# tensorflow.python.keras.layers.pooling.MaxPooling2D
# tensorflow.python.keras.layers.normalization_v2.BatchNormalization
# tensorflow.python.keras.layers.core.Flatten
# tensorflow.python.keras.layers.core.Dropout

def pruning_f(W, pruning):
    W_pruned = np.copy(W)
    mask = np.abs(W) > np.percentile(np.abs(W), pruning)
    W_pruned *= mask
    return W_pruned, mask

class pruning(compressed_nn.Compressed_NN):
    def __init__(self, model, perc_prun_for_dense, perc_prun_for_cnn, level_idx = None):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense  # 0 disables pruning on FC layers
        self.perc_prun_for_cnn = perc_prun_for_cnn      # 0 disables pruning on CNN layers
        self.level_idxs = self.list_layer_idx()

    # scans model.trainable_weights and use a regex to select FC and/or CNN layers
    def list_layer_idx(self):
        outList = []
        convKern_re = re.compile(".*conv.*kernel.*")
        denseKern_re = re.compile("((fc)|(dense)|(predictions)).*kernel.*")
        lista_handle = [x._handle_name for x in self.model.trainable_weights]
        for i in range(len(lista_handle)):
            if (self.perc_prun_for_cnn is not 0) and (convKern_re.match(lista_handle[i]) is not None):
                outList.append(i)
            elif (self.perc_prun_for_dense is not 0) and (denseKern_re.match(lista_handle[i]) is not None):
                outList.append(i)
        return outList

    def apply_pruning(self, list_trainable=None):
        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        self.masks, self.masks_cnn, self.masks_fc = [], [], []
        for layer in self.model.layers:
            if isinstance(layer,(Conv1D, Conv2D, Conv3D)):
                ww = layer.get_weights()[0]
                W_pruned, mask = pruning_f(ww, self.perc_prun_for_cnn)
                self.masks_cnn.append(mask)
                if self.perc_prun_for_cnn is not 0: # Devo fare così per risolvere bug in update_centers_and_recompose di pruning_weightsharing
                    if len(layer.get_weights()) > 1:
                        layer.set_weights([W_pruned, layer.get_weights()[1]])
                    else:
                        layer.set_weights([W_pruned])
                    self.masks.append(mask)
                
            elif isinstance(layer,Dense):
                ww = layer.get_weights()[0]
                W_pruned, mask = pruning_f(ww, self.perc_prun_for_dense)
                self.masks_fc.append(mask)
                if self.perc_prun_for_dense is not 0:
                    if len(layer.get_weights()) > 1:
                        layer.set_weights([W_pruned, layer.get_weights()[1]])
                    else:
                        layer.set_weights([W_pruned])
                    self.masks.append(mask)
                
    @tf.function
    def train_step_pr(self, images, labels):
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss_value = self.loss_object(labels, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights) ### Grads è una lista di tensori (no specifiche o altro)
        self.temp_masks = self.masks.copy()
        grads_pruned = [tf.multiply(grads[i],tf.cast(self.temp_masks.pop(0), tf.float32)) if i in self.level_idxs else grads[i] for i in range(0,len(grads))]

        self.optimizer.apply_gradients(zip(grads_pruned, self.model.trainable_weights))

    def train_pr(self, epochs, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1):
        with tf.device('gpu:0'):
            self.patience = patience
            self.acc_train = []
            self.acc_test = []
            self.temp_masks = []
            STOP = False
            for epoch in range(epochs):
                if STOP == True:
                    break
                for (batch, (images, labels)) in enumerate(dataset):
                    self.train_step_pr(images, labels)
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

    def train_pr_deepdta(self, epochs, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1):
        with tf.device('gpu:0'):
            self.patience = patience
            self.acc_train = []
            self.acc_test = []
            STOP = False
            for epoch in range(epochs):
                if STOP == True:
                    break
                for (batch, (images, labels)) in enumerate(dataset):
                    self.train_step_pr(images, labels)
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

                test_acc_epoch = self.model.evaluate(X_test, y_test)
                self.acc_train.append(train_acc_epoch)
                self.acc_test.append(test_acc_epoch)
                print ('Epoch {} --> train MSE: {}'.format(epoch, train_acc_epoch))
            print ('Epoch {} --> test MSE: {}'.format(epoch, test_acc_epoch))


    def get_pruned_weights(self):
        return self.model.get_weights()
