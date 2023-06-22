import numpy as np
import tensorflow as tf

from sHAM import compressed_nn


def pruning(W, pruning):
    W_pruned = np.copy(W)
    mask = np.abs(W) > np.percentile(np.abs(W), pruning)
    W_pruned *= mask
    return W_pruned, mask

class nu_pruning(compressed_nn.Compressed_NN):
    def __init__(self, model, perc_prun_for_dense, index_first_dense, apply_compression_bias=False, div=None):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense
        self.index_first_dense = index_first_dense
        if div:
            self.div=div
        else:
            self.div = 1 if apply_compression_bias else 2

    def apply_pruning(self, list_trainable=None, untrainable_per_layers=None):
        p = self.perc_prun_for_dense
        if not list_trainable:
            list_weights = self.model.get_weights()
        else:
            list_weights=[]
            for w in (list_trainable):
                list_weights.append(w.numpy())

        d = self.index_first_dense
        if not type(p)==list:
            self.list_weights_pruned = list_weights[:d]+[pruning(list_weights[i], p)[0] if i % self.div == 0 else list_weights[i] for i in range(d,len(list_weights))]
            self.masks = [pruning(list_weights[i], p)[1] if i % self.div == 0 else (np.zeros_like(list_weights[i])+True).astype("bool") for i in range(d,len(list_weights))]
        else:
            p_prov_1 = p.copy()
            p_prov_2 = p.copy()
            self.list_weights_pruned = list_weights[:d]+[pruning(list_weights[i], p_prov_1.pop(0))[0] if i % self.div == 0 else list_weights[i] for i in range(d,len(list_weights))]
            self.masks = [pruning(list_weights[i], p_prov_2.pop(0))[1] if i % self.div == 0 else (np.zeros_like(list_weights[i])+True).astype("bool") for i in range(d,len(list_weights))]

        if not list_trainable:
            self.model.set_weights(self.list_weights_pruned)
        else:
            self.model.set_weights(self.trainable_to_weights(self.model.get_weights(), self.list_weights_pruned, untrainable_per_layers))

    @tf.function
    def train_step_pr(self, images, labels):
        d = self.index_first_dense
        with tf.GradientTape() as tape:
            logits = self.model(images, training=True)
            loss_value = self.loss_object(labels, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        grads_pruned = [tf.multiply(grads[i],tf.cast(self.masks[i-d], tf.float32)) if i % self.div == 0 else grads[i] for i in range(d,len(grads))]
        grads_comb = grads[:d]+grads_pruned

        self.optimizer.apply_gradients(zip(grads_comb, self.model.trainable_weights))

    #@tf.function
    def train_pr(self, epochs, dataset, X_train, y_train, X_test, y_test, step_per_epoch=None, patience=-1):
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
