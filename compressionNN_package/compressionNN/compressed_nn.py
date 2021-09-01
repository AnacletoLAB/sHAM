import abc
import tensorflow as tf

class Compressed_NN(metaclass=abc.ABCMeta):

    def predict(self, x_test):
        return tf.math.argmax(self.model.predict(x_test), axis=1)

    def accuracy(self, x_test, true):
        prediction = self.predict(x_test)
        equality = tf.equal(prediction, tf.argmax(true, axis=1))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return accuracy.numpy() * 100


    def set_optimizer(self, opt):
        self.optimizer = opt

    def set_loss(self, loss):
        self.loss_object = loss

    def print_shape_layers(list_weights):
        for i in range(len(list_weights)):
            print(list_weights[i].shape)

    def trainable_to_weights(self, list_weights, list_trainable, untrainable_per_layers):
        tr = list_trainable
        w = list_weights
        aw = []
        d = self.div
        for i in range(d,len(w),d+untrainable_per_layers):
            for j in range(untrainable_per_layers):
                aw.append(w[i+j])
        l=[]
        i=1
        for tre in tr:
            l.append(tre)
            if i % d == 0:
                for _ in range(untrainable_per_layers):
                    l.append(aw.pop(0))
            i+=1

        return l    
