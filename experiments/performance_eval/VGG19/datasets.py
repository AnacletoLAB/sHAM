import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10

def MNIST(batch_size):
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    dataset = tf.data.Dataset.from_tensor_slices(
                (tf.cast(x_train, tf.float32), tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset, x_train, y_train, x_test, y_test


def CIFAR10(batch_size):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # data preprocessing
    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

    dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train, tf.float32),
   tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset, x_train, y_train, x_test, y_test
