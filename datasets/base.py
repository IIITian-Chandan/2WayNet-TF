import tensorflow as tf

class BaseDataset(object):
    def __init__(self, params):
        self._params = params
        # handle old code (2WayNet) unsuprted features
        if self._params.pca and self._params.pca.split(" ") != ["0", "0"]:
            raise NotImplementedError("PCA not supported")
        if self._params.whiten and self._params.whiten != "0":
            raise NotImplementedError("PCA 'withen' not supported")

    def params(self):
        return self._params
    def load(self):
        raise NotImplementedError()
    def x_train(self):
        raise NotImplementedError()
    def x_test(self):
        raise NotImplementedError()
    def y_train(self):
        raise NotImplementedError()
    def y_test(self):
        raise NotImplementedError()


class MNISTDataset(BaseDataset):
    def load(self):
        mnist = tf.keras.datasets.mnist
        (z_train, label_train),(z_test, label_test) = mnist.load_data()
        # are the full images - out x and y are the two halves of the images
        img_sz = z_train.shape[1] * z_train.shape[2]
        def flatten(z):
            return tf.reshape(z / 255.0, shape=(z.shape[0], img_sz))
        flat_train = flatten(z_train)
        flat_test = flatten(z_test)
        # upper half of image
        self._x_train = flat_train[:,:img_sz // 2]
        self._x_test = flat_test[:,:img_sz // 2]
        # lower half of image
        self._y_train = flat_train[:,img_sz // 2:]
        self._y_test = flat_test[:,img_sz // 2:]

    def x_train(self):
        return self._x_train
    def x_test(self):
        return self._x_test
    def y_train(self):
        return self._y_train
    def y_test(self):
        return self._y_test
