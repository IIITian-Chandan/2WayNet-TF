import tensorflow as tf

class BaseDataset(object):
    def __init__(self, params):
        self.params = params
        # handle old code (2WayNet) unsuprted features
        if self.params.pca and self.params.pca.split(" ") != ["0", "0"]:
            raise NotImplementedError("PCA not supported")
        if self.params.whiten and self.params.whiten != "0":
            raise NotImplementedError("PCA 'withen' not supported")

    def load(self):
        raise NotImplementedError()


class MNISTDataset(BaseDataset):
    def load(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        print(y_test)