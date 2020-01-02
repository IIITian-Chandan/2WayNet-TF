import tensorflow as tf
import platform
import random

DEBUG_SMALL_DATASET = False


class BaseDataset(object):
    def __init__(self, params):
        self._params = params
        # handle old code (2WayNet) unsuprted features
        if params and self._params.pca and self._params.pca.split(" ") != ["0", "0"]:
            raise NotImplementedError("PCA not supported")
        if params and self._params.whiten and self._params.whiten != "0":
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

    def get_tb_image_varibles(self, x_input, y_input, x_output, y_output):
        raise NotImplementedError()


class MNISTDataset(BaseDataset):
    def load(self):
        with tf.Session() as sess:
            mnist = tf.keras.datasets.mnist
            (z_train, label_train), (z_test, label_test) = mnist.load_data()
            if DEBUG_SMALL_DATASET and platform.system() == "Darwin":
                random.shuffle(z_train)
                z_train = z_train[:1000]
            # are the full images - out x and y are the two halves of the images
            img_sz = z_train.shape[1] * z_train.shape[2]
            half_sz = img_sz // 2
            x_train, y_train = tf.split(z_train, 2, axis=2)
            x_test, y_test = tf.split(z_test, 2, axis=2)

            def flatten(z):
                return tf.reshape(tf.cast(z, dtype=tf.float32) / 255.0, shape=(z.shape[0], half_sz))

            # left half of image
            self._x_train = sess.run(flatten(x_train))
            self._x_test = sess.run(flatten(x_test))
            random.shuffle(self._x_test)  # for some reason model.evaluate does not have suffle=True parameter
            # right half of image
            self._y_train = sess.run(flatten(y_train))
            self._y_test = sess.run(flatten(y_test))
            random.shuffle(self._y_test)  # for some reason model.evaluate does not have suffle=True parameter

    def x_train(self):
        return self._x_train

    def x_test(self):
        return self._x_test

    def y_train(self):
        return self._y_train

    def y_test(self):
        return self._y_test

    def get_tb_image_varibles(self, x_input, y_input, x_output, y_output):
        """Image for tensor board monitoring"""
        shape1 = (28, 14)
        # Create an image of 4 halves: x_input=original left half, y_output=predicted right half, x_output=predicted
        # left-half, y_input=original left half
        img_tensors = [tf.reshape(tf.cast(t[0] * 255, tf.uint8), shape=shape1) for t in
                       [x_input, y_output, x_output, y_input]]
        full_img = tf.concat(img_tensors, axis=1)

        img_var = tf.keras.backend.variable(tf.zeros(shape=(shape1[0], shape1[1] * 4), dtype=tf.uint8),
                                            name="x_y_in_out", dtype=tf.uint8)

        def dummy_metic_for_images(_y_true_unused, _y_pred_unused):
            return tf.reduce_sum(tf.assign(img_var, full_img))

        return dummy_metic_for_images, [img_var]
