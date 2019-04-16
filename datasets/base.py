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
    def get_tb_image_varibles(self, x_input, y_input, x_output, y_output):
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
        self._x_train = flat_train[:1000,:img_sz // 2]
        self._x_test = flat_test[:,:img_sz // 2]
        # lower half of image
        self._y_train = flat_train[:1000,img_sz // 2:]
        self._y_test = flat_test[:,img_sz // 2:]

    def x_train(self):
        return self._x_train
    def x_test(self):
        return self._x_test
    def y_train(self):
        return self._y_train
    def y_test(self):
        return self._y_test
    def get_tb_image_varibles(self, x_input, y_input, x_output, y_output):
        """get the 2 channels input and output
        return a list of variables containing the image tensorss for TensorBoard
        and a function-clousure to do the assignment of the x/y_input/ouput to the variables.
        the returned metric_assignment_func is of type f(y_pred, y_true)
        return: metric_assignment_func, [variables]"""

        def to_img(t, var_name):
            shape = (14, 28)
            img_var = tf.keras.backend.variable(tf.zeros(shape=shape, dtype=tf.uint8),
                                                name=var_name, dtype=tf.uint8)
            # use t[0] - first in the batch
            img = tf.reshape(tf.cast(t[0] * 255, tf.uint8), shape=(14, 28))
            return tf.assign(img_var, img), img_var

        assign_x_input_image, x_input_image = to_img(x_input, "x_input_image")
        assign_x_output_image, x_output_image = to_img(x_output, "x_output_image")
        assign_y_input_image, y_input_image = to_img(y_input, "y_input_image")
        assign_y_output_image, y_output_image = to_img(y_output, "y_output_image")

        def dummy_metic_for_images(_y_true_unused, _y_pred_unused):
            return (tf.reduce_sum(assign_x_input_image) +
                    tf.reduce_sum(assign_x_output_image) +
                    tf.reduce_sum(assign_y_input_image) +
                    tf.reduce_sum(assign_y_output_image))
        return dummy_metic_for_images, [x_input_image, x_output_image, y_input_image, y_output_image]

