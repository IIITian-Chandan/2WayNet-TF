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
        with tf.Session() as sess:
            mnist = tf.keras.datasets.mnist
            (z_train, label_train),(z_test, label_test) = mnist.load_data()
            # are the full images - out x and y are the two halves of the images
            img_sz = z_train.shape[1] * z_train.shape[2]
            half_sz = img_sz // 2
            x_train, y_train = tf.split(z_train, 2, axis=2)
            x_test, y_test = tf.split(z_train, 2, axis=2)
            def flatten(z):
                return tf.reshape(z / 255.0, shape=(z.shape[0], half_sz))
            # left half of image
            self._x_train = sess.run(flatten(x_train))
            self._x_test = sess.run(flatten(x_test))
            # right half of image
            self._y_train = sess.run(flatten(y_train))
            self._y_test = sess.run(flatten(y_test))

    def x_train(self):
        return self._x_train
    def x_test(self):
        return self._x_test
    def y_train(self):
        return self._y_train
    def y_test(self):
        return self._y_test
    def get_tb_image_varibles(self, x_input, y_input, x_output, y_output):
        shape1 = (14, 28)
        img_tensors = [tf.reshape(tf.cast(t[0] * 255, tf.uint8), shape=shape1) for t in
                       [x_input, y_input, x_output, y_output]]
        full_img = tf.concat(img_tensors, axis=0)

        img_var = tf.keras.backend.variable(tf.zeros(shape=(shape1[0] * 4, shape1[1]), dtype=tf.uint8),
                                        name="x_y_in_out", dtype=tf.uint8)

        def dummy_metic_for_images(_y_true_unused, _y_pred_unused):
            return tf.reduce_sum(tf.assign(img_var, full_img))
        return dummy_metic_for_images, [img_var]

