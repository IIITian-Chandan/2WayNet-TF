import numbers
import tensorflow as tf
import datetime
from tensorflow.keras.layers import InputSpec
from tensorflow.python.framework import tensor_shape
##from tensorflow.python.layers import base
from tensorflow.python.keras.layers.core import Dense

# for some reason tensor_shape.dimension_value is no found
def dimension_value(dimension):
    if dimension is None:
        return dimension
    if isinstance(dimension, numbers.Number):
        return dimension
    return dimension.value


class TiedDenseLayer(Dense):
    def __init__(self, *args, **kwargs):
        self.tied_kernel = kwargs.pop('tied_kernel', None)
        self.tied_bias = kwargs.pop('tied_bias', None)
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        ###return super().build(input_shape)
        # this build() was copied from the parent implementation
        # https://github.com/tensorflow/tensorflow/blob/3be3aea56e19e2bcb440ccd736ee86b4e3d6c197/tensorflow/python/keras/layers/core.py
        input_shape = tensor_shape.TensorShape(input_shape)
        if dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        if self.tied_kernel is not None:
            self.kernel = self.tied_kernel
        else:
            self.kernel = self.add_weight(
                'kernel',
                shape=[last_dim, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)
        self.bias = None
        if self.tied_bias is not None:
            self.bias = self.tied_bias
        elif self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        self.built = True

def eval_tensor(t):
    return tf.Session().run(t)

def tensor2const(t):
    return tf.constant(eval_tensor(t))



def _test_TiedDenseLayer_1(do_tied):
    Rx = tensor2const(tf.random.uniform([1, 100]))
    Ry = tensor2const(tf.random.uniform([1, 50]))
    TDxy = TiedDenseLayer(units=50)
    if do_tied:
        TDxy.build(input_shape=(100,))  # must build TDxy if we want to tie TDyx to it - we need the kernel
        TDyx = TiedDenseLayer(units=100, tied_kernel=tf.transpose(TDxy.kernel))
    else:
        TDyx = TiedDenseLayer(units=100)

    assert (not tf.executing_eagerly())
    inputs = tf.keras.Input(shape=(100,))  ##tensor=X)
    outputs = TDxy(inputs)
    Xreconstruct = TDyx(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def combined_loss_func(y0, y1):
        return (tf.losses.mean_squared_error(y0, y1) * 0.5 +
                0.5 * tf.losses.mean_squared_error(inputs, Xreconstruct))
    def XYloss(y_true, y_pred):
        return tf.losses.mean_squared_error(y_true, y_pred)
    def YXloss(y_true, y_pred):
        return tf.losses.mean_squared_error(inputs, Xreconstruct)

    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.001, momentum=0.8)
    model.compile(optimizer, loss=combined_loss_func, metrics=[XYloss, YXloss])
    log_dir = "/tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(Rx, Ry, epochs=1500, steps_per_epoch=1, callbacks=[tensorboard_callback])

def _test_TiedDenseLayer():
    # the metrics YXloss should be much better when using tied layer
    # you should be able to see this in the graph
    _test_TiedDenseLayer_1(False)
    _test_TiedDenseLayer_1(True)


class TiedDropoutLayer(tf.layers.Dropout):
    pass


class LocallyDenseLayer(tf.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LocallyDropoutLayer(tf.layers.Dropout):
    pass
