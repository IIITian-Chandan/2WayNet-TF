import numbers
import tensorflow as tf
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


class TiedDropoutLayer(tf.layers.Dropout):
    pass


class LocallyDenseLayer(tf.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LocallyDropoutLayer(tf.layers.Dropout):
    pass
