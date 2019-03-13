import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

class TiedDenseLayer(tf.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tied_kernel = kwargs.get('tied_kernel')
        self.tied_bias = kwargs.get('tied_bias')

    def call(self, *args, **kwargs):
        super().call(*args, **kwargs)

    def build(self, input_shape):
        # this build() was copied from the parent implementation
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        if self.tied_kernel:
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
        if self.tied_bias:
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
