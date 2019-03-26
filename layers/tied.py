import math
import numbers
import tensorflow as tf
import datetime
from tensorflow.keras.layers import InputSpec
from tensorflow.python.framework import tensor_shape
##from tensorflow.python.layers import base
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from tensorflow.keras.callbacks import History


# for some reason tensor_shape.dimension_value is not found
def _dimension_value(dimension):
    if dimension is None:
        return dimension
    if isinstance(dimension, numbers.Number):
        return dimension
    return dimension.value


def _dense_layer_input_spec(input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if _dimension_value(input_shape[-1]) is None:
        raise ValueError('The last dimension of the inputs to `Dense` '
                         'should be defined. Found `None`.')
    last_dim = _dimension_value(input_shape[-1])
    return last_dim, InputSpec(min_ndim=2,axes={-1: last_dim})


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
        last_dim, self.input_spec = _dense_layer_input_spec(input_shape)
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
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(tf.global_variables_initializer())
        return sess.run(t)

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
    # How we've tested the TiedDenseLayer -
    # we take 1 constant "example" Rx size 100 and 1 "label" Ry size 50
    # Out model has a single dense layer mapping y'=TDxy(Rx)
    # and another dense layer mapping x'=TDyx(y')
    # XYloss is the loss of y'-Ry
    # YXloss is the loss of x'-Rx
    # the model is trained on the combined loss with equal weights.
    # We try 2 cases - False=TDxy and TDyx are NOT tied
    #                  True =TDxy and TDyx are tied with Kernel2=transpose(Kernel1)
    # In both cases we can see XYloss decreasing.
    # the metrics YXloss is much better when using tied layer
    # you should be able to see this in the graph
    _test_TiedDenseLayer_1(False)
    _test_TiedDenseLayer_1(True)


class TiedDropoutLayer(tf.layers.Dropout):
    def __init__(self, *args, **kwargs):
        # tied_mask_variable is a "keep" mask with "1" where to keep the input.
        self.tied_mask_variable = kwargs.pop('tied_mask_variable', None)
        super().__init__(*args, **kwargs)
        self.mask_variable = None
        self.mask_compute = None

    def _create_random_mask_variable(self, shape, dtype=tf.float32):
        # assuming shape[0] is batch size
        random_tensor = tf.random.uniform(shape[1:], dtype=dtype)
        keep_mask = random_tensor >= self.rate
        self.mask_variable = tf.Variable(tf.ones(shape=shape[1:], dtype=tf.bool),
                        name="tied_mask_variable", use_resource=True)
        self.mask_compute = tf.assign(self.mask_variable, keep_mask)

    def get_mask_varible(self):
        # used to get on mask and used it for another TiedDropoutLayer
        return self.mask_variable

    def build(self, input_shape):
        if self.built:
            return
        input_shape = tensor_shape.TensorShape(input_shape)

        if self.tied_mask_variable is None:
            self._create_random_mask_variable(input_shape)
        else:
            self.mask_compute = self.tied_mask_variable
        return super().build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
          training = K.learning_phase()

        def dropped_inputs():
            keep_prob = 1 - self.rate
            scale = 1 / keep_prob
            return inputs * scale * tf.cast(self.mask_compute, inputs.dtype)

        output = tf_utils.smart_cond(training,
                                     dropped_inputs,
                                     lambda: tf.identity(inputs))
        return output


def _test_TiedDropoutLayer1_old(do_tied, rate):
    shape = (10,10)
    x = tf.ones(shape=shape)
    TD1 = TiedDropoutLayer(rate=rate)
    if do_tied:
        TD1.build(x.shape)
        TD2 = TiedDropoutLayer(rate=rate, tied_mask_variable=TD1.get_mask_varible())
    else:
        TD2 = TiedDropoutLayer(rate=rate)
    y = TD2(TD1(x))
    # because of scaling - only count zeros vs non-zero
    count_y = tf.math.reduce_sum(tf.cast(y > 0, dtype=tf.int8))
    return eval_tensor(count_y)


def _test_TiedDropoutLayer1(do_tied, rate):
    shape = (1, 10,10)
    x = tf.ones(shape=shape)
    y = tf.ones(shape=shape)
    assert (not tf.executing_eagerly())
    TD1 = TiedDropoutLayer(rate=rate)
    if do_tied:
        TD1.build(x.shape)
        TD2 = TiedDropoutLayer(rate=rate, tied_mask_variable=TD1.get_mask_varible())
    else:
        TD2 = TiedDropoutLayer(rate=rate)
    inputs = tf.keras.Input(shape=(10,10))
    outputs = TD2(TD1(inputs))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.001, momentum=0.8)
    model.compile(optimizer, loss='mean_squared_error')
    history = History()
    sess = tf.keras.backend.get_session()
    sess.run(tf.global_variables_initializer())

    model.fit(x, y, epochs=10, steps_per_epoch=1, callbacks=[history])
    return history.history["loss"][-1]

def _test_TiedDropoutLayer():
    # How we've tested TiedDropoutLayer -
    # Checked two dropout layer one on top of the other.
    # given drop_rate=p,  keep_rate = 1-p = q
    # we feed in a vector of 100 1.0
    # if we count the non-zero outputs - we expect to see
    # q**2 probability of the outputs non zero (because we have two layers)
    # if we tie the layers - the probability should be like that of a single layer
    K.set_learning_phase(1)
    rate = 0.25
    keep_rate = 1 - rate
    not_tied_list = [_test_TiedDropoutLayer1(False, rate = rate) for _ in range(10)]
    not_tied_rate = sum(not_tied_list) / len(not_tied_list) / 100.0
    print("not_tied_rate", not_tied_rate)

    tied_list = [_test_TiedDropoutLayer1(True, rate = rate) for _ in range(10)]
    tied_rate = sum(tied_list) / len(tied_list) / 100.0
    print("tied_rate", tied_rate)
    K.set_learning_phase(0)
    assert(abs(tied_rate/keep_rate - 1.0) < 0.15)  # 15% error
    assert(abs(not_tied_rate/(keep_rate**2) - 1.0) < 0.15)  # 15% error


class LocallyDenseLayer(tf.layers.Dense):
    def __init__(self, *args, **kwargs):
        # reduction_ratio - (m in the paper) - by how much to divide the input rank
        self.reduction_ratio = kwargs.pop('reduction_ratio', None)
        if not self.reduction_ratio or self.reduction_ratio < 0:
            raise ValueError("LocallyDenseLayer - reduction_ratio must be given > 0")
        if  math.floor(self.reduction_ratio) != self.reduction_ratio:
            raise ValueError("LocallyDenseLayer - reduction_ratio must be int")
        self.reduction_ratio = int(self.reduction_ratio)
        # interleave - if given and True - interleave the repeated dense layer
        self.interleave = kwargs.pop('interleave', None)
        self.dense_layers = []
        super().__init__(*args, **kwargs)
        if (self.units % self.reduction_ratio) != 0:
            raise ValueError("LocallyDenseLayer - output rank {} should be divisible by 'reduction_ratio' {}".format(
                self.units, self.reduction_ratio))

    def build(self, input_shape):
        if self.built:
            return
        input_shape = tensor_shape.TensorShape(input_shape)
        input_slice_shape = tensor_shape.TensorShape(input_shape).as_list()
        last_dim = input_slice_shape[-1]
        if (last_dim % self.reduction_ratio) != 0:
            raise ValueError("LocallyDenseLayer - input rank {} should be divisible by 'reduction_ratio' {}".format(
                last_dim, self.reduction_ratio ))
        input_slice_shape[-1] = last_dim // self.reduction_ratio
        self.dense_layers = []
        tied_kernel = None
        tied_bias = None
        split_units = self.units // self.reduction_ratio
        for _ in range(self.reduction_ratio):
            # the first of these TiedDenseLayer recieves tied_kernel=None
            # and creates intenal kernel. This kernell is then passed to all the others
            td = TiedDenseLayer(units=split_units, tied_kernel=tied_kernel, tied_bias=tied_bias)
            td.build(input_slice_shape)
            tied_kernel = td.kernel
            tied_bias = td.bias
            self.dense_layers.append(td)


    def call(self, inputs, training=None):
        rank = len(inputs.shape)
        if self.interleave:
            raise NotImplemented("FIXME")
        else:
            splits = tf.split(inputs, self.reduction_ratio, axis=rank - 1)
            output_splits = []
            for i in range(self.reduction_ratio):
                split = splits[i]
                output_splits.append(self.dense_layers[i](split))
            outputs = tf.concat(output_splits, axis=rank - 1)
            return outputs


def _test_LocallyDenseLayer_1(m):
    x = tensor2const(tf.random.uniform([1, 100]))
    y = tensor2const(tf.random.uniform([1, 50]))
    TDxy = LocallyDenseLayer(units=50, reduction_ratio=m)
    assert (not tf.executing_eagerly())
    inputs = tf.keras.Input(shape=(100,))  ##tensor=X)
    outputs = TDxy(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.001, momentum=0.8)
    model.compile(optimizer, loss='mean_squared_error')
    history = History()
    model.fit(x, y, epochs=10, steps_per_epoch=1, callbacks=[history])
    return history.history["loss"][-1]


def _test_LocallyDenseLayer():
    #K.set_learning_phase(1)
    _test_LocallyDenseLayer_1(2)
    try:
        _test_LocallyDenseLayer_1(3)
        raise RuntimeError("ERROR - we should had ValueError = test failed")
    except ValueError as e:
        print("Got Value Errr m==3 - OK!")
    _test_LocallyDenseLayer_1(2)
    _test_LocallyDenseLayer_1(10)


    #not_tied_list = [_test_TiedDropoutLayer1(False, rate = rate) for _ in range(100)]
    #not_tied_rate = sum(not_tied_list) / len(not_tied_list) / 100.0
    #print("not_tied_rate", not_tied_rate)

    #tied_list = [_test_TiedDropoutLayer1(True, rate = rate) for _ in range(100)]
    #tied_rate = sum(tied_list) / len(tied_list) / 100.0
    #print("tied_rate", tied_rate)
    #K.set_learning_phase(0)
    #assert(abs(tied_rate/keep_rate - 1.0) < 0.15)  # 15% error
    #assert(abs(not_tied_rate/(keep_rate**2) - 1.0) < 0.15)  # 15% error
