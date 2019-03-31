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
from tensorflow.keras.callbacks import Callback


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

class _TiedLossCallback(Callback):
    def on_train_begin(self, logs={}):
        self.last_XYloss = None
        self.last_YXloss = None

    def on_epoch_end(self, epoch, logs={}):
        self.last_XYloss = logs.get("XYloss")
        self.last_YXloss = logs.get("YXloss")


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
    tied_loss_cb = _TiedLossCallback()
    model.fit(Rx, Ry, epochs=1500, steps_per_epoch=1, callbacks=[tied_loss_cb])
    print(tied_loss_cb.last_XYloss, tied_loss_cb.last_YXloss)
    return tied_loss_cb.last_XYloss, tied_loss_cb.last_YXloss

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
    xy_loss, yx_loss = _test_TiedDenseLayer_1(False)
    xy_loss_tied, yx_loss_tied = _test_TiedDenseLayer_1(True)
    print(f"xy_loss ~~== xy_loss_tied: {xy_loss} ~~== {xy_loss_tied}")
    print(f"yx_loss > yx_loss_tied: {yx_loss} > {yx_loss_tied}")
    assert(yx_loss > 2 * yx_loss_tied)


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


def _test_TiedDropoutLayer1(do_tied, rate):
    shape = (10,10)
    x = tf.ones(shape=shape)
    TD1 = TiedDropoutLayer(rate=rate)
    if do_tied:
        TD1.build(x.shape)
        print(f"DEBUG>>>> TD1.count_params {TD1.count_params}")
        TD2 = TiedDropoutLayer(rate=rate, tied_mask_variable=TD1.get_mask_varible())
    else:
        TD2 = TiedDropoutLayer(rate=rate)
    y = TD2(TD1(x))
    # because of scaling - only count zeros vs non-zero
    count_y = tf.math.reduce_sum(tf.cast(y > 0, dtype=tf.int8))
    return eval_tensor(count_y) / 100.0


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
    not_tied_rate = sum(not_tied_list) / len(not_tied_list)
    print("not_tied_rate", not_tied_rate)

    K.set_learning_phase(1)
    tied_list = [_test_TiedDropoutLayer1(True, rate = rate) for _ in range(10)]
    tied_rate = sum(tied_list) / len(tied_list)
    print("tied_rate", tied_rate)
    K.set_learning_phase(0)
    print(f"tied_rate/keep_rate = {tied_rate}/{keep_rate} = {tied_rate/keep_rate}")
    assert(abs(tied_rate/keep_rate - 1.0) < 0.15)  # 15% error
    print(f"not_tied_rate/keep_rate**2 = {not_tied_rate}/{keep_rate**2} = {not_tied_rate/keep_rate**2}")
    assert(abs(not_tied_rate/(keep_rate**2) - 1.0) < 0.15)  # 15% error


class LocallyDenseLayer(tf.layers.Layer):
    def __init__(self, *args, **kwargs):
        # reduction_ratio - (m in the paper) - by how much to divide the input rank
        self.units = kwargs.pop('units', None)
        self.reduction_ratio = self._validate_reduction_ratio(kwargs)
        self.built = False
        # interleave - if given and True - interleave the repeated dense layer
        self.interleave = kwargs.pop('interleave', None)
        self.not_tied_for_testing = kwargs.pop('not_tied_for_testing', None)
        self.dense_layers = []
        super().__init__(*args, **kwargs)

    def _validate_reduction_ratio(self, kw_dict):
        reduction_ratio = kw_dict.pop('reduction_ratio', None)
        if not reduction_ratio or reduction_ratio < 0:
            raise ValueError("LocallyDenseLayer - reduction_ratio must be given > 0")
        if  math.floor(reduction_ratio) != reduction_ratio:
            raise ValueError("LocallyDenseLayer - reduction_ratio must be int")
        reduction_ratio = int(reduction_ratio)
        if (self.units % reduction_ratio) != 0:
            raise ValueError("LocallyDenseLayer - output rank {} should be divisible by 'reduction_ratio' {}".format(
                self.units, reduction_ratio))
        return reduction_ratio


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
            # need to add inner layers to this layers trainable variables
            # or traning will not happen
            self._trainable_weights.extend(td.trainable_weights)
            tied_kernel = td.kernel
            tied_bias = td.bias
            if self.not_tied_for_testing:
                # allow independent dense layers for testing
                tied_kernel = None
                tied_bias = None
            self.dense_layers.append(td)
        self.built = True

    def call(self, inputs, training=None):
        rank = len(inputs.shape)
        if self.interleave:
            output_splices = []
            for i in range(self.reduction_ratio):
                # slice goes over the last dimention.
                # it starts from i and jumps m to interleave
                slice = inputs[...,i::self.reduction_ratio]
                output_splices.append(self.dense_layers[i](slice))
            outputs = tf.concat(output_splices, axis=rank - 1)
        else:
            splits = tf.split(inputs, self.reduction_ratio, axis=rank - 1)
            output_splits = []
            for i in range(self.reduction_ratio):
                split = splits[i]
                output_splits.append(self.dense_layers[i](split))
            outputs = tf.concat(output_splits, axis=rank - 1)
        return outputs

    def compute_output_shape(self, input_shape):
        #return (input_shape[:-1] + (self.units,) )
        return self.dense_layers[0].compute_output_shape(input_shape)

def _test_LocallyDenseLayer_1(x, y, m, not_tied_for_testing=False, interleave=False):
    ##y = tensor2const(tf.random.uniform([1, 64]))
    inputs = tf.keras.Input(shape=(128,))
    TDxy = LocallyDenseLayer(units=64, reduction_ratio=m,
                             not_tied_for_testing=not_tied_for_testing,
                             interleave=interleave)
    #TDxy = tf.keras.layers.Dense(units=64) ##
    assert(not tf.executing_eagerly())
    outputs = TDxy(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.01, momentum=0.8)
    model.compile(optimizer, loss='mean_squared_error')
    history = History()
    model.fit(x, y, epochs=500, steps_per_epoch=1, callbacks=[history])
    return history.history["loss"][-1]


def _test_LocallyDenseLayer():
    assert(not tf.executing_eagerly())
    K.set_learning_phase(1)
    x1 = tf.ones(shape=[1, 31])  # 31 does not divide by 2/8/32
    c = [1+ (i+1)/10.0 for i in range(4)]
    x_list = [x1 * cc for cc in c]
    x2 = tf.ones(shape=[1, 128-31*4])  # the rest
    x_list.append(x2)
    x = tensor2const(tf.concat(x_list, axis=1))
    y = tf.ones(shape=[1, 64])

    loss2 = _test_LocallyDenseLayer_1(x, y, 2)
    try:
        _test_LocallyDenseLayer_1(x, y, 3)
        raise RuntimeError("ERROR - we should had ValueError = test failed")
    except ValueError as e:
        print("Got Value Errr m==3 - OK!")
    loss8 = _test_LocallyDenseLayer_1(x, y, 8)
    loss32 = _test_LocallyDenseLayer_1(x, y, 32)
    print(f"loss2<loss8<loss32 {loss2}<{loss8}<{loss32}")

    # TODO(franji) we have some anomaly in this test data selection that makes
    #   loss2 > loss32 or sometimes loss2>loss32
    assert(loss2<loss8 or loss2<loss32)
    # try not tied
    loss2_not_tide = _test_LocallyDenseLayer_1(x, y, 2, not_tied_for_testing=True)
    loss8_not_tide = _test_LocallyDenseLayer_1(x, y, 8, not_tied_for_testing=True)
    loss32_not_tide = _test_LocallyDenseLayer_1(x, y,32, not_tied_for_testing=True)
    print(f"loss2<loss8<loss32 {loss2}<{loss8}<{loss32}")
    print(f"loss2_not_tide<loss8_not_tide<loss32_not_tide {loss2_not_tide}<{loss8_not_tide}<{loss32_not_tide}")
    assert(loss2_not_tide<loss2)
    assert(loss8_not_tide<loss8)
    assert(loss32_not_tide<loss32)

    # check interleave vs not:
    x1 = tf.ones(shape=[1, 64])
    x = tensor2const(tf.concat([x1, x1 * 1.5], axis=1))
    y = tf.ones(shape=[1, 64])
    loss2 = _test_LocallyDenseLayer_1(x, y, 2)
    loss2_interleave = _test_LocallyDenseLayer_1(x, y, 2, interleave=True)
    print(f"input [1,1...,1.5,1.5...] loss2>loss2_interleave {loss2}<{loss2_interleave}")
    assert(loss2>loss2_interleave)
    x = tf.constant([ [1,1.5] * 64 ])
    loss2 = _test_LocallyDenseLayer_1(x, y, 2)
    loss2_interleave = _test_LocallyDenseLayer_1(x, y, 2, interleave=True)
    print(f"input [1,1.5,1.1.5,......] loss2<loss2_interleave {loss2}>{loss2_interleave}")
    assert(loss2<loss2_interleave)


def _test_all():
    _test_LocallyDenseLayer()
    _test_TiedDropoutLayer()
    _test_LocallyDenseLayer()
