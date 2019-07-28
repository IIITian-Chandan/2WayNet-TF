# ##
# # for Python 2 on mac use: $ export LC_ALL=en_US.UTF-8;export LANG=en_US.UTF-8
# from math import ceil
#
# import matplotlib
# import traceback
#
# from testing import test_model
#
# to run with dirsync, ngrok on Google colab:
#$ python ~/miscutil/dirsync.py --source  --dir . --rex_include "." --server http://9d963e8b.ngrok.io

import configparser
import math
import os
import sys
import util

from tensorflow.python.keras import losses
from tensorflow.python.keras import regularizers

import layers.tied
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
import tensorboardimage
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

def create_dataset(name, config):
    import params
    cls_params = params.get_params_class_for_dataset_name(name)
    params = cls_params(config)
    data_loader = params.DATA_CLASS(params)
    return data_loader

# TODO(franji): remove if not used for gamma regulazation in BatchNormalization
def inverse_l2_reg_func(coeff):
    def reg(weight_matrix):
        return K.sum(coeff * K.square(1.0/weight_matrix))
    return reg

BOOKMARK_REPRESENTATION_LAYER = "representation_layer"


class LearningRateControl(object):
    def __init__(self, min_lr,max_lr, step_max_lr, step_min_lr, tensorboardimage=None):
        self.tensorboardimage = tensorboardimage
        self.added_scalar_to_tensorboard = False
        assert(step_min_lr > step_max_lr)  # starting accelerate, decay towards the end
        assert(max_lr > min_lr)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_max_lr = step_max_lr
        total_accelerate_log = math.log(max_lr) - math.log(min_lr)
        # how much to accelerate from step 0 to max
        self.step_accelerate_log = total_accelerate_log / self.step_max_lr
        self.step_deccelerate_log = -total_accelerate_log / (step_min_lr - self.step_max_lr + 1)
        self.global_step = None

    def __call__(self, *args, **kwargs):
        gs = tf.train.get_global_step()
        if gs is None:
            # if not set - create a variable
            self.global_step = K.variable(tf.zeros(shape=(), dtype=tf.int64), dtype=tf.int64, name="lr_global_step")
            tf.train.global_step(K.get_session(), self.global_step)
            gs = K.update_add(self.global_step, 1) ###tf.train.get_global_step()
        else:
            self.global_step = gs
        assert(gs is not None)
        gstep = tf.cast(gs, dtype=tf.float32)
        lr_up = K.exp(self.step_accelerate_log * gstep) * self.min_lr
        lr_down = K.exp(self.step_deccelerate_log * (gstep - self.step_max_lr)) * self.max_lr
        lr = K.switch(K.less(gs, self.step_max_lr), lr_up, lr_down)
        if self.tensorboardimage and not self.added_scalar_to_tensorboard:
            self.tensorboardimage.add_scalar("learning_rate", lr)
            self.added_scalar_to_tensorboard = True  # add once
        return lr

def scale1(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    return (x - x_min) / (x_max - x_min)

def get_cov_image_varibles(cov_x, cov_y):
    """Image for tensor board monitoring - covariance matrices"""
    # Create an image of 2 halves: cov_x, cov_y
    img_tensors = [tf.cast(scale1(t) * 255.0 , tf.uint8) for t in [cov_x, cov_y]]
    full_img = tf.concat(img_tensors, axis=1)

    img_var = tf.keras.backend.variable(tf.zeros_like(full_img), name="cov_x_y", dtype=tf.uint8)

    def dummy_metic_for_images(_y_true_unused, _y_pred_unused):
        return tf.reduce_sum(tf.assign(img_var, full_img))
    return dummy_metic_for_images, [img_var]


def build_model(data_set, tensorboard_callback):
    x_train = data_set.x_train()
    y_train = data_set.y_train()
    assert(len(x_train.shape) == 2)
    assert(len(y_train.shape) == 2)
    x_input_size = util.shape_i(x_train, 1)
    y_input_size = util.shape_i(y_train, 1)
    x_input = tf.keras.Input(shape=(x_input_size,))
    y_input = tf.keras.Input(shape=(y_input_size,))
    prev_layer_size = x_input_size
    assert(len(y_train.shape) == 2)
    y_ouput_size = util.shape_i(y_train, 1)
    layers_x_to_y = []
    layers_y_to_x = []
    is_last_layer = False
    representation_layer_size = None
    for spec in data_set.params().LAYERS_SPEC:
        assert(not is_last_layer)
        is_representation_layer = False
        if len(spec) > 2:
            is_representation_layer = spec[2]
        layer_type = spec[0]
        layer_size = spec[1]
        if layer_size == -1:
            layer_size = y_ouput_size
            is_last_layer = True # size -1 is only for last layer
        is_tied = layer_type in [layers.tied.TiedDenseLayer, layers.tied.LocallyDenseLayer]
        layer_kwargs = dict(units=layer_size,
                         kernel_regularizer=regularizers.l2(data_set.params().WEIGHT_DECAY))
        LXY = layer_type(**layer_kwargs)
        activation_xy = activation_yx = None
        if not is_last_layer:
            activation_xy = LeakyReLU(alpha=data_set.params().LEAKINESS)
            activation_yx = LeakyReLU(alpha=data_set.params().LEAKINESS)
        batch_norm_xy = batch_norm_yx = None
        if not is_last_layer and data_set.params().BN:
            gamma_coef = data_set.params().GAMMA_COEF
            batch_norm_xy = BatchNormalization(gamma_regularizer=inverse_l2_reg_func(gamma_coef))
            batch_norm_yx = BatchNormalization(gamma_regularizer=inverse_l2_reg_func(gamma_coef))
        # We need to build LXY so we can tie internal kernel to LYX
        xy_input_shape = (None, prev_layer_size)
        print(f"Layer X->Y build {type(LXY)}(kwargs={layer_kwargs})")
        LXY.build(input_shape=xy_input_shape)
        # We use prev_layer_size as number of units to the reverse layer
        layer_kwargs = dict(units=prev_layer_size,
                            kernel_regularizer=regularizers.l2(data_set.params().WEIGHT_DECAY))
        if is_tied:
            layer_kwargs["tied_layer"]=LXY
        LYX = layer_type(**layer_kwargs)
        print(f"Layer Y->X build {type(LYX)}(kwargs={layer_kwargs})")
        noise_XY = noise_YX = None
        if not is_last_layer and data_set.params().NOISE_LAYER:
            noise_kwargs = dict(rate=data_set.params().DROP_PROBABILITY)
            noise_XY = data_set.params().NOISE_LAYER(**noise_kwargs)
            print(f"Noise X->Y build {type(noise_XY)}(input_shape={(None,layer_size)})")
            noise_XY.build(input_shape=(None,layer_size))
            if data_set.params().NOISE_LAYER == layers.tied.TiedDropoutLayer:
                noise_kwargs["tied_layer"] = noise_XY
            noise_YX = data_set.params().NOISE_LAYER(**noise_kwargs)
        # Build channel x-->y
        if is_representation_layer:
            # add a "bookmark" to the layers list
            layers_x_to_y.append(BOOKMARK_REPRESENTATION_LAYER)
            representation_layer_size = layer_size
        layers_x_to_y.append(LXY)
        if data_set.params().BN_ACTIVATION:
            layers_x_to_y.append(batch_norm_xy)
            layers_x_to_y.append(activation_xy)
        else:
            layers_x_to_y.append(activation_xy)
            layers_x_to_y.append(batch_norm_xy)
        layers_x_to_y.append(noise_XY)
        # Build channel x-->y in reverse
        layers_y_to_x.append(LYX)
        layers_y_to_x.append(noise_YX)
        if data_set.params().BN_ACTIVATION:
            # oposite from above if because reversed
            layers_y_to_x.append(activation_yx)
            layers_y_to_x.append(batch_norm_yx)
        else:
            layers_y_to_x.append(batch_norm_yx)
            layers_y_to_x.append(activation_yx)
        if is_representation_layer:
            # add a "bookmark" to the layers list (in reverse)
            layers_y_to_x.append(BOOKMARK_REPRESENTATION_LAYER)

        prev_layer_size = layer_size

    channel_x_to_y = x_input
    is_representation_layer = False
    representation_layer_xy = None
    # loop layers_x_to_y to build the channel
    for lay in layers_x_to_y:
        if lay is None:
            continue
        if lay == BOOKMARK_REPRESENTATION_LAYER:
            is_representation_layer = True  # mark for next
            continue
        # Using Keras functional API to stack the layers.
        channel_x_to_y = lay(channel_x_to_y)
        if is_representation_layer:
            # in this channel the bookmark is BEFORE the layer
            assert(representation_layer_xy is None)
            representation_layer_xy = channel_x_to_y
            is_representation_layer = False

    channel_y_to_x = y_input
    representation_layer_yx = None
    # loop reversed(layers_y_to_x) to build the other channel
    for lay in reversed(layers_y_to_x):
        if lay is None:
            continue
        if lay == BOOKMARK_REPRESENTATION_LAYER:
            # in this channel the bookmark is AFTER the layer
            assert (representation_layer_yx is None)
            representation_layer_yx = channel_y_to_x
            continue
        # Using Keras functional API to stack the layers.
        channel_y_to_x = lay(channel_y_to_x)
    # Combined Loss
    loss_x = data_set.params().LOSS_X * losses.mean_squared_error(x_input, channel_y_to_x)
    loss_y = data_set.params().LOSS_Y * losses.mean_squared_error(y_input, channel_x_to_y)
    loss_representation = 0.0
    #assert(representation_layer_xy is not None and representation_layer_yx is not None)
    if data_set.params().L2_LOSS != 0.0 and representation_layer_xy is not None:
        # loss_representation is named 'loss_l2' in original code.
        loss_representation = data_set.params().L2_LOSS * losses.mean_squared_error(representation_layer_xy, representation_layer_yx)

    loss_withen_x = 0.0
    loss_withen_y = 0.0
    cov_x = None
    cov_y = None
    if representation_layer_xy is not None:
        # mean_squared_error takes into account the batch size.
        # when calculating the covariance matrix - we need to do this also
        cov_x = K.dot(tf.transpose(representation_layer_xy),
                      representation_layer_xy) / data_set.params().BATCH_SIZE
        # TODO(Franji): using BACH_SIZE here means in test mode loss_withen_x is wrong
        loss_withen_x = data_set.params().WITHEN_REG_X * (
                    K.sqrt(K.sum(K.sum(cov_x ** 2))) - K.sqrt(K.sum(tf.diag(cov_x) ** 2)))
    if representation_layer_yx is not None:
        cov_y = K.dot(tf.transpose(representation_layer_yx),
                      representation_layer_yx) / data_set.params().BATCH_SIZE
        loss_withen_y = data_set.params().WITHEN_REG_Y * (
                    K.sqrt(K.sum(K.sum(cov_y ** 2))) - K.sqrt(K.sum(tf.diag(cov_y) ** 2)))

    def combined_loss(_y_true_unused, _y_pred_unused):
        return loss_x + loss_y + loss_representation + loss_withen_x + loss_withen_y

    # add images to see what's going on:
    dummy_metic_for_images, image_variables = data_set.get_tb_image_varibles(
        x_input, y_input, channel_y_to_x, channel_x_to_y)

    tensorboard_callback.add_image_variables(image_variables)

    if representation_layer_yx is not None:
        dummy_metric_for_cov, cov_image_variables = get_cov_image_varibles(cov_x, cov_y)
        tensorboard_callback.add_image_variables(cov_image_variables)
    # We have a model
    model = tf.keras.Model(inputs=[x_input, y_input],
                           outputs=[channel_x_to_y, channel_y_to_x])

    base_lr = data_set.params().BASE_LEARNING_RATE
    batches = util.shape_i(x_train, 0) // data_set.params().BATCH_SIZE
    steps = data_set.params().EPOCH_NUMBER * batches
    learning_rate_control = LearningRateControl(
        min_lr=base_lr,
        max_lr=base_lr * 50,
        step_max_lr=int(steps) // 2, step_min_lr=int(steps),
        tensorboardimage=tensorboard_callback)
    optimizer = tf.train.MomentumOptimizer(
        use_nesterov=True,
        learning_rate=learning_rate_control, ###data_set.params().BASE_LEARNING_RATE,
        momentum=data_set.params().MOMENTUM)
    #if tensorboard_callback:
    #    tensorboard_callback.add_scalar("combined_loss", combined_loss(0,0))
    def metric_learning_rate(_y_true_unused, _y_pred_unused):
        return learning_rate_control()
    def calculate_cca():
        return util.cross_correlation_analysis(representation_layer_xy, representation_layer_yx, representation_layer_size)

    def metric_cca(_y_true_unused, _y_pred_unused):
        #return K.switch(K.learning_phase(), tf.constant(0.0), calculate_cca)
        return calculate_cca()

    def metric_var_x(_y_true_unused, _y_pred_unused):
        return K.mean(K.var(representation_layer_xy))

    def metric_var_y(_y_true_unused, _y_pred_unused):
        return K.mean(K.var(representation_layer_yx))


    model.compile(optimizer, loss=combined_loss,
                  metrics=[
                            #dummy_metic_for_images,
                            #metric_learning_rate,
                            #dummy_metric_for_cov,
                            metric_cca,
                            metric_var_x,
                            metric_var_y,
                           ])
    return model


def load_data_set(data_set_config):
    data_config = configparser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = data_config["dataset_parameters"]
    for key in data_parameters:
        print(f"data_parameters[{key}] = {data_parameters[key]}")
    # construct data set
    data_set = create_dataset(data_parameters['name'], data_parameters)
    data_set.load()
    return data_set

def train_model(data_set, tensorboard_callback):
    model = build_model(data_set, tensorboard_callback)
    model.fit([data_set.x_train(), data_set.y_train()],
              [data_set.y_train(), data_set.x_train()],
              epochs=data_set.params().EPOCH_NUMBER,
              #steps_per_epoch=data_set.x_train().shape.dims[0].value // data_set.params().BATCH_SIZE,
              batch_size=data_set.params().BATCH_SIZE,
              callbacks=[tensorboard_callback],
              shuffle=True,
              )
    return model

def test_model(model, data_set, tensorboard_callback, test_batch_size):
    x_test = data_set.x_test()
    y_test = data_set.y_test()
    assert(len(x_test.shape) == 2)
    assert(len(y_test.shape) == 2)
    res = model.evaluate([x_test, y_test], [y_test, x_test],
                         batch_size=test_batch_size,
                         #callbacks=[tensorboard_callback],
                         )

    print(list(zip(model.metrics_names,res)))


def check_data(data_set, tensorboard_callback):
    x_test = data_set.x_test()
    y_test = data_set.y_test()
    print(util._eval_tensor(util.cross_correlation_analysis(x_test, y_test, 50, False)))

def train_and_test(dataset_file_ini):
    data_set = load_data_set(dataset_file_ini)
    tensorboard_callback = tensorboardimage.create_tensorboard_callback()
    m = train_model(data_set, tensorboard_callback)
    #m.save("model2way.h5")
    test_model(m, data_set, tensorboard_callback, data_set.params().BATCH_SIZE)
    test_model(m, data_set, tensorboard_callback, 6000)
    ##check_data(data_set, tensorboard_callback)


def main(argv):
    if len(argv) < 2:
        print("ERROR - must give a <DATASET>.ini file name")
        return 3
    train_and_test(argv[1])
    return 0

if __name__ == '__main__':
    main(sys.argv)
