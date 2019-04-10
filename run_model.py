# ##
# # for Python 2 on mac use: $ export LC_ALL=en_US.UTF-8;export LANG=en_US.UTF-8
# from math import ceil
#
# import matplotlib
# import traceback
#
# from testing import test_model
#
import configparser
import os
import sys
import layers.tied
import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate


def create_dataset(name, config):
    import params
    cls_params = params.get_params_class_for_dataset_name(name)
    params = cls_params(config)
    # DEBUG
    data_loader = params.DATA_CLASS(params)
    return data_loader

def build_model(data_set):
    x_train = data_set.x_train()
    y_train = data_set.y_train()
    assert(len(x_train.shape) == 2) # TODO(franji): flatten!
    assert(len(y_train.shape) == 2)
    x_input_size = x_train.shape.as_list()[1]
    y_input_size = y_train.shape.as_list()[1]
    xy_input = tf.keras.Input(shape=(x_input_size + y_input_size,))
    x_input = xy_input[:, :x_input_size]
    y_input = xy_input[:, x_input_size:]
    prev_layer_size = x_input_size
    assert(len(y_train.shape) == 2)
    y_ouput_size = y_train.shape[1]
    layers_x_to_y = []
    layers_y_to_x = []
    is_last_layer = False
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
        LXY = layer_type(units=layer_size)
        activation_xy = activation_yx = None
        if not is_last_layer:
            activation_xy = LeakyReLU(alpha=data_set.params().LEAKINESS)
            activation_yx = LeakyReLU(alpha=data_set.params().LEAKINESS)
        batch_norm_xy = batch_norm_yx = None
        if not is_last_layer and data_set.params().BN:
            batch_norm_xy = BatchNormalization()
            batch_norm_yx = BatchNormalization()
        # We need to build LXY so we can tie internal kernel to LYX
        xy_input_shape = (None, prev_layer_size)
        print(f"Layer X->Y build {type(LXY)}(units={layer_size},input_shape={xy_input_shape})")
        LXY.build(input_shape=xy_input_shape)
        # We use prev_layer_size as number of units to the reverse layer
        layer_kwargs = dict(units=prev_layer_size)
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

        prev_layer_size = layer_size

    channel_x_to_y = x_input
    for lay in layers_x_to_y:
        if lay is None:
            continue
        print(type(channel_x_to_y))
        channel_x_to_y = lay(channel_x_to_y)

    channel_y_to_x = y_input
    for lay in reversed(layers_y_to_x):
        if lay is None:
            continue
        print(type(channel_y_to_x))
        channel_y_to_x = lay(channel_y_to_x)
    xy_output = concatenate([channel_x_to_y, channel_y_to_x], axis=1)
    model = tf.keras.Model(input=xy_input, output=xy_output)
    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.01, momentum=0.8)
    model.compile(optimizer, loss='mean_squared_error')
    return model


def run_model(data_set_config):
    model_results = {'train': [], 'validate': []}
    results_folder = os.path.join(os.getcwd(), 'results')

    data_config = configparser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = data_config["dataset_parameters"]
    for key in data_parameters:
        print(f"data_parameters[{key}] = {data_parameters[key]}")
    # construct data set
    data_set = create_dataset(data_parameters['name'], data_parameters)
    data_set.load()
    model = build_model(data_set)
    history = History()
    model.fit(data_set.x_train(), data_set.y_train, epochs=50, steps_per_epoch=100, callbacks=[history])


def main(argv):
    if len(argv) < 2:
        print("ERROR - must give a <DATASET>.ini file name")
        return 3
    run_model(argv[1])


if __name__ == '__main__':
    main(sys.argv)
