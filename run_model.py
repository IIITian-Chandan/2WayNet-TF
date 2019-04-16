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

from tensorflow.python.keras import losses
import datetime
import layers.tied
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate

# some magic to allow using tensor board both on laptop and on Google Colab
try:
    from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
    g_tensorboard_colab = TensorBoardColab()
    class_callback_TensorBoard = TensorBoardColabCallback
    use_tensorboardcolab = True
except:
    use_tensorboardcolab = False
    class_callback_TensorBoard = TensorBoard


class TensorBoardImage(class_callback_TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_tensors = []
        self.image_tags = []

    def add_image_tensors(self, tensors, tags):
        assert(len(tensors) == len(tags))
        self.image_tensors.extend(tensors)
        self.image_tags.extend(tags)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        sess = tf.keras.backend.get_session()
        ##tensors = [sess.run(t) for t in self.image_tensors]

        for img, tag in zip(self.image_tensors, self.image_tags):
            writer = tf.summary.FileWriter(self.log_dir)
            img_gray = tf.stack([img], axis=2) # tf.summary.image wants 4D tensor
            img_gray = tf.stack([img_gray], axis=0)
            writer.add_summary(sess.run(tf.summary.image(tag, img_gray)))
            writer.close()

        return

    def on_batch_end(self, batch, logs={}):
        super().on_batch_end(batch, logs)


def create_dataset(name, config):
    import params
    cls_params = params.get_params_class_for_dataset_name(name)
    params = cls_params(config)
    # DEBUG
    data_loader = params.DATA_CLASS(params)
    return data_loader


BOOKMARK_REPRESENTATION_LAYER = "representation_layer"


def build_model(data_set, tensorboard_callback):
    x_train = data_set.x_train()
    y_train = data_set.y_train()
    assert(len(x_train.shape) == 2) # TODO(franji): flatten!
    assert(len(y_train.shape) == 2)
    x_input_size = x_train.shape.as_list()[1]
    y_input_size = y_train.shape.as_list()[1]
    x_input = tf.keras.Input(shape=(x_input_size,))
    y_input = tf.keras.Input(shape=(y_input_size,))
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
        if is_representation_layer:
            # add a "bookmark" to the layers list
            layers_x_to_y.append(BOOKMARK_REPRESENTATION_LAYER)
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
    for lay in layers_x_to_y:
        if lay is None:
            continue
        if lay == BOOKMARK_REPRESENTATION_LAYER:
            is_representation_layer = True  # mark for next
            continue
        channel_x_to_y = lay(channel_x_to_y)
        if is_representation_layer:
            # in this channel the bookmark is BEFORE the layer
            assert(representation_layer_xy is None)
            representation_layer_xy = channel_x_to_y
            is_representation_layer = False

    channel_y_to_x = y_input
    representation_layer_yx = None
    for lay in reversed(layers_y_to_x):
        if lay is None:
            continue
        if lay == BOOKMARK_REPRESENTATION_LAYER:
            # in this channel the bookmark is AFTER the layer
            assert (representation_layer_yx is None)
            representation_layer_yx = channel_y_to_x
            continue
        channel_y_to_x = lay(channel_y_to_x)

    loss_x = data_set.params().LOSS_X * losses.mean_squared_error(x_input, channel_y_to_x)
    loss_y = data_set.params().LOSS_Y * losses.mean_squared_error(y_input, channel_x_to_y)
    loss_representation = 0.0
    assert(representation_layer_xy is not None
           and representation_layer_yx is not None)
    if data_set.params().L2_LOSS != 0.0:
        loss_representation = data_set.params().L2_LOSS * losses.mean_squared_error(representation_layer_xy, representation_layer_yx)

    def combined_loss(_y_true_unused, _y_pred_unused):
        return loss_x + loss_y + loss_representation

    # add images to see what's going on:
    # this is dataset specific
    # TODO(franji): move into the data set somehow
    def to_img(t, var_name):
        shape = (14, 28)
        img_var = tf.keras.backend.variable(tf.zeros(shape=shape, dtype=tf.uint8),
                                                    name=var_name, dtype=tf.uint8)
        # use t[0] - first in the batch
        img = tf.reshape(tf.cast(t[0] * 255, tf.uint8), shape=(14,28))
        return tf.assign(img_var, img), img_var

    assign_x_input_image, x_input_image = to_img(x_input, "x_input_image")
    assign_x_output_image, x_output_image = to_img(channel_y_to_x, "x_output_image")
    assign_y_input_image, y_input_image = to_img(y_input, "y_input_image")
    assign_y_output_image, y_output_image = to_img(channel_x_to_y, "y_output_image")
    tensorboard_callback.add_image_tensors([x_input_image, x_output_image, y_input_image, y_output_image],
                                           ["x_input", "x_output", "y_input", "y_output"])
    def dummy_metic_for_images(_y_true_unused, _y_pred_unused):
        return (tf.reduce_sum(assign_x_input_image) +
                tf.reduce_sum(assign_x_output_image) +
                tf.reduce_sum(assign_y_input_image) +
                tf.reduce_sum(assign_y_output_image))

    model = tf.keras.Model(inputs=[x_input, y_input],
                           outputs=[channel_x_to_y, channel_y_to_x])
    optimizer = tf.train.MomentumOptimizer(use_nesterov=True, learning_rate=0.01, momentum=0.8)
    model.compile(optimizer, loss=combined_loss, metrics=[dummy_metic_for_images])
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
    log_dir = "/Users/talfranji/tmp/log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if use_tensorboardcolab:
        tensorboard_callback = TensorBoardImage(g_tensorboard_colab)
    else:
        tensorboard_callback = TensorBoardImage(log_dir=log_dir)
    model = build_model(data_set, tensorboard_callback)
    #history = History()

    model.fit([data_set.x_train(), data_set.y_train()],
              [data_set.y_train(), data_set.x_train()],
              epochs=50, steps_per_epoch=100, callbacks=[tensorboard_callback]
              )

def debug_game():
    with tf.Session() as sess:
        shape = (14, 28)
        img_var = tf.keras.backend.variable(tf.zeros(shape=shape, dtype=tf.uint8),
                                            name="test_img_var", dtype=tf.uint8)
        v = tf.random.uniform(
            shape,
            minval=0,
            maxval=255,
            dtype=tf.int32,

        )
        img = tf.reshape(tf.cast(v, tf.uint8), shape=shape)
        assign = tf.assign(img_var, img)
        sess.run(assign)
        tensors = [img]

        for img, tag in zip(tensors, ["test_image"]):
            # image = make_image(img)
            writer = tf.summary.FileWriter("/Users/talfranji/tmp/log/")
            img_gray = tf.stack([img], axis=2)
            img_gray = tf.stack([img_gray], axis=0)
            writer.add_summary(sess.run(tf.summary.image(tag, img_gray)))
            writer.flush()
            writer.close()


def main(argv):
    if len(argv) < 2:
        print("ERROR - must give a <DATASET>.ini file name")
        return 3
    run_model(argv[1])


if __name__ == '__main__':
    main(sys.argv)
