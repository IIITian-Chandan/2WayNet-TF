import tensorflow as tf


class LocallyDenseLayer(tf.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
