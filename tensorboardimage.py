import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

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
        self.image_variables = []
        self.image_tags = []

    def add_image_variables(self, variables):
        self.image_variables.extend(variables)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        sess = tf.keras.backend.get_session()
        for var in self.image_variables:
            writer = tf.summary.FileWriter(self.log_dir)
            img_gray = tf.stack([var], axis=2) # tf.summary.image wants 4D tensor
            img_gray = tf.stack([img_gray], axis=0)
            writer.add_summary(sess.run(tf.summary.image(var.name, img_gray)))
            writer.close()

        return

g_log_dir = "/Users/talfranji/tmp/log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
def create_tensorboard_callback():
    if use_tensorboardcolab:
        tensorboard_callback = TensorBoardImage(g_tensorboard_colab)
    else:
        tensorboard_callback = TensorBoardImage(log_dir=g_log_dir)
    return tensorboard_callback

