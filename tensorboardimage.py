import datetime
import os
import re
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# some magic to allow using tensor board both on laptop and on Google Colab
use_tensorboardcolab = False
if use_tensorboardcolab:
    try:
        from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
        g_tensorboard_colab = TensorBoardColab()
        class_callback_TensorBoard = TensorBoardColabCallback
        use_tensorboardcolab = True
    except:
        use_tensorboardcolab = False

if not use_tensorboardcolab:
    class_callback_TensorBoard = TensorBoard


class TensorBoardImage(class_callback_TensorBoard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_variables = []
        self.scalar_metics = []
        self.count_epoch = 0

    def add_image_variables(self, variables):
        #self.image_variables.extend(variables)
        for var in variables:
            self._summary_image_var(var)

    def add_scalar(self, name, scalar):
        #self.scalar_metics.append( (name, scalar) )
        return tf.summary.scalar(name, scalar)

    def _summary_image_var(self, var):
        img_gray = tf.stack([var], axis=2)  # tf.summary.image wants 4D tensor
        img_gray = tf.stack([img_gray], axis=0)
        name = var.name.replace(":", "_")
        return tf.summary.image(name, img_gray)

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        self.count_epoch += 1
        sess = tf.keras.backend.get_session()
        merged_summary_op = tf.summary.merge_all()
        with tf.summary.FileWriter(self.log_dir, session=sess) as writer:
            writer.add_summary(sess.run(merged_summary_op), global_step=self.count_epoch)
        return
        with tf.summary.FileWriter(self.log_dir, session=sess) as writer:
            for var in self.image_variables:
                img_gray = tf.stack([var], axis=2) # tf.summary.image wants 4D tensor
                img_gray = tf.stack([img_gray], axis=0)
                name = var.name.replace(":", "_")
                writer.add_summary(sess.run(tf.summary.image(name, img_gray)), global_step=self.count_epoch)
        summaries = []
        with tf.summary.FileWriter(self.log_dir, session=sess) as writer:
            for name, scalar in self.scalar_metics:
                summ = tf.summary.scalar(name, scalar)
                writer.add_summary(sess.run(summ), global_step=self.count_epoch)
            #summaries.append(summ)
        #writer.add_summary(sess.run(tf.summary.merge(summaries)), global_step=self.count_epoch)


def find_base_log_dir():
    for s in ["/home/ubuntu/filestore/chandan/2WayNet-TF/content", "/home/ubuntu/filestore/chandan/2WayNet-TF/tmp"]:
        if os.path.isdir(s):
            logs = os.path.join(s, "log")
            if not os.path.isdir(logs):
                os.mkdir(logs)
            return logs

g_log_dir = find_base_log_dir() + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
def create_tensorboard_callback():
    if use_tensorboardcolab:
        tensorboard_callback = TensorBoardImage(g_tensorboard_colab)
    else:
        tensorboard_callback = TensorBoardImage(log_dir=g_log_dir)
    return tensorboard_callback

