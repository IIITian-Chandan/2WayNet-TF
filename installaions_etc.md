# 2WayNet-TF - environment configurations
This is a kind of a dump of things I did to develop/run 2WayNet-TF on my Mac

installed conda for mac

conda create -n tensorflow_env tensorflow

to activate:

conda activate tensorflow_env

\# configparser used in the original project for dataset config files

conda install -c anaconda configparser=3.7.3

# Changes in design - TF
Moving the code to TensorFlow required some changes - I list them here as I
work thorugh it.

old DataSetBase uses .npy file for saving the datasets. TensorFlow uses TFRecord protobuf format.
You can see discussion about these formats look here: 
https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline

I have desided to stick to TFRecord

