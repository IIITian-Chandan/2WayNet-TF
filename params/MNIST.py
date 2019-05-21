import params.DatasetConfig
from layers.tied import *
import datasets.base

class MNIST_Params(params.DatasetConfig.DatasetConfig):
    """
    Parameters for the training and inference of the 2-WayNet
    """
    # region Dataset
    name = "MNIST"
    DATA_CLASS = datasets.base.MNISTDataset
    # endregion

    # region Training Params
    BATCH_SIZE = 128  # number of samples in the batch for training
    VALIDATION_BATCH_SIZE = 1000  # number of samples in the batch for testing
    CROSS_VALIDATION = True  # enable the running on validation after each epoch
    EPOCH_NUMBER = 20  # was 80, using accelerated/deaccelrated learning-rate - we can do with less 80  # number of epochs
    ##DECAY_EPOCH = [20, 40, 60, 80]  # epochs which include a learning rate decay
    ##DECAY_RATE = 0.5  # The factor to multiply the learning rate in each decay
    BASE_LEARNING_RATE = 0.0001  # starting learning rate
    MOMENTUM = 0.9  # momentum for the training
    # endregion

    # region Loss Weights
    # Coefficients for the loss and regularization terms
    WEIGHT_DECAY = 0.05 # 0.05
    GAMMA_COEF = 0.05 # 0.05
    WITHEN_REG_X = 0.05 #0.05
    WITHEN_REG_Y = 0.05 #0.05
    L2_LOSS = 0.5 #0.5
    LOSS_X =  1.
    LOSS_Y =  1.
    # endregion

    # region Architecture
    LAYERS_SPEC = [
        # format of a layer spec - (type, size)
        # for (only) one of the layers - the representaton layer - the format is (type, size, True)
        # Types of layers can be TiedDenseLayer or LocallyDenseLayer
        # size==-1 is for the output layer. the size is same as output
        (TiedDenseLayer, 392),
        (TiedDenseLayer, 50, True),
        (TiedDenseLayer, 392),
        (TiedDenseLayer, -1)
    ]
    DROP_PROBABILITY = 0.5  # Probability for removing a neuron in the dropout/tied dropout layer
    LEAKINESS = 0.3  # Leakiness coefficient
    LOCALLY_DENSE_M = -1  # The number of sub-dense layer in the locally dense layer
    NOISE_LAYER = TiedDropoutLayer  # The type of dropout layer can be TiedDropoutLayer or Dropoutlayer
    BN = True  # If True uses batch normalization
    BN_ACTIVATION = False  # Controls the order of non-linearity, if True the non-linearity is performed after the BN
    # endregion
