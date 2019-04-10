import params.DatasetConfig
from layers.tied import *

class XRMB_Params(params.DatasetConfig.DatasetConfig):
    name = "XRMB"
    # region Training Params
    BATCH_SIZE = 128
    VALIDATION_BATCH_SIZE = 1000
    CROSS_VALIDATION = True
    EPOCH_NUMBER = 60
    DECAY_EPOCH = [20, 40, 60, 80]
    DECAY_RATE = 0.5
    BASE_LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    # endregion

    # region Loss Weights
    WEIGHT_DECAY = 0.05
    GAMMA_COEF = 0.05
    WITHEN_REG_X = 0.05
    WITHEN_REG_Y = 0.05
    L2_LOSS = 0.5
    LOSS_X = 1
    LOSS_Y = 1
    # endregion

    # region Architecture
    LAYERS_SPEC = [
        # format of a layer spec - (type, size)
        # or a single layer - the representaton layer the format is (type, size, True)
        # Types of layers can be TiedDenseLayer or LocallyDenseLayer
        (TiedDenseLayer, 560, True),
        (TiedDenseLayer, 280),
        (TiedDenseLayer, 112),
        (TiedDenseLayer, 680),
        (TiedDenseLayer, 1365),
        (TiedDenseLayer, -1),
    ]
    DROP_PROBABILITY = [0.5, 0.5, 0.5]
    LEAKINESS = 0.3
    LOCALLY_DENSE_M = 2
    NOISE_LAYER = TiedDropoutLayer
    BN = True
    BN_ACTIVATION = False
    SIMILARITY_METRIC = 'correlation'

    # endregion
