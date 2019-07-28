import tensorflow as tf
from tensorflow.keras import backend as K
import random
import numpy
import scipy

import os
# HACK needed for some stupid error riunning Aviv's code
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Some code taken from here:
# https://github.com/DTaoo/DCCA
# with some fixes and modifications

def center(m):
    """m is MxN matrix. M vectors from R^N.
    center each of the M vectors by subracting the mean of the N numers
    """
    # Note - this implementation of center() is a little different
    # than in the original 2WayNet by Aviv.
    # this code is impler but creates small differences in numerical resulst
    # When compared to the original version.
    mean = K.mean(m, axis=0)
    return m - mean

def _eval_tensor(t):
    if isinstance(t, tf.Tensor):
        return tf.Session().run(t)
    else:
        return t

def _tensor2const(t):
    return tf.constant(_eval_tensor(t))

def _test_center():
    x = tf.constant([
        [1,20,0.3]*10,
        [2,10,0.1]*10,
        [3, 30, 0.2]*10,
    ])
    x1 = center(x)
    r = _eval_tensor(x1)
    eps = 1e-7
    assert(abs(r[0,0] - -1) <= eps)
    assert(abs(r[0,1] - 0) <= eps)
    assert(abs(r[0,2] - 0.1) <= eps)


def shape_i(x, i):
    if hasattr(x.shape, "as_list"):
        return x.shape.as_list()[i]
    return x.shape[i]


def inverse_root_via_diag(m):
    ev= tf.linalg.diag_part(m)
    ev = K.clip(ev, 1e-8, None)  # fix zero variance column
    ev_inv_root = tf.math.reciprocal(tf.math.sqrt(ev))
    return tf.linalg.diag(ev_inv_root)


def inverse_root_via_eigenvalues(m):
    ev, v = tf.linalg.eigh(m)
    epsillon = 1e-8  # for numerical stability - clip
    ev = tf.where(ev > epsillon, x=ev, y=K.ones_like(ev))
    v = tf.where(ev > epsillon, x=v, y=K.zeros_like(v))
    u = v
    ev_inv_root = tf.math.reciprocal(tf.math.sqrt(ev))
    res = tf.matmul(tf.matmul(u, tf.diag(ev_inv_root)), tf.transpose(v))
    return res



def debug_tf(title, t):
    #print(title, _eval_tensor(t))
    pass


def aviv_center(M):
    """
    Centers a given matrix by subtracting each column by it's mean
    :param M: the input matrix
    :return: the centered matrix and the calculated mean for each column
    """
    if M is None:
        return

    mean = M.mean(axis=1).reshape([M.shape[0], 1])
    M -= mean * numpy.ones([1, M.shape[1]])
    return M, mean

def aviv_calculate_correlation_numpy(x, y):
    """
    Returns the total correlation between x and y, if visualize equals true then a correlation matrix is saved as image
    :param x: view x - matrix of size MxD
    :param y: view y - matrix of size MxD
    :param visualize: If true outputs an image of the correlation
    :return: the sum of correlation between x and y vectors
    """
    x = _eval_tensor(x)
    y = _eval_tensor(y)

    try:
        set_size = x.shape[0]
        dim = x.shape[1]

        x, mean_x = aviv_center(x.T)
        print("DEBUG x", numpy.sum(x))
        y, mean_y = aviv_center(y.T)
        print("DEBUG y", numpy.sum(y))

        s11 = numpy.diag(numpy.diag(numpy.dot(x, x.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
        print("DEBUG s11", numpy.trace(s11))
        s22 = numpy.diag(numpy.diag(numpy.dot(y, y.T) / (set_size - 1) + 10 ** (-8) * numpy.eye(dim, dim)))
        print("DEBUG s22", numpy.trace(s22))
        s12 = numpy.dot(x, y.T) / (set_size - 1)
        print("DEBUG s12", numpy.trace(s12))

        s11_chol = scipy.linalg.sqrtm(s11)
        print("DEBUG s11_chol", numpy.trace(s11_chol))
        s22_chol = scipy.linalg.sqrtm(s22)
        print("DEBUG s22_chol", numpy.trace(s22_chol))

        s11_chol_inv = scipy.linalg.inv(s11_chol)
        print("DEBUG s11_chol_inv", numpy.trace(s11_chol_inv))
        s22_chol_inv = scipy.linalg.inv(s22_chol)
        print("DEBUG s22_chol_inv", numpy.trace(s22_chol_inv))

        mat_T = numpy.dot(numpy.dot(s11_chol_inv, s12), s22_chol_inv)

        return numpy.trace(mat_T)

    except Exception as e:
        print('Error while calculating meridia error', e)
        return 0

def aviv_calculate_correlation_tf(x, y, visualize=False):
    """
    Returns the total correlation between x and y, if visualize equals true then a correlation matrix is saved as image
    :param x: view x - matrix of size MxD
    :param y: view y - matrix of size MxD
    :param visualize: If true outputs an image of the correlation
    :return: the sum of correlation between x and y vectors
    """
    try:
        set_size = tf.cast(K.shape(x)[0], x.dtype)  # size of sample/batch
        dim = tf.cast(K.shape(x)[1], x.dtype)

        x = center(x)
        y = center(y)

        s11 = tf.diag(tf.diag_part(K.dot(tf.transpose(x), x) / (set_size - 1) + 10 ** (-8) * tf.eye(dim, dim)))
        s22 = tf.diag(tf.diag_part(K.dot(tf.transpose(y), y) / (set_size - 1) + 10 ** (-8) * tf.eye(dim, dim)))
        s12 = K.dot(tf.transpose(x), y) / (set_size - 1)

        s11_chol = tf.linalg.sqrtm(s11)
        s22_chol = tf.linalg.sqrtm(s22)

        s11_chol_inv = tf.linalg.inv(s11_chol)
        s22_chol_inv = tf.linalg.inv(s22_chol)

        mat_T = K.dot(K.dot(s11_chol_inv, s12), s22_chol_inv)
        return tf.trace(mat_T)

    except Exception as e:
        print('Error while calculating meridia error', e)
        return 0

def cross_correlation_analysis(X, Y, top_k, use_eigenvals=False):
    """Given X: shape[n,P], Y: shape[n,Q]
    Calaculate the top_k correlation between the P and Q features.
    a.k.a. CCA
    related papers:
    [ANDREW2013] https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
    https://arxiv.org/pdf/1506.08170
    original code:
    [AVIV2WAY] https://github.com/aviveise/2WayNet/blob/master/MISC/utils.py
    Reference code (implemets [ANDREW2013])
    [DTAOO] https://github.com/DTaoo/DCCA

    """
    inverse_root_func = inverse_root_via_diag
    if use_eigenvals:
        inverse_root_func = inverse_root_via_eigenvalues

    assert(shape_i(X, 0) == shape_i(Y,0))  # same number of samples
    Xm = center(X)
    ##Nm1 = tf.cast(K.shape(Xm)[0] - 1, Xm.dtype)  # size of sample/batch
    Nm1 = tf.cast(K.shape(Xm)[0], Xm.dtype) # size of sample/batch
    Ym = center(Y)
    # SigmaXY - the cross correlation
    SigmaXY = tf.divide(tf.matmul(tf.transpose(Xm), Ym), Nm1)  # X_T*Y/n
    # TODO(franji): regularisation r*I hurts testing - need to check if it helps in real
    # learning scenario
    WHITEN_REG = 1e-8
    eye_reg_x = tf.eye(K.shape(X)[1]) * WHITEN_REG
    eye_reg_y = tf.eye(K.shape(Y)[1]) * WHITEN_REG

    # We use code inspired by [DTAOO]
    debug_tf("Xm\n", Xm)
    SigmaXX = tf.divide(tf.matmul(tf.transpose(Xm), Xm), Nm1) ##+ eye_reg_x
    debug_tf("SigmaXX\n", SigmaXX)
    SigmaXX_inv_root = inverse_root_func(SigmaXX)
    debug_tf("SigmaXX_inv_root\n", SigmaXX_inv_root)
    SigmaYY = tf.divide(tf.matmul(tf.transpose(Ym), Ym), Nm1) ##+ eye_reg_y
    SigmaYY_inv_root = inverse_root_func(SigmaYY)

    C = tf.matmul(tf.matmul(SigmaXX_inv_root, SigmaXY), SigmaYY_inv_root)
    if not use_eigenvals:
        # trace of C is same as sqrt(trace(traspose(C)*C)
        U = tf.linalg.diag_part(C)
        U_sort, _ = tf.nn.top_k(U, top_k)
        corr = K.sum(U_sort)
        return corr
    CC = tf.matmul(tf.transpose(C), C)
    debug_tf("CC\n", CC)
    U = tf.linalg.eigvalsh(CC)
    U_sort, _ = tf.nn.top_k(U, top_k)
    corr = K.sum(K.sqrt(U_sort))
    return corr


def _test_concat_normals(count, stddev_list):
    columns = []
    create_linear_dependency = True
    create_column_dependency = True
    for stddev in stddev_list:
        # create 2 copies of all data - to create a linear dependeny
        if create_linear_dependency:
            assert(count % 2 == 0)
            column_parts = []
            n = count // 2
            r = tf.random.normal(shape=[n, 1], mean=0., stddev=stddev) # half
            column_parts.append(r)
            column_parts.append(r)
            column = tf.concat(column_parts, axis=0)
        else:
            column = tf.random.normal(shape=[count, 1], mean=10., stddev=stddev) # half
        columns.append(column)
    # when create_column_dependency==True - CCA should be lower - but happens only when use_eigenvals==True
    if create_column_dependency:
        # overwrite the even colums with the od ones
        for c in range(len(stddev_list) // 2):
            columns[c * 2] = columns[c * 2 + 1]
    return tf.concat(columns, axis=1)


def _test_cross_correlation_analysis():
    M = 5000
    width = 10
    stddevs = [random.uniform(0.1, 50.0) for _ in range(width)]
    #X = _tensor2const(_test_concat_normals(M, [1., 2., 30., 7.5, 13.]))
    X = _tensor2const(_test_concat_normals(M, stddevs))
    top_k = width
    top = _eval_tensor(cross_correlation_analysis(X, 1.3*X,
                                                  top_k=top_k,
                                                  use_eigenvals=False))
    print("corr=", top)
    assert(abs(top - top_k) < 1e-4)
    top = _eval_tensor(cross_correlation_analysis(X, 1.3*X,
                                                  top_k=top_k,
                                                  use_eigenvals=True))
    print("corr=", top, top - top_k // 2)
    assert(abs(top - top_k // 2) < 1e-2)

def _test_inverse_root():
    # check that for the simple case of diagonal matrics the two functions agree
    x = tf.diag([1.0, 10.0, 100.0, 1000.0])
    x1 = inverse_root_via_eigenvalues(x)
    x2 = inverse_root_via_diag(x)
    #debug_tf("inverse_root_via_eigenvalues\n", x1)
    #debug_tf("inverse_root_via_diag\n", x2)
    assert(_eval_tensor(tf.reduce_all(tf.math.equal(x1, x2))))

def _test():
    _test_center()
    _test_inverse_root()
    _test_cross_correlation_analysis()

def avg(a):
    return sum(a) / len(a)

def aviv_load(path_img):
    import struct
    import numpy as np
    import gzip
    with gzip.open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got %d' % magic)
        image_data = np.frombuffer(file.read(), dtype=np.uint8)
    images = np.reshape(image_data, (size, rows * cols))
    #images = []
    # for i in xrange(size):
    #     images.append([0] * rows * cols)
    #
    # for i in xrange(size):
    #     images[i][:] = image_data[i * rows * cols: (i + 1) * rows * cols]
    img_sz = images.shape[1]
    half_sz = img_sz // 2
    x_train, y_train = tf.split(images, 2, axis=1)

    def cast1(z):
        return tf.cast(z, dtype=tf.float32) / 255.0

    return cast1(x_train), cast1(y_train)


def _debug():
    #import datasets.base
    #datasets.base.DEBUG_SMALL_DATASET = False
    #import datasets.base
    #data = datasets.base.MNISTDataset(None)
    #data.load()
    #xfull = data.x_train()
    #yfull = data.y_train()
    M = 50000
    xfull, yfull = aviv_load("/Users/talfranji/data/MNIST/train-images-idx3-ubyte.gz")
    print("M,corr-orig,corr-whiten-diag,corr-whiten-eigenval")
    csv_end = ''  # used to chenge to new-line in debug mode
    #csv_end = "\n==============\n"
    for m in [5, 10, 50, 500, 5000, 50000]:
        #print("-" * 50)
        for i in range(1):
            start = i * m
            end = (i+1) * m
            if start > xfull.shape[0]:
                break
            x = xfull[start:end]
            y = yfull[start:end]
            print("{}".format(m), end=csv_end)
            for use_eigenvals in [False, True]:
                if use_eigenvals is None:
                    top = _eval_tensor(aviv_calculate_correlation_numpy(x, y))  ### HACK
                else:
                    top = _eval_tensor(cross_correlation_analysis(x, y,
                                            top_k=392,
                                            use_eigenvals=use_eigenvals))
                top = round(top, 2)
                print(",{}".format(top), end=csv_end)
            print()

if __name__ == '__main__':
    #_test()
    _debug()


