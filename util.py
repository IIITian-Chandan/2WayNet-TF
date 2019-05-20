import tensorflow as tf
from tensorflow.keras import backend as K

# Some code taken from here:
# https://github.com/DTaoo/DCCA
# with some fixes and modifications

def center(m):
    """m is MxN matrix. M vectors from R^N.
    center each of the M vectors by subracting the mean of the N numers
    """
    dim = shape_i(m, 1)
    if dim < 30:
        # don't center on low dimentions as it creates dependency
        return m
    N = tf.cast(dim, m.dtype)
    # m*Ones(N,N) gives an MxN matric where each line has identical N numbers - the sum of
    # the vector.
    # divide by N to get mean and then subtract.
    return m - tf.divide(tf.matmul(m, tf.ones([N, N], dtype=m.dtype)), N)

def _eval_tensor(t):
    return tf.Session().run(t)

def _tensor2const(t):
    return tf.constant(_eval_tensor(t))

def _test_center():
    x = tf.constant([
        [1,2,3]*10,
        [10,20,30]*10,
        [0.1, 0.2, 0.3]*10,
        [-70, -80, -90]*10
    ])
    x1 = center(x)
    r = _eval_tensor(x1)
    eps = 1e-7
    assert(abs(r[0,0] - -1) <= eps)
    assert(abs(r[1,0] - -10) <= eps)
    assert(abs(r[2,0] - -0.1) <= eps)
    assert(abs(r[3,2] - -10) <= eps)


def shape_i(x, i):
    if hasattr(x.shape, "as_list"):
        return x.shape.as_list()[i]
    return x.shape[i]

def inverse_root_via_eigenvalues(m):
    ev, v = tf.linalg.eigh(m)
    u = v
    epsillon = 1e-8  # for numerical stability - clip
    ev = tf.where(ev > epsillon, x=ev, y=K.ones_like(ev))
    ev_inv_root = tf.math.reciprocal(tf.math.sqrt(ev))
    return tf.matmul(tf.matmul(u, tf.diag(ev_inv_root)), tf.transpose(v))


def debug_tf(title, t):
    #print(title, _eval_tensor(t))
    pass


def cross_correlation_analysis(X, Y, top_k):
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
    assert(shape_i(X, 0) == shape_i(Y,0))  # same number of samples
    Nm1 = tf.cast(K.shape(X)[0] - 1, X.dtype)  # size of sample/batch
    Xm = center(X)
    debug_tf("Xm\n", Xm)
    Ym = center(Y)
    # SigmaXY - the cross correlation
    SigmaXY = tf.divide(tf.matmul(tf.transpose(Xm), Ym), Nm1)  # X_T*Y/n
    # TODO(franji): regularisation r*I hurts testing - need to check if it helps in real
    # learning scenario
    WHITEN_REG = 1e-8
    eye_reg_x = tf.eye(K.shape(X)[1]) * WHITEN_REG
    eye_reg_y = tf.eye(K.shape(Y)[1]) * WHITEN_REG

    # We use code inspired by [DTAOO]
    SigmaXX = tf.divide(tf.matmul(tf.transpose(Xm), Xm), Nm1) #+ eye_reg_x
    SigmaXX_inv_root = inverse_root_via_eigenvalues(SigmaXX)
    SigmaYY = tf.divide(tf.matmul(tf.transpose(Ym), Ym), Nm1) #+ eye_reg_y
    SigmaYY_inv_root = inverse_root_via_eigenvalues(SigmaYY)

    C = tf.matmul(tf.matmul(SigmaXX_inv_root, SigmaXY), SigmaYY_inv_root)
    CC = tf.matmul(tf.transpose(C), C)
    U = tf.linalg.eigvalsh(CC)
    U_sort, _ = tf.nn.top_k(U, top_k)
    corr = K.sum(K.sqrt(U_sort))
    return corr


def _test_concat_normals(count, stddev_list):
    columns = []
    for stddev in stddev_list:
        columns.append(tf.random.normal(shape=[count, 1], mean=0., stddev=stddev))
    return tf.concat(columns, axis=1)


def _test_cross_correlation_analysis():
    M = 50
    X = _tensor2const(_test_concat_normals(M, [1., 2., 30., 7.5, 13.]))
    top_k = 5
    top = _eval_tensor(cross_correlation_analysis(X, 1.3*X, top_k=top_k))
    assert(abs(top - top_k) < 1e-4)


def _test():
    _test_center()
    _test_cross_correlation_analysis()