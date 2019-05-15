import tensorflow as tf
from tensorflow.keras import backend as K

# Some code taken from here:
# https://github.com/DTaoo/DCCA
# with some fixes and modifications

def center(m):
    """m is MxN matrix. M vectors from R^N.
    center each of the M vectors by subracting the mean of the N numers
    """
    N = shape_i(m.shape, 1)
    # m*Ones(N,N) gives an MxN matric where each line has identical N numbers - the sum of
    # the vector.
    # divide by N to get mean and then subtract.
    return m - tf.divide(tf.matmul(m, tf.ones([N, N], dtype=m.dtype)), N)

def _eval_tensor(t):
    return tf.Session().run(t)

def _test_center():
    x = tf.constant([
        [1,2,3],
        [10,20,30],
        [0.1, 0.2, 0.3],
        [-70, -80, -90]
    ])
    x1 = center(x)
    r = _eval_tensor(x1)
    eps = 1e-8
    assert(abs(r[0,0] - -1) <= eps)
    assert(abs(r[1,0] - -10) <= eps)
    assert(abs(r[2,0] - -0.1) <= eps)
    assert(abs(r[3,2] - -10) <= eps)


def DELME_eigenvalues_eps(m, eps):
    """calculate eigen values and vectors for eigenvalues > eps
    return: (eigenvalues_tensor, eigenvectors_tensor)"""
    e, v = tf.linalg.eigh(m)
    mask = tf.squeeze(tf.where(e > eps, x=tf.ones([m.shape[0]]), y=tf.zeros([m.shape[0]])))
    e1 = mask * e
    v1 = tf.reshape(mask, shape=[-1,1]) * v
    return e1, v1


def DELME_test_eigenvalues_eps():
    diag = tf.constant([1.,2.,3.,4.])
    x = tf.diag(diag)
    e, v = eigenvalues_eps(x, 1)
    assert(_eval_tensor(e[0]) == 0)
    assert(_eval_tensor(tf.math.count_nonzero(v[0])) == 0)
    assert(_eval_tensor(tf.math.count_nonzero(v[1])) == 1)
    assert(_eval_tensor(tf.math.count_nonzero(v[2])) == 1)
    assert(_eval_tensor(tf.math.count_nonzero(v[3])) == 1)


def DELME_stable_mat_inv_root(m, eps):
    """calculate numerically stable m^(-0.5)
    Use the eigen values of m which are larger than epsilon"""
    e, sv = DELME_eigenvalues_eps(m, eps)
    e_inv_sqrt = tf.pow(e, -0.5)
    m_inv_sqrt = tf.matmul(tf.matmul(sv, tf.diag(e_inv_sqrt)), tf.transpose(sv))
    return m_inv_sqrt

def shape_i(shape, i):
    if hasattr(shape, "as_list"):
        return shape.as_list()[i]
    return shape[i]

def cross_correlation_analysis(X, Y, top_k):
    """Given X: shape[n,P], Y: shape[n,Q]
    Calaculate the top_k correlation between the P and Q features.
    a.k.a. CCA
    related papers:
    [ANDREW2013] https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
    https://arxiv.org/pdf/1506.08170
    original code:
    [AVIV2WAY] https://github.com/aviveise/2WayNet/blob/master/MISC/utils.py
    """

    assert(shape_i(X.shape, 0) == shape_i(Y.shape,0))  # same number of samples
    n = tf.cast(K.shape(X)[0], X.dtype)  # size of sample/batch
    Xm = center(X)
    Ym = center(Y)
    # SigmaXY - the cross correlation
    SigmaXY = tf.divide(tf.matmul(tf.transpose(Xm), Ym), n)  # X_T*Y/n

    # Original code
    ##WHITEN_REG = 1e-8
    ##eye_reg_x = tf.eye(K.shape(X)[1]) * WHITEN_REG
    # SigmaXX "whitening" of X - removing self correlation
    ##SigmaXX = eye_reg_x + tf.divide(tf.matmul(tf.transpose(Xm), Xm), n)  # X_T*X/n
    # Note - the use of diag(diag_part()) is taken from [AVIV2WAY] (see ref above)
    # the paper [ANDREW2013] does not use that and uses the eye_reg_x to claim
    # this makes SigmaXX non-singular. In practice we got SigmaXX singular in many cases
    # which I assume is why [AVIV2WAY] uses diag - but there is no formal analysis of that
    ##SigmaXX = tf.diag(tf.diag_part(SigmaXX))
    #SigmaXX_inv_root = tf.linalg.inv(tf.linalg.sqrtm(SigmaXX))  # Sxx^-0.5

    # because we use diagonal - the above code is optimized to
    SigmaXX = tf.divide(tf.matmul(tf.transpose(Xm), Xm), n)
    SigmaXX_inv_root = tf.diag(
          tf.math.reciprocal(
              tf.math.sqrt(tf.diag_part(SigmaXX))))

    SigmaYY = tf.divide(tf.matmul(tf.transpose(Ym), Ym), n)
    SigmaYY_inv_root = tf.diag(
          tf.math.reciprocal(
              tf.math.sqrt(tf.diag_part(SigmaYY))))

    C = tf.matmul(tf.matmul(SigmaXX_inv_root, SigmaXY), SigmaYY_inv_root)
    CC = tf.matmul(tf.transpose(C), C)
    ev = tf.linalg.eigvalsh(CC)
    return tf.math.reduce_sum(tf.nn.top_k(ev, top_k)[0])


def _test_concat_normals(count, stddev_list):
    columns = []
    for stddev in stddev_list:
        columns.append(tf.random.normal(shape=[count, 1], stddev=stddev))
    return tf.concat(columns, axis=1)


def _test_cross_correlation_analysis():
    M = 5
    X = _test_concat_normals(M, [1., 2., 3.])
    Y = _test_concat_normals(M, [0.5, 5., 12.])

    top = _eval_tensor(cross_correlation_analysis(X,Y, top_k=1))
    print(top)
    #assert(top[0] > 0.25 and top[0] < 0.35)


def _test():
    _test_center()
    ##DELME _test_eigenvalues_eps()
    _test_cross_correlation_analysis()