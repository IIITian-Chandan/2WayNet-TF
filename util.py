import tensorflow as tf
# Some code taken from here:
# https://github.com/DTaoo/DCCA
# with some fixes and modifications

def center(m):
    """m is MxN matrix. M vectors from R^N.
    center each of the M vectors by subracting the mean of the N numers
    """
    N = m.shape.as_list()[1]
    # m*Ones(N,N) gives an MxN matric where each line has identical N numbers - the sum of
    # the vector.
    # divide by N to get mean and then subtract.
    return m - tf.divide(tf.matmul(m, tf.ones([N, N])), N)

def _eval_tensor(t):
    return tf.Session().run(t)

def _test_center():
    x = tf.constant([
        [1,2,3],
        [10,20,30],
        [0.1, 0.2, 0.3],
        [70,80,90]
    ])
    x1 = center(x)
    r = _eval_tensor(x1)
    eps = 1e-8
    assert(abs(r[0,0] - -1) <= eps)
    assert(abs(r[1,0] - -10) <= eps)
    assert(abs(r[2,0] - -0.1) <= eps)
    assert(abs(r[3,2] - 10) <= eps)


def eigenvalues_eps(m, eps):
    """calculate eigen values and vectors for eigenvalues > eps
    return: (eigenvalues_tensor, eigenvectors_tensor)"""
    e, v = tf.linalg.eigh(m)
    mask = tf.squeeze(tf.where(e > eps, x=tf.ones([m.shape[0]]), y=tf.zeros([m.shape[0]])))
    e1 = mask * e
    v1 = tf.reshape(mask, shape=[-1,1]) * v
    return e1, v1


def _test_eigenvalues_eps():
    diag = tf.constant([1.,2.,3.,4.])
    x = tf.diag(diag)
    e, v = eigenvalues_eps(x, 1)
    assert(_eval_tensor(e[0]) == 0)
    assert(_eval_tensor(tf.math.count_nonzero(v[0])) == 0)
    assert(_eval_tensor(tf.math.count_nonzero(v[1])) == 1)
    assert(_eval_tensor(tf.math.count_nonzero(v[2])) == 1)
    assert(_eval_tensor(tf.math.count_nonzero(v[3])) == 1)


def stable_mat_inv_root(m, eps):
    """calculate numerically stable m^(-0.5)
    Use the eigen values of m which are larger than epsilon"""
    e, sv = eigenvalues_eps(m, eps)
    e_inv_sqrt = tf.pow(e, -0.5)
    m_inv_sqrt = tf.matmul(tf.matmul(sv, tf.diag(e_inv_sqrt)), tf.transpose(sv))
    return m_inv_sqrt

def cross_correlation_analysis(X, Y, top_k, eps=1e-12):
    """Given X: shape[n,P], Y: shape[n,Q]
    Calaculate the top_k correlation between the P and Q features.
    a.k.a. CCA
    related papers:
    https://arxiv.org/pdf/1506.08170
    https://ttic.uchicago.edu/~klivescu/papers/andrew_icml2013.pdf
    """
    assert(X.shape[0] == Y.shape[0])  # same number of samples
    n = X.shape.as_list()[0]  # size of sample/batch
    epsilon = 1e-8
    Xm = center(X) + epsilon
    Ym = center(Y) + epsilon
    # SigmaXY - the cross correlation
    SigmaXY = tf.divide(tf.matmul(tf.transpose(Xm), Ym), n)  # X_T*Y/n
    # SigmaYY "whitening" of Y - removing self correlation
    SigmaYY = tf.divide(tf.matmul(tf.transpose(Ym), Ym), n)  # Y_T*Y/n
    # TODO(franji): is there a numerical stability issue here?
    #   - the code in DCCA uses eigen values >eps for this here. why?
    ##SigmaYY_inv_root = stable_mat_inv_root(SigmaYY, epsilon)  # Syy^-0.5
    SigmaYY_inv_root = tf.linalg.inv(tf.linalg.sqrtm(SigmaYY))  # Syy^-0.5
    # SigmaXX "whitening" of X - removing self correlation
    SigmaXX = tf.divide(tf.matmul(tf.transpose(Xm), Xm), n)  # X_T*X/n
    ##SigmaXX_inv_root = stable_mat_inv_root(SigmaXX, epsilon)  # Sxx^-0.5
    SigmaXX_inv_root = tf.linalg.inv(tf.linalg.sqrtm(SigmaXX))  # Sxx^-0.5
    C = tf.matmul(tf.matmul(SigmaXX_inv_root, SigmaXY), SigmaYY_inv_root)
    CC = tf.matmul(tf.transpose(C), C)
    ev = tf.linalg.eigvalsh(CC)
    return tf.nn.top_k(ev, top_k)


def _test_cross_correlation_analysis():
    M = 1000
    #C1 = tf.ones(shape=[M, 1])
    C2 = tf.random.normal(shape=[M, 1], stddev=1.0)
    C3 = tf.random.normal(shape=[M, 1], stddev=2.0)
    C4 = tf.random.normal(shape=[M, 1], stddev=3.0)
    X = tf.concat([C2, C3, C4], axis=1)
    C1 = tf.ones(shape=[M, 1])
    C2 = tf.random.normal(shape=[M, 1], stddev=1.0)
    C3 = tf.random.normal(shape=[M, 1], stddev=2.0)
    C4 = tf.random.normal(shape=[M, 1], stddev=3.0)
    Y = tf.concat([C4, C2, C3], axis=1)
    print(_eval_tensor(cross_correlation_analysis(X,Y, top_k=3)))


def _test():
    _test_center()
    _test_eigenvalues_eps()
    _test_cross_correlation_analysis()