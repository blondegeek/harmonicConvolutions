import numpy as np
import tensorflow as tf
from sympy.physics.quantum.cg import CG
from mpmath import spherharm
from sympy.physics.quantum.spin import Rotation

order_lm_dict = {0: (0, 0),
                 1: (1, -1),
                 2: (1, 0),
                 3: (1, 1),
                 4: (2, -2),
                 5: (2, -1),
                 6: (2, 0),
                 7: (2, 1),
                 8: (2, 2),
                 9: (3, -3),
                 10: (3, -2),
                 11: (3, -1),
                 12: (3, 0),
                 13: (3, 1),
                 14: (3, 2),
                 15: (3, 3)}

def conv(X, W, strides, padding, name):
    """Shorthand for tf.nn.conv3d"""
    return tf.nn.conv3d(X, W, strides=strides, padding=padding, name=name)


def h_conv(X, W, strides=(1, 1, 1, 1, 1, 1), padding='VALID', max_order=1, name='N'):
    """Inter-order (cross-stream) convolutions can be implemented as single
    convolutions. For this we store data as 6D tensors and filters as 8D
    tensors, at convolution, we reshape down to 4D tensors and expand again.

    X: tensor shape [mbatch,h,w,channels,complex,order]
    Q: tensor dict---reshaped to [h,w,in,in.comp,in.ord,out,out.comp,out.ord]
    P: tensor dict---phases
    strides: as per tf convention (default (1,1,1,1))
    padding: as per tf convention (default VALID)
    filter_size: (default 3)
    max_order: (default 1)
    name: (default N)
    """
    with tf.name_scope('hconv' + str(name)) as scope:
        # Build data tensor
        Xsh = X.get_shape().as_list()
        Wsh = W.get_shape().as_list()
        X_ = tf.reshape(X, tf.concat(0, [Xsh[:3], [-1]]))

        #   Construct the stream-convolutions as one big filter
        Q_ = []
        for output_order in xrange(max_order + 1):
            # For each output order build input
            Qr = []
            Qi = []
            for input_order in xrange(Xsh[3]):
                for filter_order in xrange(Wsh[0]):
                    # Compute Clebsch-Gordon coeff
                    cg_input = list(order_lm_dict[input_order])
                    cg_input += list(order_lm_dict[filter_order])
                    cg_input += list(order_lm_dict[output_order])
                    cg = CG(*cg_input)
                    cg = float(cg.doit())
                    c = tf.scalar_mul(cg, W[filter_order])
                    # Choose a different filter depending on whether input is real
                    if Xsh[4] == 2:
                        Qr += [c[0], -c[1]]
                        Qi += [c[1],  c[0]]
                    else:
                        Qr += [c[0]]
                        Qi += [c[1]]
                # Need to concat to a diff dimension here?
            Q_ += [tf.concat(2, Qr), tf.concat(2, Qi)]
        Q_ = tf.concat(3, Q_)

        R = conv(X_, Q_, strides, padding, name + 'cconv')
        Rsh = R.get_shape().as_list()
        ns = tf.concat(0, [Rsh[:3], [max_order + 1, 2], [Rsh[3] / (2 * (max_order + 1))]])
        return tf.reshape(R, ns)

##### FUNCTIONS TO CONSTRUCT STEERABLE FILTERS #####
def get_filters(R, filter_size, P=None):
    """
    Return a complex filter of the form $u(r,t,psi) = R(r)e^{im(t-psi)}

    """
    filters = {}
    k = filter_size
    for order, r in R.iteritems():
        rsh = r.get_shape().as_list()
        # Get the basis matrices
        real, imag = get_complex_basis_matrices(k, order=order)
        real = tf.reshape(real, tf.pack([k * k * k, rsh[0]]))
        imag = tf.reshape(imag, tf.pack([k * k * k, rsh[0]]))

        # Project rings on to rotational basis
        r = tf.reshape(r, tf.pack([rsh[0], rsh[1] * rsh[2]]))
        ureal = tf.reshape(tf.matmul(real, r), tf.pack([k, k, k, rsh[1], rsh[2]]))
        uimag = tf.reshape(tf.matmul(imag, r), tf.pack([k, k, k, rsh[1], rsh[2]]))

        filters[order] = (ureal, uimag)

    # Each l has three steering angles that must be applied to all m's using
    # WignerD matrices

    order_max = max(R.keys())
    l_list = range(order_lm_dict[order_max][0]+1)

    # \sum^{n}_{l=0} 2*l + 1 = (n+1)**2

    # The phase dictionary contains 3 angles for every l
    if P is not None:
        for l in l_list:
            # Let angles for given l
            alpha, beta, gamma = P[l]
            # Construct -l, -l+1,...,l-1,l list
            lin = np.linspace([-l,l,2l+1])
            # Construct matrix indices for m and mprime
            m, mp = np.meshgrid(lin,lin)
            # Wigner D function
            wigD = lambda l, m, mp, alpha, beta, gamma: Rotation.D(l, m, mp, alpha, beta, gamma).doit()
            wigD_vectorize = np.vectorize(wigD)
            # Wigner D matrix
            D = wigD_vectorize(l, m, mp, alpha, beta, gamma)

            # Apply matrix to all filters
            # Get real and imaginary filters for a give l
            real_filters = np.array([filters[order][0] for order in range((l+1)**2-(2l+1),(l+1)**2)])
            imag_filters = np.array([filters[order][0] for order in range((l+1)**2-(2l+1),(l+1)**2)])

            # Save results from
            for order in range((l+1)**2-(2l+1),(l+1)**2):
                L,M = order_lm_dict(order)
                # Note, this likely won't work because of mixing numpy and tf ops -- will rewrite later
                filters[order] = (np.sum(np.multiply(D[L + M, :], real_filters), axis=0),
                                  np.sum(np.multiply(D[L + M, :], imag_filters), axis=0))
    return filters


def get_complex_basis_matrices(filter_size, order=1):
    """Return complex basis component e^{imt} (ODD sizes only).

    filter_size: int of filter height/width (default 3) CAVEAT: ODD supported
    order: rotation order (default 1)
    """

    k = filter_size
    radius = (k + 1) / 2
    n_rings = (radius * (radius + 1) * (radius + 2)) / 6

    lin = np.linspace((1. - k) / 2., (k - 1.) / 2., k)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    unique = np.unique(R)

    theta = np.arccos(Z, R)
    # why minus?
    phi = np.arctan2(-Y, X)

    # Real and imaginary mask components
    rmasks = []
    imasks = []

    # Make vectorized functions for getting spherical harmonic filters
    sh_r = lambda l, m, theta, phi: float(spherharm(l, m, theta, phi).real)
    sh_i = lambda l, m, theta, phi: float(spherharm(l, m, theta, phi).imag)
    ylm_r = np.vectorize(sh_r)
    ylm_i = np.vectorize(sh_i)

    for i in xrange(n_rings):
        l, m = order_lm_dict(order)
        if order == 0:
            # For order == 0 there is nonzero weight on the center pixel
            rmask = (R == unique[i]) * ylm_r(l,m,theta,phi)
            rmasks.append(to_constant_float(rmask))
            imask = (R == unique[i]) * ylm_i(l,m,theta,phi)
            imasks.append(to_constant_float(imask))
        elif order > 0:
            # For order > 0 there is zero weights on the center pixel
            if unique[i] != 0.:
                rmask = (R == unique[i]) * ylm_r(l,m,theta,phi)
                rmasks.append(to_constant_float(rmask))
                imask = (R == unique[i]) * ylm_i(l,m,theta,phi)
                imasks.append(to_constant_float(imask))
    rmasks = tf.pack(rmasks, axis=-1)
    rmasks = tf.reshape(rmasks, [k, k, k, n_rings - (order > 0)])
    imasks = tf.pack(imasks, axis=-1)
    imasks = tf.reshape(imasks, [k, k, k, n_rings - (order > 0)])
    return rmasks, imasks


##### CREATING VARIABLES #####
def to_constant_float(Q):
    """Converts a numpy tensor to a tf constant float

    Q: numpy tensor
    """
    Q = tf.Variable(Q, trainable=False)
    return tf.to_float(Q)


def get_weights(filter_shape, W_init=None, std_mult=0.4, name='W', device='/cpu:0'):
    """Initialize weights variable with He method

    filter_shape: list of filter dimensions
    W_init: numpy initial values (default None)
    std_mult: multiplier for weight standard deviation (default 0.4)
    name: (default W)
    device: (default /cpu:0)
    """
    with tf.device(device):
        if W_init == None:
            stddev = std_mult * np.sqrt(2.0 / np.prod(filter_shape[:3]))
        return tf.get_variable(name, dtype=tf.float32, shape=filter_shape,
                               initializer=tf.random_normal_initializer(stddev=stddev))