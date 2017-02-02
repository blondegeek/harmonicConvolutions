import numpy as np
import tensorflow as tf
from sympy.physics.quantum.cg import CG


order_lm_dict = {0: (0, 0),
                 1: (1, -1),
                 2: (1, 0),
                 3: (1, 1),
                 4: (2, -2),
                 5: (2, -1),
                 6: (2, 0),
                 7: (2, 1),
                 8: (2, 2)}

def conv(X, W, strides, padding, name):
    """Shorthand for tf.nn.conv3d"""
    return tf.nn.conv3d(X, W, strides=strides, padding=padding, name=name)


def h_conv(X, W, strides=(1, 1, 1, 1), padding='VALID', max_order=1, name='N'):
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
                    cg = cg.doit()
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
