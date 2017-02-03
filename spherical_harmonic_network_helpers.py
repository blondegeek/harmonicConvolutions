import os
import sys
import time

import numpy as np
import tensorflow as tf
import scipy as sp
from spherical_harmonic_network_ops import get_weights

"""
Helper functions for spherical harmonic (3D) network.
"""

def get_weights_dict(shape, max_order, std_mult=0.4, name='W', device='/cpu:0'):
    """Return a dict of weights.

    This function is identical to get_weights_dict in harmonic_network_helpers.py
    except n_rings and sh is adjusted for 3D. Note, the radial function for a given order (l)
    is the same for all m's.

    shape: list of filter shape [h,w,d,i,o] --- note we use h=w=d
    max_order: returns weights for m=0,1,...,max_order
    std_mult: He init scaled by std_mult (default 0.4)
    name: (default 'W')
    dev: (default /cpu:0)
    """
    weights_dict = {}
    radius = (shape[0 ] +1 ) /2
    n_rings = ( radius * (radius + 1) * (radius + 2) ) / 6
    for i in xrange( max_order +1):
        sh = [ n_rings -( i >0)] + shape[3:]
        nm = name + '_' + str(i)
        weights_dict[i] = get_weights(sh, std_mult=std_mult, name=nm, device=device)
    return weights_dict

def get_bias_dict(n_filters, order, name='b', device='/cpu:0'):
	"""Return a dict of biases"""
	with tf.device(device):
		bias_dict = {}
		for i in xrange(order+1):
			bias = tf.get_variable(name+'_'+str(i), dtype=tf.float32,
								   shape=[n_filters],
				initializer=tf.constant_initializer(1e-2))
			bias_dict[i] = bias
	return bias_dict


def get_phase_dict(n_in, n_out, order, name='b',device='/cpu:0'):
	"""Return a dict of phase offsets"""
	# NEED THREE PHASES IN 3D
	with tf.device(device):
		phase_dict = {}
		for i in xrange(order+1):
			init = np.random.rand(1,1,1,n_in,n_out) * 2. *np.pi
			init = np.float32(init)
			phase = tf.get_variable(name+'_'+str(i), dtype=tf.float32,
									shape=[1,1,1,n_in,n_out],
				initializer=tf.constant_initializer(init))
			phase_dict[i] = phase
	return phase_dict

