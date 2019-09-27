"""
	@author: Ingrid Navarro 
	@date:   May 10th, 2019
	@brief:  Layer wrappers 
"""
import tensorflow as tf

def variable_summary(var):
	""" Wrapper that adds summary to Tensor variable to be visualized 
		with TensorBoard. """
	
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.compat.v1.summary.scalar('mean', mean)
		
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		
		# Visualizing gradient  
		tf.compat.v1.summary.scalar('stddev', stddev)
		tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
		tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
		
		# Visualizing activation distribution
		tf.compat.v1.summary.histogram('histogram', var)

def conv(name, x, fsize, nfilters, stride=1, padding='SAME', groups=1, stddev=0.05, binit_val=0.05):
	""" Wrapper for Convolutional layer. """
	
	ninputs = int(x.get_shape()[-1].value / groups)
	convolve = lambda i, w: tf.nn.conv2d(i, w, 
		strides=[1, stride, stride, 1], padding=padding)

	with tf.compat.v1.variable_scope(name) as scope:

		# Create random weights 
		w_init = tf.random.truncated_normal(shape=[fsize, fsize, ninputs, nfilters], stddev=stddev, dtype=tf.float32)
		w = tf.compat.v1.get_variable('weights', initializer=w_init, dtype=tf.float32)

		# Initialize biases 
		b_init = tf.constant(binit_val, shape=[nfilters], dtype=tf.float32)
		b = tf.compat.v1.get_variable('biases', initializer=b_init, dtype=tf.float32)

		# Convolution 
		if groups == 1:
			layer = convolve(x, w)
		else:
			x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
			w_groups = tf.split(axis=3, num_or_size_splits=groups, value=w)

			out_groups = [convolve(i, k) for i, k in zip(x_groups, w_groups)]
			layer = tf.concat(axis=3, values=out_groups)

		layer += b

		print("\t[LAYER] {} has shape {}".format(name, layer.shape))
		return tf.nn.relu(layer, name=scope.name)

def lrn(x, radius, alpha, beta, bias=1.0):
	""" Wrapper for Local Response Normalization layer (LRN). """
	return tf.nn.local_response_normalization(x,
		depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def maxpool(name, x, fsize=3, stride=2, padding='SAME'):
	""" Wrapper for Pooling layer. """
	with tf.compat.v1.variable_scope(name):
		layer = tf.compat.v1.nn.max_pool(value=x, ksize=[1, fsize, fsize, 1], 
			strides=[1, stride, stride, 1], padding=padding)       
		
		variable_summary(layer)

		print("\t[LAYER] {} has shape {}".format(name, layer.shape))	
		return layer

def flatten(x):
	""" Wrapper to convert mutlidimensional tensor into a 
		one-dimension tensor. """
	layer_shape = x.get_shape()
	nfeatures = layer_shape[1:].num_elements()
	print("\t[LAYER] flatted to {} features".format(nfeatures))
	return tf.reshape(x, [-1, nfeatures])

def fc(name, x, noutputs, relu=True, stddev=0.05, binit_val=0.05):
	""" Wrapper for Fully Connected layer. """ 
	
	layer_shape = x.get_shape()
	ninputs = layer_shape[1:].num_elements()

	with tf.compat.v1.variable_scope(name) as scope:
		# Create random weights
		w_init = tf.random.truncated_normal(shape=[ninputs, noutputs], stddev=stddev, dtype=tf.float32)
		w = tf.compat.v1.get_variable('weights', initializer=w_init, dtype=tf.float32)

		# Initialize bias 
		b_init = tf.constant(binit_val, shape=[noutputs], dtype=tf.float32)
		b = tf.compat.v1.get_variable('biases', initializer=b_init, dtype=tf.float32)

		layer = tf.compat.v1.nn.xw_plus_b(x, w, b, name=scope.name)

		if relu:
			layer = tf.nn.relu(layer)
		print("\t[LAYER] {} has shape {}".format(name, layer.shape))
		return layer

def dropout(x, keep_prob=0.5):
	""" Wrapper for Dropout layer. """
	return tf.nn.dropout(x, rate=1-keep_prob)


def inception(name, x, conv1_size, conv3_red_size, conv3_size, conv5_red_size, conv5_size, pool_proj_size):
	""" Inception module. """
	with tf.compat.v1.variable_scope(name) as scope:
		
		print("\n\t[MODULE] {}".format(name))
		conv1 = conv('{}_1x1'.format(name), x=x, fsize=1, nfilters=conv1_size)
		
		conv3_red = conv('{}_3x3_red'.format(name), x=x, fsize=1, nfilters=conv3_red_size)
		conv3 = conv('{}_3x3'.format(name), x=conv3_red, fsize=3, nfilters=conv3_size)
		
		conv5_red = conv('{}_5x5_red'.format(name), x=x, fsize=1, nfilters=conv5_red_size)
		conv5 = conv('{}_5x5'.format(name), x=conv5_red, fsize=5, nfilters=conv5_size)
		
		pool = maxpool('{}_pool'.format(name), x=x, fsize=1, stride=1, padding='VALID')
		pool_proj = conv('{}_pool_proj'.format(name), x=pool, fsize=1, nfilters=pool_proj_size)

		return tf.concat([conv1, conv3, conv5, pool_proj], axis=3, name='{}_concat'.format(name))

