"""
	@author: Ingrid Navarro 
	@date:   May 10th, 2019
	@brief:  Layer and module wrappers 
"""
import tensorflow.compat.v1 as tf
from utils.logger import watch_msg

def variable_summary(var):
	""" Wrapper that adds summary to Tensor variable to be visualized 
		with TensorBoard. """
	
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		
		# Visualizing gradient  
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		
		# Visualizing activation distribution
		tf.summary.histogram('histogram', var)

def conv(name, x, fsize, nfilters, stride=1, padding='SAME', groups=1, stddev=0.05, binit_val=0.05):
	""" Wrapper for Convolutional layer. """

	ninputs = int(x.get_shape()[-1].value / groups)
	convolve = lambda i, w: tf.nn.conv2d(i, w, 
																strides=[1, stride, stride, 1], padding=padding)

	with tf.variable_scope(name) as scope:

		# Create random weights 
		w_init = tf.random.truncated_normal(shape=[fsize[0], fsize[1], ninputs, nfilters], stddev=stddev, dtype=tf.float32)
		w = tf.get_variable('weights', initializer=w_init, dtype=tf.float32)

		# Initialize biases 
		b_init = tf.constant(binit_val, shape=[nfilters], dtype=tf.float32)
		b = tf.get_variable('biases', initializer=b_init, dtype=tf.float32)

		# Convolution 
		if groups == 1:
			layer = convolve(x, w)
		else:
			x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
			w_groups = tf.split(axis=3, num_or_size_splits=groups, value=w)

			out_groups = [convolve(i, k) for i, k in zip(x_groups, w_groups)]
			layer = tf.concat(axis=3, values=out_groups)

		layer += b

		watch_msg("\tLayer {} has shape {}".format(name, layer.shape))
		return tf.nn.relu(layer, name=scope.name)

def lrn(x, radius, alpha, beta, bias=1.0):
	""" Wrapper for Local Response Normalization layer (LRN). """
	return tf.nn.local_response_normalization(x,
		depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def maxpool(name, x, fsize=3, stride=2, padding='SAME'):
	""" Wrapper for Max Pooling layer. """
	with tf.variable_scope(name):
		layer = tf.nn.max_pool(value=x, ksize=[1, fsize, fsize, 1], 
			strides=[1, stride, stride, 1], padding=padding)       
		
		variable_summary(layer)
		watch_msg("\tLayer {} has shape {}".format(name, layer.shape))	
		return layer

def avgpool(name, x, fsize=3, stride=2, padding='SAME'):
	""" Wrapper for Average Pooling layer. """
	with tf.variable_scope(name):
		layer = tf.nn.avg_pool(value=x, ksize=[1, fsize, fsize, 1], 
			strides=[1, stride, stride, 1], padding=padding)       
		
		variable_summary(layer)
		watch_msg("\tLayer {} has shape {}".format(name, layer.shape))	
		return layer

def flatten(x):
	""" Wrapper to convert mutlidimensional tensor into a 
		one-dimension tensor. """
	layer_shape = x.get_shape()
	nfeatures = layer_shape[1:].num_elements()
	watch_msg("\tLayer flatted to {} parameters".format(nfeatures))
	return tf.reshape(x, [-1, nfeatures])

def fc(name, x, noutputs, relu=True, stddev=0.05, binit_val=0.05):
	""" Wrapper for Fully Connected layer. """ 
	
	layer_shape = x.get_shape()
	ninputs = layer_shape[1:].num_elements()

	with tf.variable_scope(name) as scope:
		# Create random weights
		w_init = tf.random.truncated_normal(shape=[ninputs, noutputs], stddev=stddev, dtype=tf.float32)
		w = tf.get_variable('weights', initializer=w_init, dtype=tf.float32)

		# Initialize bias 
		b_init = tf.constant(binit_val, shape=[noutputs], dtype=tf.float32)
		b = tf.get_variable('biases', initializer=b_init, dtype=tf.float32)

		layer = tf.nn.xw_plus_b(x, w, b, name=scope.name)

		if relu:
			layer = tf.nn.relu(layer)
		watch_msg("\tLayer {} has shape {}".format(name, layer.shape))
		return layer

def inception(name, x, conv1_size, conv3_red_size, conv3_size, conv5_red_size, conv5_size, pool_proj_size):
	""" Inception module. """
	with tf.variable_scope(name) as scope:
		
		watch_msg("\n\tModule {}".format(name))
		conv1 = conv('{}_1x1'.format(name), x=x, fsize=[1,1], nfilters=conv1_size)
		
		conv3_red = conv('{}_3x3_red'.format(name), x=x, fsize=[1,1], nfilters=conv3_red_size)
		conv3 = conv('{}_3x3'.format(name), x=conv3_red, fsize=[3,3], nfilters=conv3_size)
		
		conv5_red = conv('{}_5x5_red'.format(name), x=x, fsize=[1,1], nfilters=conv5_red_size)
		conv5 = conv('{}_5x5'.format(name), x=conv5_red, fsize=[5,5], nfilters=conv5_size)
		
		pool = maxpool('{}_pool'.format(name), x=x, fsize=[3,3], stride=1)
		pool_proj = conv('{}_pool_proj'.format(name), x=pool, fsize=[1,1], nfilters=pool_proj_size)

		concat = tf.concat([conv1, conv3, conv5, pool_proj], axis=3, name='{}_concat'.format(name))
		watch_msg("\tLayer {} has shape {}".format(name, concat.shape))
		return concat

def stem(name, x):
	""" Schema of the stem module for InceptionV4 """
	with tf.variable_scope(name): 
		watch_msg("\n\tModule {}".format(name))
		conv1_1 = conv('{}_conv1_1_3x3'.format(name), x=x, fsize=[3,3], nfilters=32, stride=2, padding='VALID')
		conv1_2 = conv('{}_conv1_2_3x3'.format(name), x=conv1_1, fsize=[3,3], nfilters=32, stride=1, padding='VALID')
		conv1_3 = conv('{}_conv1_3_3x3'.format(name), x=conv1_2, fsize=[3,3], nfilters=64, stride=1)

		pool1_4a = maxpool('{}_mxpool1'.format(name), x=conv1_3, fsize=3, stride=2, padding='VALID')
		conv1_4b = conv('{}_conv1_4_3x3'.format(name), x=conv1_3, fsize=[3,3], nfilters=96, stride=2, padding='VALID')
		concat1 = tf.concat([pool1_4a, conv1_4b], axis=3, name='{}_concat1'.format(name))
		watch_msg("\tFilter concatenation {}_1 has shape {}\n".format(name, concat1.shape))

		conv2_1a = conv('{}_conv2_1a_1_1x1'.format(name), x=concat1, fsize=[1,1], nfilters=64, stride=1)
		conv2_2a = conv('{}_conv2_2a_3x3'.format(name), x=conv2_1a, fsize=[3,3], nfilters=96, stride=1, padding='VALID')

		conv2_1b = conv('{}_conv2_1b_1x1'.format(name), x=concat1, fsize=[1,1], nfilters=64, stride=1)
		conv2_2b = conv('{}_conv2_2b_7x1'.format(name), x=conv2_1b, fsize=[7,1], nfilters=64, stride=1)
		conv2_3b = conv('{}_conv2_3b_1x7'.format(name), x=conv2_2b, fsize=[1,7], nfilters=64, stride=1)
		conv2_4b = conv('{}_conv2_4b_3x3'.format(name), x=conv2_3b, fsize=[3,3], nfilters=96, stride=1, padding='VALID')
		concat2 = tf.concat([conv2_2a, conv2_4b], axis=3, name='{}_concat2'.format(name))
		watch_msg("\tFilter concatenation {}_2 has shape {}\n".format(name, concat2.shape))

		conv3_1a = conv('{}_conv3_1_3x3'.format(name), x=concat2, fsize=[3,3], nfilters=192, stride=2, padding='VALID')
		pool3_1b = maxpool('{}_mxpool3'.format(name), x=concat2, fsize=3, stride=2, padding='VALID')

		concat3 = tf.concat([conv3_1a, pool3_1b], axis=3, name='{}_concat3'.format(name))
		watch_msg("\tFilter concatenation {}_3 has shape {}\n".format(name, concat3.shape))
		return concat3

