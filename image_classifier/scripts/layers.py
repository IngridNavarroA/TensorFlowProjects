import tensorflow as tf

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

def conv(name, x, filter_size, num_filters, stride, padding='SAME', groups=1, train=False):
	""" Wrapper for Convolutional layer. """
	num_inputs = int(x.get_shape()[-1].value / groups)

	convolve = lambda i, k: tf.nn.conv2d(i, k, 
		strides=[1, stride, stride, 1], padding=padding)

	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', 
			shape=[filter_size, filter_size, num_inputs, num_filters], 
			trainable=train)
		b = tf.get_variable('biases', 
			shape=[num_filters], trainable=train)

		if groups == 1:
			layer = convolve(x, w)
		else:
			x_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
			w_groups = tf.split(axis=3, num_or_size_splits=groups, value=w)

			out_groups = [convolve(i, k) for i, k in zip(x_groups, w_groups)]
			layer = tf.concat(axis=3, values=out_groups)

		layer += b

		print("[SHAPE] Layer {} has shape {}".format(name, layer.shape))
		return tf.nn.relu(layer, name=scope.name)

def lrn(x, radius, alpha, beta, bias=1.0):
	""" Wrapper for Local Response Normalization layer (LRN). """
	return tf.nn.local_response_normalization(x,
		depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def maxpool(name, input, fsize, stride, padding='SAME'):
	""" Wrapper for Pooling layer. """
	with tf.variable_scope(name):
		layer = tf.nn.max_pool(value=input, ksize=[1, fsize, fsize, 1], 
					strides=[1, stride, stride, 1], padding=padding)       
		variable_summary(layer)

		print("[SHAPE] Layer {} has shape {}".format(name, layer.shape))	
		return layer

def flatten(name, x):
	""" Wrapper to convert mutlidimensional tensor into a 
		one-dimension tensor. """
	layer_shape = x.get_shape()
	nfeatures = layer_shape[1:].num_elements()
	print("[SHAPE] Layer flatted to {} features".format(nfeatures))
	return tf.reshape(x, [-1, nfeatures])

def fc(name, x, noutputs, relu=True):
	""" Wrapper for Fully Connected layer. """ 
	layer_shape = x.get_shape()
	ninputs = layer_shape[1:].num_elements()
	with tf.variable_scope(name) as scope:
		w = tf.get_variable('weights', shape=[ninputs, noutputs], trainable=True)
		b = tf.get_variable('biases', shape=[noutputs], trainable=True)

		layer = tf.nn.xw_plus_b(x, w, b, name=scope.name)

		if relu:
			layer = tf.nn.relu(layer)
		print("[SHAPE] Connected layer {} has shape {}".format(name, layer.shape))
		return layer

def dropout(x, keep_prob=0.5):
	""" Wrapper for Dropout layer. """
	return tf.nn.dropout(x, keep_prob)