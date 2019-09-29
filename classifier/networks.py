"""
	@author: IngridNavarroA
	@date:   June 11th, 2019
	@brief:  Network architectures.
"""
import layers as L
import tensorflow as tf
import numpy as np
from utils.logger import watch_msg

class Alexnet():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		conv1 = L.conv('conv1', x=x, fsize=11, nfilters=96, stride=4, padding='SAME') #default padding='SAME'
		pool1 = L.maxpool('pool1', x=conv1, padding='SAME') # default fsize=3, stride=2
		norm1 = L.lrn(x=pool1, radius=2, alpha=2e-05, beta=0.75)

		conv2 = L.conv('conv2', x=norm1, fsize=5, nfilters=256, stride=1, groups=2)
		pool2 = L.maxpool('pool2', x=conv2, padding='SAME')
		norm2 = L.lrn(x=pool2, radius=2, alpha=2e-05, beta=0.75)
		
		conv3 = L.conv('conv3', x=norm2, fsize=3, nfilters=384, stride=1)
		conv4 = L.conv('conv4', x=conv3, fsize=3, nfilters=384, stride=1, groups=2)
		conv5 = L.conv('conv5', x=conv4, fsize=3, nfilters=256, stride=1, groups=2)
		pool5 = L.maxpool('pool5', x=conv5, padding='SAME') 

		flat  = L.flatten(x=pool5)
		fc6  = L.fc('fc6', x=flat, noutputs=4096)
		if self.cfg.is_training:
			fc6  = L.dropout(x=fc6, keep_prob=self.cfg.dropout_rate)
		
		fc7  = L.fc('fc7', x=fc6, noutputs=4096)
		if self.cfg.is_training:
			fc7  = L.dropout(x=fc7, keep_prob=self.cfg.dropout_rate)

		return L.fc('fc8', x=fc7, noutputs=self.cfg.num_classes, relu=False)

	def load_weights(self, sess):
		weights = np.load( self.cfg.net_dict[ "weights" ], encoding='bytes').item()
		for layer in weights:
			# Skip layers that will be trained
			if layer in self.cfg.net_dict[ "train_layers" ]:
				continue

			# Load pre-trained weights on layers that won't be trained
			with tf.compat.v1.variable_scope(layer, reuse=True):
				watch_msg("\tLoading parameters for layer {}".format(layer))
				for data in weights[layer]:
					if len(data.shape) == 1:
						var = tf.get_variable('biases', trainable=False)
					else:
						var = tf.get_variable('weights', trainable=False)
					sess.run(var.assign(data))

class VGG16():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		conv1_1 = L.conv('conv1_1', x=x, fsize=3, nfilters=64) #default padding='SAME', stride=1
		conv1_2 = L.conv('conv1_2', x=conv1_1, fsize=3, nfilters=64)
		pool1   = L.maxpool('pool1', x=conv1_2, fsize=2, stride=2)

		conv2_1 = L.conv('conv2_1', x=pool1, fsize=3, nfilters=128)
		conv2_2 = L.conv('conv2_2', x=conv2_1, fsize=3, nfilters=128)
		pool2   = L.maxpool('pool2', x=conv2_2, fsize=2, stride=2)

		conv3_1 = L.conv('conv3_1', x=pool2, fsize=3, nfilters=256)
		conv3_2 = L.conv('conv3_2', x=conv3_1, fsize=3, nfilters=256)
		conv3_3 = L.conv('conv3_3', x=conv3_2, fsize=3, nfilters=256)
		pool3   = L.maxpool('pool3', x=conv3_3, fsize=2, stride=2)
		
		conv4_1 = L.conv('conv4_1', x=pool3, fsize=3, nfilters=512)
		conv4_2 = L.conv('conv4_2', x=conv4_1, fsize=3, nfilters=512)
		conv4_3 = L.conv('conv4_3', x=conv4_2, fsize=3, nfilters=512)
		pool4   = L.maxpool('pool4', x=conv4_3, fsize=2, stride=2)

		conv5_1 = L.conv('conv5_1', x=pool4, fsize=3, nfilters=512)
		conv5_2 = L.conv('conv5_2', x=conv5_1, fsize=3, nfilters=512)
		conv5_3 = L.conv('conv5_3', x=conv5_2, fsize=3, nfilters=512)
		pool5   = L.maxpool('pool5', x=conv5_3, fsize=2, stride=2)

		# FIX THIS DESIGN 
		flat  = L.flatten(x=pool5)
		fc6   = L.fc('fc6', x=flat, noutputs=4096, binit_val=1.0)
		if self.cfg.is_training:
			fc6  = L.dropout(x=fc6, keep_prob=self.cfg.dropout_rate)

		fc7   = L.fc('fc7', x=fc6, noutputs=4096, binit_val=1.0)
		if self.cfg.is_training:
			fc7  = L.dropout(x=fc7, keep_prob=self.cfg.dropout_rate)

		return L.fc('fc8', x=fc7, noutputs=self.cfg.num_classes, relu=False)

	def load_weights(self, sess):
		weights = np.load( self.cfg.net_dict[ "weights" ] )

		for i, layer_name in enumerate(weights.keys()):
			name_split = layer_name.split('_') # split w, b from name
			layer = '_'.join(name_split[:-1])
			
			if layer in self.cfg.net_dict[ "train_layers" ]:
				continue

			with tf.variable_scope(layer, reuse=True):
				watch_msg("\tLoading parameters for layer {}".format(layer))
				if name_split[-1] == 'W':
					var = tf.get_variable('weights', trainable=False)
				elif name_split[-1] == 'b':
					var = tf.get_variable('biases', trainable=False)
				sess.run(var.assign(weights[layer_name]))

class Inception():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		conv1 = L.conv('conv1', x=x, fsize=7, nfilters=64, stride=2 )
		pool1 = L.maxpool('pool1', x=conv1)
		conv2 = L.conv('conv2', x=pool1, fsize=3, nfilters=192, stride=1 )
		pool2 = L.maxpool('pool2', x=conv2)
		inception3a = L.inception('inception3a', x=pool2, 
		 	conv1_size=64, conv3_red_size=96, conv3_size=128, 
		 	conv5_red_size=16, conv5_size=32, pool_proj_size=32)
		inception3b = L.inception('inception3b', x=inception3a, 
			conv1_size=128, conv3_red_size=128, conv3_size=192, 
			conv5_red_size=32, conv5_size=96, pool_proj_size=64)
		pool3 = L.maxpool('pool2', x=inception3b)
		inception4a = L.inception('inception4a', x=pool3, 
			conv1_size=192, conv3_red_size=96, conv3_size=208, 
			conv5_red_size=16, conv5_size=48, pool_proj_size=64)
		inception4b = L.inception('inception4b', x=inception4a, 
			conv1_size=160, conv3_red_size=112, conv3_size=224, 
			conv5_red_size=24, conv5_size=64, pool_proj_size=64)
		inception4c = L.inception('inception4c', x=inception4b, 
			conv1_size=128, conv3_red_size=128, conv3_size=256, 
			conv5_red_size=24, conv5_size=64, pool_proj_size=64)
		inception4d = L.inception('inception4d', x=inception4c, 
			conv1_size=112, conv3_red_size=144, conv3_size=288, 
			conv5_red_size=32, conv5_size=64, pool_proj_size=64)
		inception4e = L.inception('inception4e', x=inception4d, 
			conv1_size=256, conv3_red_size=160, conv3_size=320, 
			conv5_red_size=32, conv5_size=128, pool_proj_size=128)
		pool4 = L.maxpool('pool4', x=inception4e)
		inception5a = L.inception('inception5a', x=pool4, 
			conv1_size=256, conv3_red_size=160, conv3_size=320, 
			conv5_red_size=32, conv5_size=128, pool_proj_size=128)
		inception5b = L.inception('inception5b', x=inception5a, 
			conv1_size=384, conv3_red_size=192, conv3_size=384, 
			conv5_red_size=48, conv5_size=128, pool_proj_size=128)
		pool5 = L.avgpool('pool5', x=inception5b, fsize=7, stride=1, padding='VALID')
		if self.cfg.is_training:
			pool5  = L.dropout(x=pool5, keep_prob=self.cfg.dropout_rate)
		flat  = L.flatten(x=pool5)
		return L.fc('fc4', x=flat, noutputs=self.cfg.num_classes, relu=False)

	def load_weights(self, weights_path, train_layer, sess):
		pass 