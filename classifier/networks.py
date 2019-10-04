"""
	@author: IngridNavarroA
	@date:   June 11th, 2019
	@brief:  Network architectures.
"""
import tensorflow as tf
import modules as md
import numpy as np
from utils.logger import watch_msg

class Alexnet():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		""" Builds model """
		conv1 = md.conv('conv1', x=x, fsize=[11,11], nfilters=96, stride=4, padding='VALID') #default padding='SAME'
		pool1 = md.maxpool('pool1', x=conv1, padding='VALID') # default fsize=3, stride=2
		norm1 = md.lrn(x=pool1, radius=2, alpha=2e-05, beta=0.75)

		conv2 = md.conv('conv2', x=norm1, fsize=[5,5], nfilters=256, stride=1, groups=2)
		pool2 = md.maxpool('pool2', x=conv2, padding='VALID')
		norm2 = md.lrn(x=pool2, radius=2, alpha=2e-05, beta=0.75)
		
		conv3 = md.conv('conv3', x=norm2, fsize=[3,3], nfilters=384, stride=1)
		conv4 = md.conv('conv4', x=conv3, fsize=[3,3], nfilters=384, stride=1, groups=2)
		conv5 = md.conv('conv5', x=conv4, fsize=[3,3], nfilters=256, stride=1, groups=2)
		pool5 = md.maxpool('pool5', x=conv5, padding='VALID') 

		flat  = md.flatten(x=pool5)
		fc6  = md.fc('fc6', x=flat, noutputs=4096)
		if self.cfg.is_training:
			fc6  = md.dropout(x=fc6, keep_prob=self.cfg.dropout_rate)
		
		fc7  = md.fc('fc7', x=fc6, noutputs=4096)
		if self.cfg.is_training:
			fc7  = tf.nn.dropout(fc7, rate=1-self.cfg.dropout_rate)

		return md.fc('fc8', x=fc7, noutputs=self.cfg.num_classes, relu=False)

	def load_weights(self, sess):
		""" Loads weights from specified layers (cfg) from a pre-trained network. """
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
	
	def lr_decay(self, lr ):
		""" Learning rate decay """
		return np.float32( lr / 10.0 )

class VGG16():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		""" Builds model """
		conv1_1 = md.conv('conv1_1', x=x, fsize=[3,3], nfilters=64) #default padding='SAME', stride=1
		conv1_2 = md.conv('conv1_2', x=conv1_1, fsize=[3,3], nfilters=64)
		pool1   = md.maxpool('pool1', x=conv1_2, fsize=2, stride=2)

		conv2_1 = md.conv('conv2_1', x=pool1, fsize=[3,3], nfilters=128)
		conv2_2 = md.conv('conv2_2', x=conv2_1, fsize=[3,3], nfilters=128)
		pool2   = md.maxpool('pool2', x=conv2_2, fsize=2, stride=2)

		conv3_1 = md.conv('conv3_1', x=pool2, fsize=[3,3], nfilters=256)
		conv3_2 = md.conv('conv3_2', x=conv3_1, fsize=[3,3], nfilters=256)
		conv3_3 = md.conv('conv3_3', x=conv3_2, fsize=[3,3], nfilters=256)
		pool3   = md.maxpool('pool3', x=conv3_3, fsize=2, stride=2)
		
		conv4_1 = md.conv('conv4_1', x=pool3, fsize=[3,3], nfilters=512)
		conv4_2 = md.conv('conv4_2', x=conv4_1, fsize=[3,3], nfilters=512)
		conv4_3 = md.conv('conv4_3', x=conv4_2, fsize=[3,3], nfilters=512)
		pool4   = md.maxpool('pool4', x=conv4_3, fsize=2, stride=2)

		conv5_1 = md.conv('conv5_1', x=pool4, fsize=[3,3], nfilters=512)
		conv5_2 = md.conv('conv5_2', x=conv5_1, fsize=[3,3], nfilters=512)
		conv5_3 = md.conv('conv5_3', x=conv5_2, fsize=[3,3], nfilters=512)
		pool5   = md.maxpool('pool5', x=conv5_3, fsize=2, stride=2)

		# FIX THIS DESIGN 
		flat  = md.flatten(x=pool5)
		fc6   = md.fc('fc6', x=flat, noutputs=4096, binit_val=1.0)
		if self.cfg.is_training:
			fc6  = tf.nn.dropout(fc6, rate=1-self.cfg.dropout_rate)

		fc7   = md.fc('fc7', x=fc6, noutputs=4096, binit_val=1.0)
		if self.cfg.is_training:
			fc7  = tf.nn.dropout(fc7, rate=1-self.cfg.dropout_rate)

		return md.fc('fc8', x=fc7, noutputs=self.cfg.num_classes, relu=False)

	def load_weights(self, sess):
		""" Loads weights from specified layers (cfg) from a pre-trained network. """
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

	def lr(self, lr ):
		return np.float32( lr / 10.0 )

class InceptionV4():
	def __init__(self, cfg):
		self.cfg = cfg

	def load_net(self, x):
		stem = md.stem('stem', x=x)

		# Inception A
		in1a = md.inceptionA('inceptionA_1', x=stem)
		in2a = md.inceptionA('inceptionA_2', x=in1a)
		in3a = md.inceptionA('inceptionA_3', x=in2a) 
		in4a = md.inceptionA('inceptionA_4', x=in3a) 
		
		# Reduction A
		reda = md.reductionA('reductionA', x=in4a)

		# Inception B
		in1b = md.inceptionB('inceptionB_1', x=reda)
		in2b = md.inceptionB('inceptionB_2', x=in1b)
		in3b = md.inceptionB('inceptionB_3', x=in2b)
		in4b = md.inceptionB('inceptionB_4', x=in3b)
		in5b = md.inceptionB('inceptionB_5', x=in4b)
		in6b = md.inceptionB('inceptionB_6', x=in5b)
		in7b = md.inceptionB('inceptionB_7', x=in6b)

		# Reduction B
		redb = md.reductionB('reductionB', x=in7b)

		# Inception C
		in1c = md.inceptionC('inceptionC_1', x=redb)
		in2c = md.inceptionC('inceptionC_2', x=in1c)
		in3c = md.inceptionC('inceptionC_3', x=in2c)

		# Average Pooling 
		pool = md.avgpool('avgpool', x=in3c, fsize=3, stride=1)

		# Dropout 
		if self.cfg.is_training:
			pool = tf.nn.dropout(pool, rate=1-self.cfg.dropout_rate)

		flat  = md.flatten(x=pool)
		return md.fc('fc4', x=flat, noutputs=self.cfg.num_classes, relu=False)

	def lr(self, lr ):
		return np.float32( lr / 10.0 )