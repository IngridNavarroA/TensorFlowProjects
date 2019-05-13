import layers as L
import tensorflow as tf
import numpy as np

class Alexnet():
	def __init__(self, nclasses, prob=0.5):
		self.num_classes = nclasses
		self.keep_prob = prob

	def load_net(self, x):
		conv1 = L.conv('conv1', x, 11, 96, 4)
		norm1 = L.lrn(conv1, 2, 2e-05, 0.75)
		pool1 = L.maxpool('pool1', norm1, 3, 2, padding='VALID')

		conv2 = L.conv('conv2', pool1, 5, 256, 1, groups=2)
		norm2 = L.lrn(conv2, 2, 2e-05, 0.75)
		pool2 = L.maxpool('pool2', norm2, 3, 2, padding='VALID')

		conv3 = L.conv('conv3', pool2, 3, 384, 1)
		conv4 = L.conv('conv4', conv3, 3, 384, 1, groups=2)
		conv5 = L.conv('conv5', conv4, 3, 256, 1, groups=2)
		pool5 = L.maxpool('pool5', conv5, 3, 2, padding='VALID') 

		flat  = L.flatten('flat', pool5)
		fcl6  = L.fc('fc6', flat, 4096)
		drp6  = L.dropout(fcl6, self.keep_prob)

		fcl7  = L.fc('fc7', drp6, 4096)
		drp7  = L.dropout(fcl7, self.keep_prob)

		return L.fc('fc8', drp7, self.num_classes, False)

	def load_weights(self, weights_path, train_layer, sess):
		weights = np.load(weights_path, encoding='bytes').item()
		for layer in weights:

			# Skip layers that will be trained
			if layer in train_layer:
				continue

			# Load pre-trained weights on layers that won't be trained
			with tf.variable_scope(layer, reuse=True):
				for data in weights[layer]:

					# biases
					if len(data.shape) == 1:
						var = tf.get_variable('biases', trainable=False)
						sess.run(var.assign(data))
					# weights
					else:
						var = tf.get_variable('weights', trainable=False)
						sess.run(var.assign(data))