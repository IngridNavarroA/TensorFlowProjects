"""
	@author: Ingrid Navarro
	@date:   Dec 17th, 2018
	@brief:  Network configuration
"""
import os

class Configuration():
	def	__init__(self, data_path, network, training):

		self.data_path   = data_path
		self.classes     = os.listdir(self.data_path)
		self.num_classes = len(self.classes)
		self.is_training = training # defines if training or testing

		# Image configuration
		self.img_scale  = 1.0
		self.img_depth  = 3
		
		# Training configuration 
		self.split_size = 0.15
		self.batch_size = 32
		self.dropout_rate = 0.5
		
		self.adam_momentum = 0.5
		self.save_each_n = 20
		self.img_width  = 224
		self.img_height = 224

		if network == "alexnet":
			self.learning_rate = 1e-2
			self.img_width  = 227
			self.img_height = 227
			self.batch_size = 128
			self.net_dict      = {
				"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5', 'conv4'],
				"meta_file"	   : './pretrained/alexnet/alexnet.meta',
				"weights"	     : './pretrained/alexnet/alexnet.npy'
			}

		# This is VGG16
		elif network == "vgg":
			self.learning_rate = 1e-4
			self.net_dict = {
				"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5_3', 'conv5_2', 'conv5_1'],
				"meta_file"    : './pretrained/vgg16/vgg16.meta',
				"weights"	     : './pretrained/vgg16/vgg16_weights.npz'
			}

		# This is Inception v4
		elif network == "inception":
			self.img_width  = 299
			self.img_height = 299
			self.learning_rate = 1e-4
			self.dropout_rate = 0.8
			self.net_dict = {
				"train_layers" : [],
				"meta_file"    : './pretrained/inception/inception.meta',
				"weights"	     : './pretrained/inception/inception.npz'
			}