"""
	@author: Ingrid Navarro
	@date:   Dec 17th, 2018
	@brief:  Network configuration
"""
import os

class Configuration():
	def	__init__(self, data_path, stage, num_epochs, network):

		self.data_path   = data_path
		self.classes     = os.listdir(self.data_path)
		self.num_classes = len(self.classes)
		self.stage       = stage # defines if training or testing

		# Image configuration
		self.img_scale  = 1.0
		self.img_depth  = 3
		self.img_width  = 256
		self.img_height = 256

		# Training configuration 
		self.split_size = 0.15
		self.batch_size = 32
		self.dropout_rate = 0.5
		
		self.adam_momentum = 0.5
		self.num_epochs = num_epochs
		self.save_each_n = 20
		self.restore = 0.90

		if network == "alexnet":
			self.learning_rate = 1e-4
			self.net_dict = {
				"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'],
				"meta_file"	   : './pretrained/alexnet/alexnet.meta',
				"weights"	     : './pretrained/alexnet/alexnet.npy'
			}
		elif network == "vgg":
			self.learning_rate = 1e-4
			self.net_dict = {
				"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5_3', 'conv5_2', 'conv5_1', 'conv4_3', 'conv4_2', 'conv4_1'],
				"meta_file"    : './pretrained/vgg16/vgg16.meta',
				"weights"	     : './pretrained/vgg16/vgg16_weights.npz'
			}

		elif network == "resnet":
			pass
		elif network == "inception":
			pass
		elif network == "squeezenet":
			pass