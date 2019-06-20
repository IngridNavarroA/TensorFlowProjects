"""
	@author: Ingrid Navarro
	@date:   Dec 17th, 2018
	@brief:  Network's base configuration
"""
import os
from easydict import EasyDict as edict

def base_config(data_path):
	""" Defines the parameters to train / test the convnet. 

		Parameters
		----------
			data_path : str
				Path to the training data. 
	"""
	config = edict()
	
	# Dataset 
	config.DATA_PATH = data_path
	config.CLASSES = os.listdir(config.DATA_PATH)
	config.NUM_CLS = len(config.CLASSES)

	# Image config
	config.NUM_CHANNELS = 3
	config.SCALE 		= 1.0
	config.IMG_WIDTH    = 768 # 128 # [128, 224, 256]
	config.IMG_HEIGHT   = 256 # 128 # [128, 224, 256]

	# Training config
	config.NUM_EPOCHS = 10
	config.BATCH_SIZE = 32
	config.VAL_SIZE   = 0.20  
	config.KEEP_PROB  = 0.20
	config.RESTORE    = 0.90 
	config.LEARNING_RATE = 1e-4

	# Alexnet
	config.ALEXNET = {
		"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'],
		"meta_file"	   : './pretrained/alexnet/alexnet.meta',
		"weights"	   : './pretrained/alexnet/alexnet.npy'
	}

	# VGG16
	config.VGG16 = {
		"train_layers" : ['fc8', 'fc7', 'fc6', 'conv5_3', 'conv5_2', 'conv5_1', 'conv4_3', 'conv4_2', 'conv4_1'],
		"meta_file"    : './pretrained/vgg16/vgg16.meta',
		"weights"	   : './pretrained/vgg16/vgg16_weights.npz'
	}
	
	# Inception 
	config.INCEPTION = {
		"train_layers" : None, 
		"meta_file"    : None,
		"weights"	   : None
	}

	return config