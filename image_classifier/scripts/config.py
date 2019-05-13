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
	config.SCALE 		= 0.5
	config.IMG_WIDTH    = 256 # 768
	config.IMG_HEIGHT   = 256

	# Training config
	config.NUM_EPOCHS = 10
	config.BATCH_SIZE = 16
	config.VAL_SIZE   = 0.20  
	config.KEEP_PROB  = 0.5

	config.LEARNING_RATE = 1e-4
	config.TRAIN_LAYERS  = ['fc8', 'fc7', 'fc6'] # to finetune 768x256
	config.WEIGHTS_ALEXNET = 'pretrained/bvlc_alexnet.npy'
	config.WEIGHTS_VGG16   = 'pretrained/vgg16_weights.npz'

	return config