"""
	@author: IngridNavarroA
	@date:   June 20th, 2019
	@brief:  Network configuration. Supports Alexnet, VGG16, ResNet, Inception and SqueezeNet.
"""
import os
from easydict import EasyDict as edict

def base_config(data_path, network):
	""" 
		Defines the parameters to train / test the CNN. 
		Parameters
		----------
			data_path : str -> path to the training data. 
			network   : str -> network to train.  
	"""
	config = edict()
	
	config.DATA_PATH = data_path
	config.CLASSES   = os.listdir(config.DATA_PATH)
	config.NUM_CLS   = len(config.CLASSES)
	config.VAL_SIZE  = 0.20 
	config.SCALE     = 1.0
	config.NUM_CHANNELS = 3
	config.RESTORE_THRESHOLD = 0.90 

	# Configuration specific to type of neural network. These are my preferred parameters.
	if network == "alexnet":
		config.KEEP_PROB     = 0.20 # dropout probability
		config.BATCH_SIZE    = 32
		config.LEARNING_RATE = 1e-4
		config.IMG_WIDTH     = 256
		config.IMG_HEIGHT    = 256
		config.TRAIN_LAYERS  = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4']
		config.META_FILE     = './pretrained/alexnet/alexnet.meta'
		config.WEIGHTS       = './pretrained/alexnet/alexnet.npy'

	elif network == 'vgg':
		config.KEEP_PROB     = 0.40 
		config.BATCH_SIZE    = 32
		config.LEARNING_RATE = 1e-4
		config.IMG_WIDTH     = 128
		config.IMG_HEIGHT    = 128
		config.TRAIN_LAYERS  = ['fc8', 'fc7', 'fc6', 'conv5_3', 'conv5_2', 'conv5_1', 'conv4_3', 'conv4_2', 'conv4_1']
		config.META_FILE     = './pretrained/vgg16/vgg16.meta'
		config.WEIGHTS       = './pretrained/vgg16/vgg16_weights.npz'

	elif network == 'resnet':
		config.KEEP_PROB     = None
		config.BATCH_SIZE    = None
		config.LEARNING_RATE = None
		config.IMG_WIDTH     = None
		config.IMG_HEIGHT    = None
		config.TRAIN_LAYERS  = None
		config.META_FILE     = None
		config.WEIGHTS       = None

	elif network == 'inception':
		config.KEEP_PROB     = None
		config.BATCH_SIZE    = None
		config.LEARNING_RATE = None
		config.IMG_WIDTH     = None
		config.IMG_HEIGHT    = None
		config.TRAIN_LAYERS  = None
		config.META_FILE     = None
		config.WEIGHTS       = None

	elif network == 'squeezenet':
		config.KEEP_PROB     = None
		config.BATCH_SIZE    = None
		config.LEARNING_RATE = None
		config.IMG_WIDTH     = None
		config.IMG_HEIGHT    = None
		config.TRAIN_LAYERS  = None
		config.META_FILE     = None
		config.WEIGHTS       = None

	return config