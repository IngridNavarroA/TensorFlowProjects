"""
	@author: IngridNavarroA
	@date:   May 10th, 2019
	@brief:  Util to load input dataset, and create and split the data
			 into training an validation sets. 
"""
import os
import glob
import cv2
import numpy as np
from sklearn.utils import shuffle

class Dataset():
	def __init__(self, images, labels, cls):
		self._num_examples = images.shape[0] #len(images)
		self._images = images
		self._labels = labels
		self._cls = cls 
		self._epochs_done = 0
		self._idx_epoch = 0

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def images(self):
		return self._images
	
	@property
	def labels(self):
		return self._labels
	
	@property
	def cls(self):
		return self._cls
	
	def next_batch(self, batch_size):
		""" Gets next batch from dataset. """
		start = self._idx_epoch
		self._idx_epoch += batch_size
		if self._idx_epoch > self._num_examples:
			self._epochs_done += 1
			start = 0
			self._idx_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._idx_epoch
		return self._images[start:end], self._labels[start:end]

def load(cfg, frmt):
	""" Loads and normalizes dataset from specified path. """
	def normalize(f):
		img = cv2.imread(f)
		img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT), 0, 0, cv2.INTER_LINEAR)
		img = img.astype(np.float32)
		return np.multiply(img, 1.0 / 255.0)

	# Get dataset 
	images, labels, classes = [], [], []

	for lbl, clss in enumerate(cfg.CLASSES):
		
		path = os.path.join(cfg.DATA_PATH, clss, '*.{}'.format(frmt))
		files = glob.glob(path)

		print("\t[DATA] Reading class {} (index: {}) from: {}".format(clss, lbl, path))
		for file in files:
			image = normalize(file)
			images.append(image)

			label = np.zeros(len(cfg.CLASSES))
			label[lbl] = 1.0
			labels.append(label)

			classes.append(clss)
		

	images, labels, classes = shuffle(np.array(images), np.array(labels), np.array(classes))

	val_size = int(cfg.VAL_SIZE * images.shape[0])

	# Split dataset
	vimages = images[:val_size]
	vlabels = labels[:val_size]
	vclass  = classes[:val_size]

	timages = images[val_size:]
	tlabels = labels[val_size:]
	tclass  = classes[val_size:]

	# Create datasets
	train_set = Dataset(timages, tlabels, tclass)
	val_set = Dataset(vimages, vlabels, vclass)

	return train_set, val_set