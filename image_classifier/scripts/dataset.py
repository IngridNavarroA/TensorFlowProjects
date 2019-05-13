import os
import glob
import cv2
import numpy as np
from sklearn.utils import shuffle

class Dataset():
	def __init__(self, images, labels, names, cls):
		self.num_examples = len(images)
		self.images = images
		self.labels = labels
		self.im_names = names
		self.cls = cls 
		self.epochs_done = 0
		self.idx_epoch = 0

	def next_batch(self, batch_size):
		""" Gets next batch from dataset. """
		start = self.idx_epoch
		self.idx_epoch += batch_size
		if self.idx_epoch > self.num_examples:
			self.epochs_done += 1
			start = 0
			self.idx_epoch = batch_size
			assert batch_size <= self.num_examples
		end = self.idx_epoch
		return self.images[start:end], self.labels[start:end], self.im_names[start:end], self.cls[start:end]

def load(cfg):
	""" Loads and normalizes dataset from specified path. """
	def normalize(f):
		img = cv2.imread(f)
		img = cv2.resize(img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), 0, 0, cv2.INTER_LINEAR)
		img = img.astype(np.float32)
		return np.multiply(img, 1.0 / 255.0)

	# Get dataset 
	dataset = [] # image, label, name, class
	for l, c in enumerate(cfg.CLASSES):
		try:
			path = os.path.join(cfg.DATA_PATH, c, '*jpg')
			files = glob.glob(path)
			
			print("[INFO] Leyendo ({}:{}) de: {}".format(l, c, path))
			for f in files:
				img = normalize(f)
				fname = os.path.basename(f)
				dataset.append([img, fname, l, c])
		except:
			return None

	dataset = np.array(dataset)
	dataset = shuffle(dataset)

	val_size = int(cfg.VAL_SIZE * len(dataset))

	# Split dataset
	vimages = dataset[0][:val_size]
	vfnames = dataset[1][:val_size]
	vlabels = dataset[2][:val_size]
	vclass  = dataset[3][:val_size]

	timages = dataset[0][val_size:]
	tfnames = dataset[1][val_size:]
	tlabels = dataset[2][val_size:]
	tclass  = dataset[3][val_size:]

	# Create datasets
	train_set = Dataset(timages, tlabels, tfnames, tclass)
	val_set = Dataset(vimages, vlabels, vfnames, vclass)
	return train_set, val_set