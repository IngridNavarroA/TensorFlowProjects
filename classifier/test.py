"""
	@author: Ingrid Navarro 
	@date:   May 16th, 2019
	@brief:  Testing an image classifier. 
"""
from sklearn.metrics import precision_recall_fscore_support
from utils import info_msg, err_msg, done_msg
import tensorflow as tf
import numpy as np
import operator
import config
import glob 
import cv2
import os

def normalize(img):
	img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT), 0, 0, cv2.INTER_LINEAR)
	img = img.astype(np.float32)
	return np.multiply(img, 1.0 / 255.0)

FLAGS = tf.app.flags.FLAGS 
tf.app.flags.DEFINE_string('data', 'data/cat-dogs/testing_data/', """ Path to images to test the model.""")
tf.app.flags.DEFINE_string('model', 'model/model-epoch-98', """ Path to model to test. """)
tf.app.flags.DEFINE_string('meta', 'pretrained/alexnet/alexnet.meta', """ Path to model to test. """)
tf.app.flags.DEFINE_integer('watch', 0, """ Path to model to test. """)
tf.app.flags.DEFINE_integer('video', 0, """ Testing with video or images. """)

sess = tf.Session()

# Define test()
# Import graph
try:
	info_msg("Loading graph form {}".format(FLAGS.meta))
	saver = tf.train.import_meta_graph(FLAGS.meta)
	done_msg()
except:
	err_msg("Could not import graph. Is your meta file: {} ?".format(FLAGS.meta))

# Load base configuration 
try:
	info_msg("Loading configuration...")
	cfg = config.base_config(FLAGS.data)
	done_msg() 
except:
	err_msg("Could not load configuration. Is your data path: {} ?".format(FLAGS.data_path))

# Restore model

info_msg("Restoring model parameters from {}...".format(FLAGS.model))
saver.restore(sess, FLAGS.model)
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y:0")
done_msg()
#except:
#	err_msg("Could not load model parameters. Is your model path: {} ?".format(FLAGS.model))
cv2.namedWindow("Test")
cv2.moveWindow("Test", 100, 100)

images = []
all_files = []
y_truth = []
y_predicted = []
if not FLAGS.video:
	# Prepare data
	info_msg("Preparing testing images...")
	for i, c in enumerate(cfg.CLASSES):
		path = os.path.join(FLAGS.data, c, '*jpg')
		files = glob.glob(path)
		for f in files:
			all_files.append(f)
			image = cv2.imread(f)
			image = normalize(image)
			images.append(image)
			y_truth.append(i)
	done_msg()

	x_test = np.array(images)
	num_samples = len(images)
	y_test = np.zeros((num_samples, cfg.NUM_CLS))

	feed_dict_testing = {x: x_test, y_true: y_test}
	result = sess.run(y_pred, feed_dict=feed_dict_testing)

	# Test data 
	for i, r in enumerate(result):
		index, value = max(enumerate(r), key=operator.itemgetter(1))
		y_predicted.append(index)

		if FLAGS.watch:
			img = x_test[i]
			cv2.putText(img, "Prediction: ({}, {:.2f}) Real: ({})".format(cfg.CLASSES[index], value, cfg.CLASSES[y_truth[i]]), 
				(10, cfg.IMG_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 1, cv2.LINE_AA)
			print(i, all_files[i], cfg.CLASSES[index], cfg.CLASSES[y_truth[i]])
			img_res = cv2.resize(img, (int(cfg.IMG_WIDTH*1.5), int(cfg.IMG_HEIGHT*1.5)))
			
			cv2.imshow("Test", img_res)
		
			key = cv2.waitKey(1500)

			# if key == ord('n'):
			# 	continue

			if key & 0xFF == ord('q'):
			 	cv2.destroyAllWindows()
			 	exit()
			else:
				continue

	info_msg(""" Computing statistics. """)
	prec, rec, fscore, _ = precision_recall_fscore_support(y_truth, y_predicted, labels=[0,1])
	for i, clss in enumerate(cfg.CLASSES):
		print("Class -- {}".format(clss))
		print("\tPrecission = {}".format(prec[i]))
		print("\tRecall = {}".format(rec[i]))
		print("\tF1-score = {}".format(fscore[i]))

	done_msg()
else:
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()
		cv2.imshow('Live Feed', frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): # quit
			break
		elif key == ord('s'): # snapshot
			images.append(normalize(frame))
			result = sess.run(y_pred, feed_dict={x: np.array(images), y_true: np.zeros((len(images), cfg.NUM_CLS))})

			for i, r in enumerate(result):
				pass

			#print(result)

	cap.release()
	cv2.destroyAllWindows()


	# Get image from video stream
	# Classify image
	# Repeat or quit