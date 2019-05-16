"""
	@author: Ingrid Navarro 
	@date:   May 16th, 2019
	@brief:  Testing an image classifier. 
"""

from utils import info_msg, err_msg, done_msg
import tensorflow as tf
import numpy as np
import operator
import config
import glob 
import cv2
import os

FLAGS = tf.app.flags.FLAGS 

tf.app.flags.DEFINE_string('data', 'data/cat-dogs/testing_data/', """ Path to images to test the model.""")
tf.app.flags.DEFINE_string('model', 'model/model-epoch-124', """ Path to model to test. """)


# Load base configuration 
try:
	info_msg("Loading configuration...")
	cfg = config.base_config(FLAGS.data)
	done_msg() 
except:
	err_msg("Could not load configuration. Is your data path: {} ?".format(FLAGS.data_path))


# Prepare data
info_msg("Preparing testing images...")
images = []
for c in cfg.CLASSES:
	path = os.path.join(FLAGS.data, c, '*jpg')
	files = glob.glob(path)
	for f in files:
		def normalize(f):
			img = cv2.imread(f)
			img = cv2.resize(img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), 0, 0, cv2.INTER_LINEAR)
			img = img.astype(np.float32)
			return np.multiply(img, 1.0 / 255.0)

		image = normalize(f)
		images.append(image)
done_msg()

x_test = np.array(images)

# Import model 
sess = tf.Session()
saver = tf.train.import_meta_graph(FLAGS.model + '.meta')
saver.restore(sess, FLAGS.model)

# Test images
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y:0")
y_test = np.zeros((len(images), cfg.NUM_CLS))

feed_dict_testing = {x: x_test, y_true: y_test}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

for i, r in enumerate(result):
	
	img = x_test[i]
	index, value = max(enumerate(r), key=operator.itemgetter(1))
	
	cv2.putText(img, "Class: ({}, {:.2f})".format(cfg.CLASSES[1-index], value), (10, cfg.IMG_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0), 2, cv2.LINE_AA)
	cv2.imshow("Test {}".format(i), img)


	if cv2.waitKey(0) & 0xFF == ord('n'):
		continue

	if cv2.waitKey(0) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		exit()