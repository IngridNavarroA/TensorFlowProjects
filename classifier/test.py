"""
	@author: Ingrid Navarro 
	@date:   May 16th, 2019
	@brief:  Testing an image classifier. 
"""
from sklearn.metrics import precision_recall_fscore_support
from utils.logger import info_msg, err_msg, done_msg
from config import *

import tensorflow.compat.v1 as tf
import numpy as np
import operator
import config
import glob 
import cv2
import os

def normalize( img, cfg ):
	img = cv2.resize( img, ( cfg.img_width, cfg.img_height ), 
		                0, 0, cv2.INTER_LINEAR ) 
	img = img.astype( np.float32 )
	return np.multiply( img, 1.0 / 255.0 )

FLAGS = tf.app.flags.FLAGS 
tf.app.flags.DEFINE_string( 'data_path', 'data/cat-dog/test_data/', 
	                          """ Path to images to test the model.""")
tf.app.flags.DEFINE_string( 'model_path', 'model/model-epoch-9', 
	                          """ Path to model to test. """)

tf.app.flags.DEFINE_string( 'net', 'alexnet', 
	                          """Specify network to load: [alexnet | vgg | inception].""" )

tf.app.flags.DEFINE_boolean( 'watch', False, 
	                           """ Path to model to test. """)

def test():
	# Assert correct parameters. 
	assert FLAGS.net.lower() == 'alexnet' or FLAGS.net.lower() == 'inception' \
	    or FLAGS.net.lower() == 'vgg', err_msg( "Network not supported." )

	# Load configuration
	try:
		info_msg( "Loading training configuration..." )
		cfg = Configuration( FLAGS.data_path, FLAGS.net.lower(), True )
		done_msg() 
	except:
		err_msg( "Could not load configuration. Is your data path: {} ?".format( 
			       FLAGS.data_path ) )

	sess = tf.Session()

	# Import graph
	try:
		info_msg( "Loading graph form {}".format( cfg.net_dict[ "meta_file" ] ) )
		saver = tf.train.import_meta_graph( cfg.net_dict[ "meta_file" ] )	
		done_msg()
	except:
		err_msg( "Could not import graph. Is your meta file: {} ?".format(
			       FLAGS.meta_path ) )

	# Restore model
	try:
		info_msg( "Restoring model parameters from {}...".format( FLAGS.model_path) )
		saver.restore( sess, FLAGS.model_path )
		graph = tf.get_default_graph()
		y_hat = graph.get_tensor_by_name( "y_hat:0" )
		x = graph.get_tensor_by_name( "x:0" )
		y_true = graph.get_tensor_by_name( "y:0" )
		done_msg()
	except:
		err_msg( "Could not load model parameters. Is your model path: {} ?".format(
			       FLAGS.model_path ) )

	cv2.namedWindow( "Test" )
	cv2.moveWindow( "Test", 100, 100 ) 

	images = []
	all_files = []
	y_truth = []
	y_predicted = []

	# Prepare data
	info_msg("Preparing testing images...")
	for i, c in enumerate( cfg.classes ):
		path = os.path.join( FLAGS.data_path, c, '*jpg' )
		files = glob.glob( path )
		for f in files:
			all_files.append( f )
			image = cv2.imread( f )
			# Normalize 
			image = cv2.resize( image, ( cfg.img_width, cfg.img_height ), 
				                  0, 0, cv2.INTER_LINEAR ) 
			image = np.multiply( image.astype( np.float32 ), 1.0 / 255.0 )
			images.append( image )
			y_truth.append( i )
	done_msg()

	x_test = np.array( images )
	num_samples = len( images )
	y_test = np.zeros( ( num_samples, cfg.num_classes ) )

	feed_dict_testing = { x: x_test, y_true: y_test }
	result = sess.run( y_hat, feed_dict=feed_dict_testing )

	# Test data 
	for i, r in enumerate( result ):
		index, value = max( enumerate(r), key=operator.itemgetter(1) )
		y_predicted.append( index )

		if FLAGS.watch:
			img = x_test[i]
			cv2.putText( img, "Prediction: ({}, {:.2f})".format( cfg.classes[ index ], 
				           value ), (10, cfg.img_height-10), cv2.FONT_HERSHEY_SIMPLEX, 
			             0.5, (0,0,255), 1, cv2.LINE_AA )
			print( i, all_files[ i ], cfg.classes[ index ], cfg.classes[ y_truth[ i ] ] )
			img_res = cv2.resize( img, ( int( cfg.img_width * 1.5 ), 
				                           int( cfg.img_height * 1.5 ) ) )
			cv2.imshow("Test", img)
			key = cv2.waitKey(100)

			# if key == ord('n'):
			# 	continue

			if key & 0xFF == ord( 'q' ):
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

	cv2.destroyAllWindows()


def main(argv=None):
	test()

if __name__ == '__main__':
	tf.app.run()