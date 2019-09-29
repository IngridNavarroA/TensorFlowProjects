"""
	@author: IngridNavarroA
	@date:   June 13th, 2019
	@brief:  An API to train multiple image classifiers. 
			 Supported neural networks:
			 	- Alexnet
			 	- VGG16
			 	- ResNet (in progress)
			 	- Inception (in progress)
			 	- SqueezeNet (in progress)
			 Types of training supported:
			 	- End-to-end
			 	- Restoring 
			 	- Finetuning 
"""
from utils.logger import * 
from config import *
import utils.dataset as dataset
import networks as cnn
import tensorflow.compat.v1 as tf
import numpy as np
import time as t
import os

np.random.seed( 1 )
tf.set_random_seed( 2 )

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string( 'data_path', './data/cat-dog/training_data/', 
	                          """Full path to training images.""" )
tf.app.flags.DEFINE_string( 'model_path', 'model/', 
	 												  """Full path where to save checkpoints.""" )
tf.app.flags.DEFINE_string( 'logs_path', 'logs/', 
	                          """Full path to save training logs.""" )
tf.app.flags.DEFINE_string( 'net', 'alexnet', 
	                          """Specify network to load: [alexnet | vgg | inception].""" )
tf.app.flags.DEFINE_string( 'img_format', 'jpg', 
	                          """Specify image format [png | jpg]. """ )

tf.app.flags.DEFINE_integer( 'max_iter', 5000, 
	                           """Max number of iterations for training.""" )
tf.app.flags.DEFINE_integer( 'ckpts_to_keep', 15, 
	                           """Max number of checkpoints to keep.""" )
tf.app.flags.DEFINE_integer( 'gpu', 0, 
	                           """ID of GPU to use. """)

tf.app.flags.DEFINE_boolean( 'cpu', False, 
														 """Use CPU (True). """)
tf.app.flags.DEFINE_boolean( 'finetune', False, 
	                           """Finetune (True) or train end-to-end (False).""" )
tf.app.flags.DEFINE_boolean( 'restore', False, 
														 """Train from checkpoint (True).""" )
tf.app.flags.DEFINE_boolean( 'save_meta', False, 
	                           """Save metafiles (True). """ )

def get_current_iteration( epoch, iter_per_batch ):
	""" If restoring a model, return the last iteration. """
	return epoch * iter_batch 


def train():
	""" Training algorithm. """

	# Assert correct parameters. 
	assert FLAGS.net.lower() == 'alexnet' or FLAGS.net.lower() == 'inception' \
	    or FLAGS.net.lower() == 'vgg'     or FLAGS.net.lower() == 'squeezenet', \
	    err_msg( "Network not supported." )

	assert FLAGS.img_format.lower() == 'png'  \
	    or FLAGS.img_format.lower() == 'jpg', \
		err_msg( "Image format not supported." )
	
	if FLAGS.cpu:
		info_msg( "Using CPU" )
		os.environ['CUDA_VISIBLE_DEVICES'] = '' # usar cpu
	else:
		info_msg( "Using GPU with ID {}...".format( FLAGS.gpu ) )
		os.environ['CUDA_VISIBLE_DEVICES'] = str( FLAGS.gpu )

	try:
		info_msg( "Loading training configuration..." )
		cfg = Configuration( FLAGS.data_path, FLAGS.net.lower(), True )
		done_msg() 
	except:
		err_msg( "Could not load configuration. Is your data path: {} ?".format( 
			       FLAGS.data_path ) )

	try:
		info_msg( "Loading training images..." )
		train_set, val_set = dataset.load( cfg, FLAGS.img_format )
		done_msg()
		info_msg( "Training files {} - Validation files {}".format(
			        len( train_set.images ), len( val_set.images ) ) ) 
	except IOError:
		err_msg( "Could not load images." )

	sess = tf.Session()
	
	x = tf.placeholder( tf.float32, 
	   		shape=[None, cfg.img_height, cfg.img_width, cfg.img_depth], name='x' )

	info_msg( "Loading architecture {}...".format( FLAGS.net ) )
	if FLAGS.net.lower() == 'alexnet':
		model     =  cnn.Alexnet( cfg )
		inference =  model.load_net( x )
	elif FLAGS.net == 'vgg':
		model = cnn.VGG16( cfg )
		inference   = model.load_net( x )

	done_msg()

	y = tf.placeholder( tf.float32, shape=[None, cfg.num_classes], name='y')
	y_cls = tf.argmax( y, axis=1 )
	y_hat = tf.nn.softmax( inference, name='y_hat' )
	y_hat_cls = tf.argmax( y_hat, axis=1 )

	# Get all trainable variables 
	var_list = [ v for v in tf.trainable_variables() ]

	with tf.name_scope('loss'):
		cost = tf.nn.softmax_cross_entropy_with_logits_v2( logits=inference, labels=y )
		cost = tf.reduce_mean( cost )
		tf.summary.scalar( 'softmax_cross_entropy', cost )

	with tf.name_scope('optimizer'):
		optimizer = tf.train.AdamOptimizer( 
			cfg.learning_rate ).minimize( cost, var_list = var_list )
		
	with tf.name_scope('accuracy'):
		correct_pred = tf.cast( tf.equal( y_hat_cls, y_cls ), tf.float32 )
		accuracy = tf.reduce_mean( correct_pred )
		tf.summary.scalar( 'accuracy', accuracy )

	train_writer = tf.summary.FileWriter( FLAGS.logs_path + 'train', sess.graph )
	val_writer = tf.summary.FileWriter( FLAGS.logs_path + 'val', sess.graph )
	merge = tf.summary.merge_all()
	saver = tf.train.Saver( max_to_keep=FLAGS.ckpts_to_keep )
	
	curr_iter = 0 
	iter_per_batch = int( train_set.num_examples / cfg.batch_size )

	if FLAGS.finetune and not FLAGS.restore:
		""" Finetuning """
		info_msg( "Finetuning {} layers {} ..."
			.format( FLAGS.net, cfg.net_dict["train_layers"] ) )
		model.load_weights( sess )
		sess.run( tf.global_variables_initializer() )
		done_msg()

	elif FLAGS.restore:
		""" Training from model checkpoint. """
		info_msg( "Loading checkpoint for {} network from {} ..."
			.format( FLAGS.net, FLAGS.model_path ) )
		try:
			saver = tf.train.import_meta_graph( model_dict["meta_file"] )
			checkpoint = tf.train.get_checkpoint_state( FLAGS.model_path )
			checkpoint = checkpoint.model_checkpoint_path
			saver.restore( sess, checkpoint )
			sess.run( tf.global_variables() )

			# Get current iteration from restored model 
			curr_iter = get_current_iteration( int( checkpoint.split('-')[-1]), iter_batch )
		except:
			err_msg("Could not load checkpoint.")
	else: 
		""" Training form scratch """
		sess.run( tf.global_variables_initializer())
		info_msg("Performing end-to-end training.")

	# Training 
	info_msg( "Starting training..." )
	start = t.time()
	for i in range( curr_iter, FLAGS.max_iter ):

		# Get next batches
		train_x, train_y = train_set.next_batch( cfg.batch_size )
		train_dict = { x: train_x, y: train_y }
		
		val_x, val_y = val_set.next_batch( cfg.batch_size )
		val_dict = { x: val_x, y: val_y }

		sess.run( optimizer, feed_dict=train_dict )

		if i % iter_per_batch == 0:
			epoch = train_set.epochs_done
		
			# Show progress 
			train_loss = sess.run( cost, feed_dict=train_dict )
			summary, train_acc = sess.run( [ merge, accuracy ], feed_dict=train_dict )
			train_writer.add_summary( summary, epoch )

			val_loss = sess.run( cost, feed_dict=val_dict )
			summary, val_acc = sess.run( [ merge, accuracy ], feed_dict=val_dict )
			val_writer.add_summary( summary, epoch )

			msg = "\tEPOCH {} --Train[Acc:{:>6.1%}, Loss:{:.3f}] --Val[Acc:{:>6.1%}, Loss:{:.3f}] --Time:{:.3f} --[{} / {}]"
			watch_msg( msg.format( epoch+1, train_acc, train_loss, val_acc, val_loss, 
				 										 t.time()-start, i, FLAGS.max_iter ) )
			start = t.time()

			saver.save( sess, FLAGS.model_path+'model-epoch-{}'.format(epoch), 
				          write_meta_graph=FLAGS.save_meta )

	done_msg("Finished training.")

def main(argv=None):
	if FLAGS.restore == 0:
		if tf.io.gfile.exists(FLAGS.logs_path):
		 	tf.io.gfile.rmtree(FLAGS.logs_path)
		tf.io.gfile.makedirs(FLAGS.logs_path)

		if tf.io.gfile.exists(FLAGS.model_path):
			tf.io.gfile.rmtree(FLAGS.model_path)
		tf.io.gfile.makedirs(FLAGS.model_path)
	train()

if __name__ == '__main__':
	tf.app.run()




