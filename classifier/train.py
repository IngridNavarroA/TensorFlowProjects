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
import networks as N
import tensorflow as tf
import numpy as np
import time as t
import os

np.random.seed(1)
tf.compat.v1.set_random_seed(2)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data', './data/cat-dog/training_data/', """Path to training images.""")
tf.app.flags.DEFINE_string('net', 'alexnet', """Networks available [alexnet | vgg | inception].""")
tf.app.flags.DEFINE_string('img_format', 'jpg', """ Image format [png | jpg]. """)
tf.app.flags.DEFINE_string('model', 'model/', """Path to checkpoints.""")
tf.app.flags.DEFINE_string('log', 'logs/', """Path to log training. """)

tf.app.flags.DEFINE_integer('max_iter', 5000, """Max number of iterations for training. """)
tf.app.flags.DEFINE_integer('ckpts_to_keep', 15, """Max number of checkpoints to keep. """)
tf.app.flags.DEFINE_integer('finetune', 0, """Finetune or train end-to-end [1 / 0].""")
tf.app.flags.DEFINE_integer('restore', 0, """Train from checkpoint [1 / 0].""")
tf.app.flags.DEFINE_integer('meta', 0, """Save metafiles [1 / 0]. """)
tf.app.flags.DEFINE_integer('gpu', 0, """What GPU (ID) to use. """)
tf.app.flags.DEFINE_integer('cpu', 0, """Use CPU [1 / 0]. """)

def get_current_iteration(epoch:int, iter_batch:int) -> int:
	""" If restoring a model, return the last iteration. """
	return int(epoch * iter_batch)


def train():
	""" Training algorithm. """

	# Assert correct parameters. 
	assert FLAGS.net.lower() == 'alexnet' or FLAGS.net.lower() == 'inception' \
	    or FLAGS.net.lower() == 'vgg'     or FLAGS.net.lower() == 'resnet' \
	    or FLAGS.net.lower() == 'squeezenet', \
	    err_msg("Network not supported.")

	assert FLAGS.img_format.lower() == 'png' or FLAGS.img_format.lower() == 'jpg', \
		err_msg("Image format not supported.")
	
	if FLAGS.cpu:
		info_msg("Using CPU")
		os.environ['CUDA_VISIBLE_DEVICES'] = '' # usar cpu
	else:
		info_msg("Using GPU with ID {}...".format(FLAGS.gpu))
		os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

	try:
		info_msg("Loading training configuration...")
		cfg = Configuration(FLAGS.data, "train", 200, FLAGS.net.lower())
		done_msg() 
	except:
		err_msg("Could not load configuration. Is your data path: {} ?".format(FLAGS.data))

	try:
		info_msg("Loading training images...")
		train_set, val_set = dataset.load(cfg, FLAGS.img_format)
		done_msg()
		info_msg("Training files {} - Validation files {}".format(
			        len(train_set.images), len(val_set.images))) 
	except IOError:
		err_msg("Could not load images.")

	sess = tf.compat.v1.Session()

	# Inputs 
	x = tf.compat.v1.placeholder(tf.float32, 
		      shape=[None, cfg.img_height, cfg.img_width, cfg.img_depth], name='x')
	y = tf.compat.v1.placeholder(tf.float32, 
		      shape=[None, cfg.num_classes], name='y')
	y_cls = tf.argmax(y, axis=1)
	
	info_msg("Loading architecture {}...".format(FLAGS.net))
	if FLAGS.net.lower() == 'alexnet':
		model =  N.Alexnet(cfg.num_classes, cfg.dropout_rate)
		out   =  model.load_net(x, training=True)

	elif FLAGS.net == 'vgg':
		model = N.VGG16(cfg.NUM_CLS)
		out   = model.load_net(x, training=True)
	model_dict = cfg.net_dict

	# elif FLAGS.net == 'inception':
	# 	model = N.Inception(cfg.NUM_CLS)
	# 	out = model.load_net(x, training=True)
	# 	model_dict = cfg.INCEPTION

	done_msg()

	# y_pred = tf.nn.softmax(out, name='y_pred')
	# y_pred_cls = tf.argmax(y_pred, axis=1)

	# # Variables to train 
	# var_list = [v for v in tf.trainable_variables()]

	# # Cost
	# with tf.name_scope('cross_entropy'):
	# 	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y)
	# 	cost = tf.reduce_mean(cross_entropy)
	# 	tf.summary.scalar('cross_entropy', cost)

	# # Optimizer 
	# with tf.name_scope('optimizer'):
	# 	gradients = tf.gradients(cost, var_list)
	# 	gradients = list(zip(gradients, var_list))
	# 	optimizer = tf.train.GradientDescentOptimizer(cfg.LEARNING_RATE)
	# 	train_opt = optimizer.apply_gradients(grads_and_vars=gradients)

	# # Accuracy
	# with tf.name_scope('accuracy'):
	# 	correct_pred = tf.equal(y_pred_cls, y_cls)
	# 	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	# 	tf.summary.scalar('accuracy', accuracy)

	# train_writer = tf.summary.FileWriter(FLAGS.log + 'train', sess.graph)
	# val_writer = tf.summary.FileWriter(FLAGS.log + 'val', sess.graph)
	# merge = tf.summary.merge_all()
	# saver = tf.train.Saver(max_to_keep=FLAGS.ckpts_to_keep)
	# curr_iter, iter_batch = 0, int(train_set.num_examples / cfg.BATCH_SIZE)

	# if FLAGS.finetune and not FLAGS.restore:
	# 	""" Finetuning """
	# 	info_msg("Finetuning {} network layers {} ..."
	# 		.format(FLAGS.net, model_dict["train_layers"]))
	# 	model.load_weights(model_dict["weights"], model_dict["train_layers"], sess)
	# 	sess.run(tf.global_variables_initializer())
	# 	done_msg()
	# elif FLAGS.restore:
	# 	""" Training from model checkpoint. """
	# 	info_msg("Loading checkpoint for {} network from {} ..."
	# 		.format(FLAGS.net, FLAGS.model))
	# 	try:
	
	# 		saver = tf.train.import_meta_graph(model_dict["meta_file"])
	# 		checkpoint = tf.train.get_checkpoint_state(FLAGS.model)
	# 		checkpoint = checkpoint.model_checkpoint_path
	# 		saver.restore(sess, checkpoint)
	# 		sess.run(tf.global_variables())

	# 		# Get current iteration from restored model 
	# 		curr_iter = get_current_iteration(int(checkpoint.split('-')[-1]), iter_batch)
	# 		curr_iter = 0 if curr_iter > int(cfg.RESTORE * FLAGS.max_iter) else curr_iter

	# 	except:
	# 		err_msg("Could not load checkpoint.")
	# else: 
	# 	""" Training form scratch """
	# 	sess.run(tf.global_variables_initializer())
	# 	info_msg("Performing end-to-end training.")

	# # Training 
	# info_msg("Starting training...")
	# start = t.time()
	# for i in range(curr_iter, FLAGS.max_iter):

	# 	# Get next batches
	# 	train_x, train_y = train_set.next_batch(cfg.BATCH_SIZE)
	# 	train_dict = {x: train_x, y: train_y}
		
	# 	val_x, val_y = val_set.next_batch(cfg.BATCH_SIZE)
	# 	val_dict = {x: val_x, y: val_y}

	# 	sess.run(train_opt, feed_dict=train_dict)

	# 	if i % iter_batch == 0:
	# 		# Compute loss 
	# 		val_loss = sess.run(cost, feed_dict=val_dict)
	# 		train_loss = sess.run(cost, feed_dict=train_dict)

	# 		epoch = int(i / iter_batch)

	# 		# Show progress 
	# 		summary, train_acc = sess.run([merge, accuracy], feed_dict=train_dict)
	# 		train_writer.add_summary(summary, epoch)

	# 		summary, val_acc = sess.run([merge, accuracy], feed_dict=val_dict)
	# 		val_writer.add_summary(summary, epoch)

	# 		msg = "\tEPOCH {0} --Train[Acc:{1:>6.1%}, Loss:{2:.3f}] --Val[Acc:{3:>6.1%}, Loss:{4:.3f}] --Time:{5:.3f} --[{6} / {7}]"
	# 		done_msg(msg.format(epoch+1, train_acc, train_loss, val_acc, val_loss, t.time()-start, i, FLAGS.max_iter))
	# 		start = t.time()

	# 		saver.save(sess, FLAGS.model+'model-epoch-{}'.format(epoch), write_meta_graph=FLAGS.meta)

	# done_msg("Finished training.")

def main(argv=None):
	if FLAGS.restore == 0:
		if tf.io.gfile.exists(FLAGS.log):
			tf.io.gfile.DeleteRecursively(FLAGS.log)
			tf.io.gfile.MakeDirs(FLAGS.log)

		if tf.io.gfile.exists(FLAGS.model):
			tf.io.gfile.DeleteRecursively(FLAGS.model)
			tf.io.gfile.MakeDirs(FLAGS.model)
	train()

if __name__ == '__main__':
	tf.compat.v1.app.run()




