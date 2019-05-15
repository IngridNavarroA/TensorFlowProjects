"""
	@author: Ingrid Navarro 
	@date:   May 10th, 2019
	@brief:  Algoritmo de entrenamiento para clasificacion de 
			 defectos en pintura. 
"""
from utils import info_msg, err_msg, done_msg
from networks import Alexnet 
import tensorflow as tf
import numpy as np
import time as t
import dataset
import config 
import os

np.random.seed(1)
tf.set_random_seed(2)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data/cat-dogs/training_data/', """Path to training images.""")
tf.app.flags.DEFINE_string('model_path', 'model/', """Path to checkpoints.""")
tf.app.flags.DEFINE_string('log_path', 'logs/', """Path to log training. """)
tf.app.flags.DEFINE_string('network', 'alexnet', """Networks available [alexnet].""")

tf.app.flags.DEFINE_integer('max_iter', 3000, """Max number of iterations for training. """)
tf.app.flags.DEFINE_integer('finetune', 0, """Finetune or train end-to-end [1 / 0].""")
tf.app.flags.DEFINE_integer('restore', 0, """Train from checkpoint [1 / 0].""")
tf.app.flags.DEFINE_integer('cpu', 0, """Use CPU [1 / 0]. """)
tf.app.flags.DEFINE_integer('gpu', 0, """What GPU (ID) to use. """)

def train():
	""" Training algorithm. """
	sess = tf.Session()

	if FLAGS.cpu:
		info_msg("Using CPU")
		os.environ['CUDA_VISIBLE_DEVICES'] = '' # usar cpu
	else:
		info_msg("Using GPU with ID {}...".format(FLAGS.gpu))
		os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

	try:
		info_msg("Loading training configuration...")
		cfg = config.base_config(FLAGS.data_path)
		done_msg() 
	except:
		err_msg("Could not load configuration. Is your data path: {} ?".format(FLAGS.data_path))

	try:
		info_msg("Loading training images...")
		train_set, val_set = dataset.load(cfg)
		done_msg()
		info_msg("Training files {} - Validation files {}".format(len(train_set.images), len(val_set.images))) 
	except IOError:
		err_msg("Could not load images.")

	x = tf.placeholder(tf.float32, 
		shape=[None, cfg.IMG_WIDTH, cfg.IMG_HEIGHT, cfg.NUM_CHANNELS],
		name='x') # 4D tensor for input images. 
	y = tf.placeholder(tf.float32, 
		shape=[None, cfg.NUM_CLS],
		name='y')
	y_cls = tf.argmax(y, axis=1)

	trainable = False if FLAGS.finetune else True

	info_msg("Loading architecture {}...".format(FLAGS.network))
	network = Alexnet(cfg.NUM_CLS)
	out = network.load_net(x, trainable)
	y_pred = tf.nn.softmax(out, name='y_pred')
	y_pred_cls = tf.argmax(y_pred, axis=1)
	done_msg()

	# Variables to train 
	var_list = [v for v in tf.trainable_variables()]

	# Type of training 
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

	if FLAGS.finetune and not FLAGS.restore:
		""" Finetuning """
		info_msg("Finetuning {} network layers {} ..."
			.format(FLAGS.network, cfg.TRAIN_LAYERS))
		network.load_weights(cfg.WEIGHTS_ALEXNET, cfg.TRAIN_LAYERS, sess)
		done_msg()
	elif FLAGS.restore:
		""" Training from model checkpoint. """
		info_msg("Loading checkpoint for {} network from {} ..."
			.format(FLAGS.network, FLAGS.model_path))
		try:
			checkpoint = tf.train.get_checkpoint_state(FLAGS.model_path)
			checkpoint_path = checkpoint.model_checkpoint_path
			saver.restore(sess, checkpoint_path)
		except:
			err_msg("Could not load checkpoint.")
	else: 
		""" Training form scratch. """
		info_msg("Performing end-to-end training.")

	# Cost
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y)
		cost = tf.reduce_mean(cross_entropy)
		tf.summary.scalar('cross_entropy', cost)

	# Optimizer 
	with tf.name_scope('optimizer'):
		gradients = tf.gradients(cost, var_list)
		gradients = list(zip(gradients, var_list))
		optimizer = tf.train.GradientDescentOptimizer(cfg.LEARNING_RATE)
		train_opt = optimizer.apply_gradients(grads_and_vars=gradients)

	# Accuracy
	with tf.name_scope('accuracy'):
		correct_pred = tf.equal(y_pred_cls, y_cls)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	train_writer = tf.summary.FileWriter(FLAGS.log_path + 'train', sess.graph)
	val_writer = tf.summary.FileWriter(FLAGS.log_path + 'val', sess.graph)
	merge = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())

	# Training 
	info_msg("Starting training...")
	start = t.time()
	for i in range(FLAGS.max_iter):

		# Get next batches
		train_x, train_y = train_set.next_batch(cfg.BATCH_SIZE)
		train_dict = {x: train_x, y: train_y}
		
		val_x, val_y = val_set.next_batch(cfg.BATCH_SIZE)
		val_dict = {x: val_x, y: val_y}

		sess.run(train_opt, feed_dict=train_dict)

		iter_batch = int(train_set.num_examples / cfg.BATCH_SIZE)
		if i % iter_batch == 0:
			# Compute loss 
			val_loss = sess.run(cost, feed_dict=val_dict)
			train_loss = sess.run(cost, feed_dict=train_dict)

			epoch = int(i / iter_batch)

			# Show progress 
			summary, train_acc = sess.run([merge, accuracy], feed_dict=train_dict)
			train_writer.add_summary(summary, epoch)

			summary, val_acc = sess.run([merge, accuracy], feed_dict=val_dict)
			val_writer.add_summary(summary, epoch)

			msg = "\tEPOCH {0} --Train[Acc:{1:>6.1%}, Loss:{2:.3f}] --Val[Acc:{3:>6.1%}, Loss:{4:.3f}] --Time:{5:.3f}"
			done_msg(msg.format(epoch+1, train_acc, train_loss, val_acc, val_loss, t.time()-start))
			start = t.time()

			saver.save(sess, FLAGS.model_path+'model-epoch-{}'.format(epoch))

	done_msg("Finished training.")

def main(argv=None):
	if FLAGS.restore == 0:
		if tf.gfile.Exists(FLAGS.log_path):
			tf.gfile.DeleteRecursively(FLAGS.log_path)
			tf.gfile.MakeDirs(FLAGS.log_path)

		if tf.gfile.Exists(FLAGS.model_path):
			tf.gfile.DeleteRecursively(FLAGS.model_path)
			tf.gfile.MakeDirs(FLAGS.model_path)
	train()

if __name__ == '__main__':
	tf.app.run()




