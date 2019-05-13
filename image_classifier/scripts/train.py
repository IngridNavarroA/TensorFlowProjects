"""
	@author: Ingrid Navarro 
	@date:   May 10th, 2019
	@brief:  Algoritmo de entrenamiento para clasificacion de 
			 defectos en pintura. 
"""

from utils import info_msg, err_msg, done_msg
from networks import Alexnet 

import tensorflow as tf
import dataset
import config 
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', './data/cat-dogs/training_data/', """Folder donde se encuentran las imagenes a entrenar.""")
tf.app.flags.DEFINE_string('network', 'alexnet', """ Arquitecturas a cargar [alexnet, vggnet]""")
tf.app.flags.DEFINE_integer('cpu', 0, """Usar CPU [1 si , 0 no]. """)
tf.app.flags.DEFINE_integer('gpu', 0, """ID del GPU. """)

sess = tf.Session()

if FLAGS.cpu:
	info_msg("Usando CPU")
	os.environ['CUDA_VISIBLE_DEVICES'] = '' # usar cpu
else:
	info_msg("Usando GPU {}...".format(FLAGS.gpu))
	os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

info_msg("Cargando configuracion de entrenamiento...")
cfg = config.base_config(FLAGS.data_path)
done_msg() if cfg else err_msg("No se pudo cargar la configuracion.")

info_msg("Cargando imagenes de entrenamiento...")
train_set, val_set = dataset.load(cfg)
done_msg() if train_set else err_msg("No se pudieron cargar las imagenes.")

# Input placeholders 
X = tf.placeholder(tf.float32, 
	shape=[None, cfg.IMG_WIDTH, cfg.IMG_HEIGHT, cfg.NUM_CHANNELS],
	name='X') # 4D tensor for input images. 
y = tf.placeholder(tf.float32, 
	shape=[None, cfg.NUM_CLS],
	name='y')
y_cls = tf.argmax(y, axis=1)
print(y_cls)


info_msg("Cargando arquitectura {}...".format(FLAGS.network))
network = Alexnet(cfg.NUM_CLS)
out = network.load_net(X)
done_msg()

info_msg("Cargando pesos pre-entrenados para {} de {}..."
	.format(FLAGS.network, cfg.WEIGHTS_ALEXNET))
network.load_weights(cfg.WEIGHTS_ALEXNET, cfg.TRAIN_LAYERS, sess)
done_msg()