import tensorflow as tf 

sess = tf.Session()
saver = tf.train.import_meta_graph('pretrain/alexnet.meta')
saver.restore(sess, 'model/model-epoch-78')