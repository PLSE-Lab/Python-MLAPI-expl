import tensorflow as tf

hello = tf.constant('eeee')

sess = tf.Session()

print(sess.run(hello))