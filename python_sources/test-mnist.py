
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with tf.Session() as sess:
    x = tf.get_variable('test', shape=(), initializer=tf.zeros_initializer())
    sess.run(x.initializer)
    x.assign_add(1)
    x = tf.Print(x, [x])
    sess.run(x)