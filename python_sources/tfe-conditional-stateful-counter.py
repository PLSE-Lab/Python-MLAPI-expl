from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow.contrib.eager as tfe

dataset = tf.data.Dataset.from_tensor_slices(([1,2,3,4,5], [-1,-2,-3,-4,-5]))

class My(object):
    def __init__(self):
        self.x = tf.get_variable("mycounter", initializer=lambda: tf.zeros(shape=[], dtype=tf.float32), dtype=tf.float32
                                 , trainable=False) 

v = My()
print(v.x)
tf.assign(v.x,tf.add(v.x,1.0))
print(v.x)

def map_fn(x,v):
    tf.cond(tf.greater_equal(v.x, tf.constant(5.0))
           ,lambda: tf.constant(0.0)
           ,lambda: tf.assign(v.x,tf.add(v.x,1.0))
           )
    return x
    
dataset = dataset.map(lambda x,y: map_fn(x,v)).batch(1)

for batch in tfe.Iterator(dataset):
    print("{} | {}".format(batch, v.x))
