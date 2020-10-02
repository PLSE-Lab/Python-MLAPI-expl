# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf



# Basic constant operations

# The value returned by the constructor represents the output

# of the Constant op.

a = tf.constant(2)

b = tf.constant(3)



# Launch the default graph.

with tf.Session() as sess:

    print("a=2, b=3")

    print("Addition with constants: %i" % sess.run(a+b))

    print("Multiplication with constants: %i" % sess.run(a*b))



# Basic Operations with variable as graph input

# The value returned by the constructor represents the output

# of the Variable op. (define as input when running session)

# tf Graph input

a = tf.placeholder(tf.int16)

b = tf.placeholder(tf.int16)



# Define some operations

add = tf.add(a, b)

mul = tf.multiply(a, b)



# Launch the default graph.

with tf.Session() as sess:

    # Run every operation with variable input

    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))

    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))





# ----------------

# More in details:

# Matrix Multiplication from TensorFlow official tutorial



# Create a Constant op that produces a 1x2 matrix.  The op is

# added as a node to the default graph.

#

# The value returned by the constructor represents the output

# of the Constant op.

matrix1 = tf.constant([[3., 3.]])



# Create another Constant that produces a 2x1 matrix.

matrix2 = tf.constant([[2.],[2.]])



# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.

# The returned value, 'product', represents the result of the matrix

# multiplication.

product = tf.matmul(matrix1, matrix2)



# To run the matmul op we call the session 'run()' method, passing 'product'

# which represents the output of the matmul op.  This indicates to the call

# that we want to get the output of the matmul op back.

#

# All inputs needed by the op are run automatically by the session.  They

# typically are run in parallel.

#

# The call 'run(product)' thus causes the execution of threes ops in the

# graph: the two constants and matmul.

#

# The output of the op is returned in 'result' as a numpy `ndarray` object.

with tf.Session() as sess:

    result = sess.run(product)

    print(result)

    # ==> [[ 12.]]