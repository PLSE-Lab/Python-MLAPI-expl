# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 http://www.activelog.org. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf

FLAGS = None

def getdata(name):
    """Return image and labels array."""
    # Import data

    # pd.read_csv -> Read CSV (comma-separated) file into DataFrame
    data = pd.read_csv(name)
    # iloc -> Purely integer-location based indexing for selection by position.
    images = data.iloc[:, 1:].values
    # type conversion, as pandas in read_csv performs dtype guessing and will use int type
    images = images.astype(np.float)

    # convert from (0-255) => (0.0-1.0)
    images = np.multiply(images, 1.0 / 255.0)
    labels = data[[0]].values.ravel()

    return (images, labels)

def getTestdata(name):
    """Return image array."""
    # Import data
    # pd.read_csv -> Read CSV (comma-separated) file into DataFrame
    data = pd.read_csv(name)
    # iloc -> Purely integer-location based indexing for selection by position.
    images = data.iloc[:, 0:].values
    # type conversion, as pandas in read_csv performs dtype guessing and will use int type
    images = images.astype(np.float)

    # Convert from [0, 255] -> [0.0, 1.0].
    images = np.multiply(images, 1.0 / 255.0)
    return images

def arrayToOneHot(data):
    """get one hot tensor."""    
    # convert get one hot tensor
    #labels = tf.one_hot(data, 10, None, None, None, tf.int8, None)
    # tf.one_hot -> dtype will default to the value tf.float32
    labels = tf.one_hot(indices=data, depth=10)
    return labels

def main():
    """main"""

    #train - 42000 records
    #test  - 28000 records

    # Import data
    train_images, train_labels = getdata('../input/train.csv')
    # print('train_images({0[0]},{0[1]})'.format(train_images.shape))
    # print('train_labels({0[0]})'.format(train_labels.shape))

    #print(pd.DataFrame(train_images).head())
    #print(pd.DataFrame(train_labels).head())

    test_images = getTestdata('../input/test.csv')
    # print('test_images({0[0]},{0[1]})'.format(test_images.shape))
    # print('test_labels({0[0]})'.format(test_labels.shape))
    #print(pd.DataFrame(test_images).head())

    batch_size = 20

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784]) # None -> a dimension can be of any length

    W = tf.Variable(tf.zeros([784, 10]))        # W defines weights
    b = tf.Variable(tf.zeros([10]))             # b defines biases for a model
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10]) # None -> a dimension can be of any length

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # convert array to one hot array using tensorflow
    # then extract array from tensor using eval function
    train_labels = arrayToOneHot(train_labels).eval()

    # print('train_labels({0[0]},{0[1]})'.format(train_labels.shape))
    # print(pd.DataFrame(train_labels).head())

    # Train
    for i in range(1000):

        if i+batch_size > len(train_images):
            break

        batch_xs = train_images[i:i+batch_size]
        batch_ys = train_labels[i:i+batch_size]

        # print(np.shape(batch_xs))
        # print(np.shape(batch_ys))

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model

    # tf.argmax - Returns the index with the largest value across axiss of a tensor.
    # first index is array (vector or matrix), second is axis
    # matrix has 2 axises
    # 0 is from top to bottom
    # 1 is from left to right

    prediction = tf.argmax(y, 1)

    #
    classification = sess.run(prediction, feed_dict={x: test_images})

    # print(classification)
    submission = pd.DataFrame(data={'ImageId':(np.arange(classification.shape[0])+1), 'Label':classification})
    submission.to_csv('submission.csv', index=False)

    print(submission.head(10))

    sess.close()
    print('done')

if __name__ == '__main__':
  main()