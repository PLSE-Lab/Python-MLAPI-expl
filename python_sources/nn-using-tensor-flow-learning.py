

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
train.head()
train.shape
images = train.iloc[:,1:]
images.shape
labels = train.iloc[:,:1]
type(labels)
labels.head()
labels['label'][0]
#np.divide(images, 255)
# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(28,28)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
from random import randint
def display_random_img():
    rows = len(images.index)
    index = randint(0, rows)
    print("Image at index:" + str(index))
    img = images.iloc[index]
    print(labels['label'][index])
    display(img)

display_random_img()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b) ## matrix mul 
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
output = dense_to_one_hot(labels['label'], 10)
output = output.astype(np.uint8)
print(output.shape)
ci = 0

  
sess.run(train_step, feed_dict={x: images, y_: output})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
print(sess.run(accuracy, feed_dict={x: images, y_: output}))




test_images = pd.read_csv('../input/test.csv').values
test_images = test_images.astype(np.float)
test_images = np.multiply(test_images, 1.0 / 255.0)

predicated_labels = sess.run(y ,feed_dict={x: test_images})

predicated_labels[:5, :]
d_pre_labels = [];
for pl in predicated_labels:
    maxIndex = 0
    maxValue = 0
    index = 0
    for p in pl:
        if (p > maxValue) :
            maxValue = p
            maxIndex = index
        index += 1 
    d_pre_labels.append(maxIndex)

np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(test_images)+1),d_pre_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')