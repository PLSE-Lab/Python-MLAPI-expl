# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
from matplotlib import pyplot as plt
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


## I am trying to do something like 
#https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
from matplotlib import pyplot as plt
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


## I am trying to do something like 
#https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn




# settings
LEARNING_RATE = 0.1
TRAINING_ITERATIONS = 50        
    
DROPOUT = 0.5
BATCH_SIZE = 50

VALIDATION_SIZE = 853


h5f = h5py.File('../input/overlapping_chromosomes_examples.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()

images= pairs[:,:,:,0]
images = np.multiply(images, 1.0 / 255.0)
print('images({0[0]},{0[1]},{0[2]})'.format(images.shape))
output= pairs[:,:,:,1]
image_size= images.shape[1:]
labels_count = 1
print('labels_count => {0}'.format(labels_count))

def dense_to_one_hot(output):
    #labels_count = np.unique(output).shape[0]
    shape_label= list(output.shape)
    shape_label.append(1)
    labels_one_hot=np.zeros(shape_label)
    labels_one_hot[:,:,:,0]+= (output==1)
    labels_one_hot[:,:,:,0]+= (output==2)
    labels_one_hot[:,:,:,0]+= (output==3)
    
    #for i in range(labels_count):
    #    labels_one_hot[:,:,:,i]+= (output==i)
    return labels_one_hot
labels=dense_to_one_hot(output)
print(labels[6,150,50,:],output[6,150,50])

#split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
x = tf.placeholder('float', shape=[None, image_size[0] , image_size[1]])

y_ = tf.placeholder('float', shape=[None, image_size[0] , image_size[1], labels_count])

# first convolutional layer
W_conv1 = weight_variable([1, 1, 1, labels_count])
b_conv1 = bias_variable([labels_count])

# (2000,190,189) => (2000,190,189,1)
image = tf.reshape(x, [-1,image_size[0] , image_size[1],1])


y = conv2d(image, W_conv1) + b_conv1


print(y.get_shape())
cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y, y_))

print(cross_entropy.get_shape())

# optimisation function
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.greater(y*(y_-0.5),0)
#correct_prediction = y>0.5
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y,3)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE]})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


y_mask = np.zeros((853,190,189,labels_count))
for i in range(0,validation_images.shape[0]//BATCH_SIZE):
    y_mask[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = y.eval(feed_dict={x: validation_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})


def plot_fig(n):
    grey = pairs[n,:,:,0]
    plt.subplot(221)
    plt.imshow(grey)
    y = y_mask[n,:,:,:]
    mask_y=np.zeros([190,189])
    mask_y+=1*(y[:,:,0]>0.)
    #mask_y+=2*((y[:,:,0]<0.) & (y[:,:,1]>0.))
    #mask_y+=3*((y[:,:,0]>0.) & (y[:,:,1]>0.))
    plt.subplot(222)
    plt.imshow(mask_y)
    
    tmask = pairs[n,:,:,1]
    plt.subplot(223)
    plt.imshow(tmask)
    plt.savefig('fig{0}.png'.format(n))
plot_fig(3)
plot_fig(70)
plot_fig(134)