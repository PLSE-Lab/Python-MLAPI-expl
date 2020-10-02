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




# settings
LEARNING_RATE_1 = 0.1
LEARNING_RATE_2 = 1e-2
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 500        
    
DROPOUT = 0.5
BATCH_SIZE = 20

# set to 0 to train on all available data
VALIDATION_SIZE = 53

# image number to output
#IMAGE_TO_DISPLAY = 10

h5f = h5py.File('../input/overlapping_chromosomes_examples.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()

images= pairs[:,:,:,0]
images = np.multiply(images, 1.0 / 255.0)
print('images({0[0]},{0[1]},{0[2]})'.format(images.shape))
output= pairs[:,:,:,1]
image_size= images.shape[1:]
labels_count = 3
print('labels_count => {0}'.format(labels_count))

def dense_to_one_hot_1(output):
    #labels_count = np.unique(output).shape[0]
    shape_label= list(output.shape)
    shape_label.append(1)
    labels_one_hot=np.zeros(shape_label)
    labels_one_hot[:,:,:,0]+= (output==1)
    labels_one_hot[:,:,:,0]+= (output==2)
    labels_one_hot[:,:,:,0]+= (output==3)
    return labels_one_hot
def dense_to_one_hot_2(output):
    #labels_count = np.unique(output).shape[0]
    shape_label= list(output.shape)
    shape_label.append(3)
    labels_one_hot=np.zeros(shape_label,dtype=int)
    labels_one_hot[:,:,:,0]+= (output==1)
    labels_one_hot[:,:,:,1]+= (output==2)
    labels_one_hot[:,:,:,2]+= (output==3)
    
    #for i in range(labels_count):
    #    labels_one_hot[:,:,:,i]+= (output==i)
    return labels_one_hot
labels_1=dense_to_one_hot_1(output)   
labels_2=dense_to_one_hot_2(output)
print(labels_2[6,50,50,:],output[6,50,50])

#split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels_1 = labels_1[:VALIDATION_SIZE]
validation_labels_2 = labels_2[:VALIDATION_SIZE]
train_images = images[VALIDATION_SIZE:]
train_labels_1 = labels_1[VALIDATION_SIZE:]
train_labels_2 = labels_2[VALIDATION_SIZE:]

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

def atrou_conv2d(x, W):
    return tf.nn.atrous_conv2d(x, W,  10, padding='SAME')

    
x = tf.placeholder('float', shape=[None, image_size[0] , image_size[1]])

y1_ = tf.placeholder('float', shape=[None, image_size[0] , image_size[1], 1])
y2_ = tf.placeholder('float', shape=[None, image_size[0] , image_size[1], labels_count])

image = tf.reshape(x, [-1,image_size[0] , image_size[1],1])
# y1 first output, to fit
W_conv = weight_variable([1, 1, 1, 1])
b_conv = bias_variable([1])

y1 = conv2d(image, W_conv) + b_conv

cross_entropy1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(y1, y1_))
train_step1 = tf.train.GradientDescentOptimizer(LEARNING_RATE_1).minimize(cross_entropy1)

# evaluation
correct_prediction1 = tf.greater(y1*(y1_-0.5),0)
#correct_prediction = y>0.5
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, 'float'))
# y2

#im_y1 = tf.concat(3, [image, y1])


#W_conv1 = weight_variable([17, 17, 2, 8])
#b_conv1 = bias_variable([8])

# (2000,190,189) => (2000,190,189,1)



#h_conv1 = tf.sigmoid(conv2d(im_y1, W_conv1) + b_conv1)

#W_conv2 = weight_variable([5, 5, 8, 16])
#b_conv2 = bias_variable([16])


#h_conv2 = tf.sigmoid(conv2d(h_conv1, W_conv2) + b_conv2)


W_conv1 = weight_variable([9, 9, 1, 16])
b_conv1 = bias_variable([16])

# (2000,190,189) => (2000,190,189,1)



h_conv1_a = tf.nn.relu(atrou_conv2d(image, W_conv1) + b_conv1)

W_conv1_b = weight_variable([5, 5, 1, 4])
b_conv1_b = bias_variable([4])


h_conv1_b = tf.nn.relu(conv2d(image, W_conv1_b) + b_conv1_b)

h_conv1=tf.concat(3, [h_conv1_a, h_conv1_b])

W_conv3 = weight_variable([1, 1, 20, labels_count])
b_conv3 = bias_variable([labels_count])

h_conv3 = tf.nn.relu(conv2d(h_conv1, W_conv3) + b_conv3)
#y = tf.sigmoid(conv2d(h_conv2, W_conv3) + b_conv3)
#y2 = conv2d(h_conv1, W_conv3) + b_conv3
y2 = tf.reshape(tf.nn.softmax(tf.reshape(h_conv3,[-1,labels_count])),[-1,190,189,labels_count])
#print(h_conv3.get_shape())
#y = tf.div(tf.exp(h_conv3),np.reshape(tf.reduce_sum(tf.exp(h_conv3),-1),[190,189,1]))
#y = tf.reshape(tf.nn.softmax(tf.reshape(h_conv3,[-1,labels_count])),[-1,190,189,labels_count])
print(y2.get_shape())
cross_entropy = -tf.reduce_sum(y2_*tf.log(y2))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y)+((1-y_)*tf.log(1.-y)))
#cross_entropy = -tf.reduce_sum(tf.mul(tf.reduce_sum(y_*tf.log(y),[0,1,2]),tf.constant([0.1,1,1,1])))
#print(cross_entropy.get_shape())

# optimisation function
train_step2 = tf.train.AdamOptimizer(LEARNING_RATE_2).minimize(cross_entropy)

# evaluation

correct_prediction = tf.boolean_mask(tf.reshape(y1_,[-1,190,189]),tf.equal(tf.argmax(y2,3), tf.argmax(y2_,3)))
#correct_prediction = y>0.5
accuracy = tf.reduce_sum(tf.cast(correct_prediction, 'float'))/tf.reduce_sum(tf.cast(tf.reshape(y1_,[-1,190,189]), 'float'))
#predict = tf.argmax(y,3)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels_1
    global train_labels_2
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
        train_labels_1 = train_labels_1[perm]
        train_labels_2 = train_labels_2[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels_1[start:end], train_labels_2[start:end]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=10
for i in range(100):
    batch_xs, batch_ys1, batch_ys2 = next_batch(BATCH_SIZE)
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy1.eval(feed_dict={x:batch_xs, 
                                                  y1_: batch_ys1,y2_: batch_ys2})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy1.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y1_: validation_labels_1[0:BATCH_SIZE],y2_: validation_labels_2[0:BATCH_SIZE]})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        #if i%(display_step*10) == 0 and i:
        #    display_step *= 10
    # train on batch
    sess.run(train_step1, feed_dict={x: batch_xs, y1_: batch_ys1, y2_: batch_ys2})
    
for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys1, batch_ys2 = next_batch(BATCH_SIZE)      

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y1_: batch_ys1,y2_: batch_ys2})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:BATCH_SIZE], 
                                                            y1_: validation_labels_1[0:BATCH_SIZE],y2_: validation_labels_2[0:BATCH_SIZE]})                                    
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        #if i%(display_step*10) == 0 and i:
        #    display_step *= 10
    # train on batch
    sess.run(train_step2, feed_dict={x: batch_xs, y1_: batch_ys1, y2_: batch_ys2})


y_label = np.zeros((VALIDATION_SIZE,190,189,labels_count))
y_mask = np.zeros((VALIDATION_SIZE,190,189,1))
for i in range(0,validation_images.shape[0]//BATCH_SIZE):
    y_label[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = y2.eval(feed_dict={x: validation_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})
    y_mask[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = y1.eval(feed_dict={x: validation_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE]})


def plot_fig(n):
    plt.figure()
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
    plt.colorbar()
    
    y_l = y_label[n,:,:,:]
    label= np.argmax(y_l, axis=2)+1
    plt.subplot(223)
    plt.imshow(label*mask_y)
    plt.colorbar()
    tmask = pairs[n,:,:,1]
    plt.subplot(224)
    plt.imshow(tmask)
    plt.colorbar()
    
    plt.savefig('fig{0}.png'.format(n))
plot_fig(3)
plot_fig(13)
plot_fig(23)