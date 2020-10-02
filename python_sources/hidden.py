
# coding: utf-8

# In[1]:

import numpy as np
from scipy import ndimage
import pandas as pd

import tensorflow as tf

# settings
LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 20000     
    
DROPOUT = 0.5
BATCH_SIZE = 75

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 10

# In[2]:

# read training data from CSV file 
data = pd.read_csv('../input/train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))
print (data.head())


# In[3]:

images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))


# In[4]:

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


# In[5]:

# display image
def display(img):
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    plt.axis('off')
    plt.imshow(one_image,cmap=cm.binary)



# In[6]:

labels_flat = data[[0]].values.ravel()

print('labels_flat({0})'.format(len(labels_flat)))
print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))



# In[7]:

labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))


# In[8]:

# Increase the training data set by appending rotated and scaled images

#rotation angle in degree
def rotateImage(img, angle, pivot):
    img = img.reshape(image_width,image_height)
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    imgR = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    imgR = imgR.reshape(image_size,)
    return( imgR )

# pivot = (image_height/2,image_width/2)
# pos_rotated = map(lambda x:rotateImage(x, 15, pivot),images)
# print("pos_complete")
# neg_rotated = map(lambda x:rotateImage(x, -15, pivot),images)
# print("neg_complete")

# images = np.concatenate((images,pos_rotated,neg_rotated))
# print("concat_complete")
# labels_flat = np.tile(labels_flat,3)


# In[9]:

# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))


# In[10]:

# split data into training & validation
# num_images = images.shape[0]
# val_size = VALIDATION_SIZE /3
# val_inds = [ np.array(range(0, val_size))+ (num_images/3 + val_size) * i for i in range(3)]
# non_val_inds = [ np.array(range(0,i*val_size)+range((i+1)*val_size,num_images/3))+num_images*i/3 for i in range(3)]

# val_inds = np.concatenate((val_inds[0],val_inds[1],val_inds[2]))
# non_val_inds = np.concatenate((non_val_inds[0],non_val_inds[1],non_val_inds[2]))

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


print('train_images({0[0]},{0[1]})'.format(train_images.shape))
print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))


# In[11]:

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[12]:

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[13]:

# pooling
# [[0,3],
#  [4,2]] => 4

# [[0,1],
#  [1,1]] => 1

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[14]:

# input & output of NN

# images
x = tf.placeholder('float', shape=[None, image_size])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])


# In[15]:

# first convolutional layer
num_feat = 32
W_conv1 = weight_variable([3, 3, 1, num_feat])
b_conv1 = bias_variable([num_feat])

# (40000,784) => (40000,28,28,1)
image = tf.reshape(x, [-1,image_width , image_height,1])
print (image.get_shape()) # =>(40000,28,28,1)


h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
print (h_conv1.get_shape()) # => (40000, 28, 28, 40)
h_pool1 = max_pool_2x2(h_conv1)
print (h_pool1.get_shape()) # => (40000, 14, 14, 40)


# Prepare for visualization
# display 40 fetures in 5 by 8 grid
a, b = 4, 8
layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, a ,b))  

# reorder so the channels are in the first dimension, x and y follow.
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))

layer1 = tf.reshape(layer1, (-1, image_height*a, image_width*b)) 


# In[16]:

# second convolutional layer
W_conv2 = weight_variable([2, 2, num_feat, num_feat * 2])
b_conv2 = bias_variable([num_feat * 2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#print (h_conv2.get_shape()) # => (40000, 14,14, 64)
h_pool2 = max_pool_2x2(h_conv2)
#print (h_pool2.get_shape()) # => (40000, 7, 7, 64)

# Prepare for visualization
# display 64 fetures in 4 by 16 grid
layer2 = tf.reshape(h_conv2, (-1, 14, 14, a ,2*b))  

# reorder so the channels are in the first dimension, x and y follow.
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))

layer2 = tf.reshape(layer2, (-1, 14*a, 14*b)) 


# In[17]:

# densely connected layer
W_fc1 = weight_variable([7 * 7 * num_feat*2, 1024])
b_fc1 = bias_variable([1024])

# (40000, 7, 7, 64) => (40000, 3136)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_feat*2])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (40000, 1024)

keep_prob = tf.placeholder('float')
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# In[18]:

hidden_nodes = 400

W_new = weight_variable([1024,hidden_nodes])
b_new = bias_variable([hidden_nodes])
h_new = tf.nn.relu(tf.matmul(h_fc1,W_new)+b_new)

h_new_dropout = tf.nn.dropout(h_new,keep_prob)




# dropout


# In[19]:

# readout layer for deep net
W_fc2 = weight_variable([hidden_nodes, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_new_dropout, W_fc2) + b_fc2)

#print (y.get_shape()) # => (40000, 10)


# In[20]:

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


# In[21]:

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)


# In[22]:

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

    # when all training data have been already used, it is reorder randomly    
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

val_epochs_completed = 0
val_index_in_epoch = 0  
# serve validation data by batches
def next_valid_batch(batch_size):
    
    global validation_images
    global validation_labels
    global val_index_in_epoch
    global val_epochs_completed
    
    start = val_index_in_epoch
    val_index_in_epoch += batch_size

    # when all training data have been already used, it is reorder randomly    
    if val_index_in_epoch > VALIDATION_SIZE:
        # finished epoch
        val_epochs_completed += 1
        # shuffle the data
        perm = np.arange(VALIDATION_SIZE)
        np.random.shuffle(perm)
        validation_images = validation_images[perm]
        validation_labels = validation_labels[perm]
        # start next epoch
        start = 0
        val_index_in_epoch = batch_size
        assert batch_size <= VALIDATION_SIZE
    end = val_index_in_epoch
    return validation_images[start:end], validation_labels[start:end]

# In[23]:

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)


# In[24]:

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=50

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            val_xs, val_ys = next_valid_batch(BATCH_SIZE)
            validation_accuracy = accuracy.eval(feed_dict={x: val_xs, 
                                                           y_: val_ys, 
                                                           keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            validation_accuracies.append(validation_accuracy)
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
                
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

# In[78]:

# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)

# In[27]:

# read test data from CSV file 
test_images = pd.read_csv('../input/test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))

# output test image and prediction
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# save results
np.savetxt('../input/sample_submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')





