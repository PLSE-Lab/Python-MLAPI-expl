#!/usr/bin/env python
# coding: utf-8

# # Digit Classification

# In[1]:


import pandas as pd, numpy as np, tensorflow as tf, sys

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt, matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py, plotly.graph_objs as go
py.init_notebook_mode()


# ## Basic exploration
# * Load train and test data
# * Find target column/columns and extract them from train data
# * Show num of features and num of entries in every dataset

# In[2]:


df_train = pd.read_csv('../input/train.csv') # Index is missing!
df_test = pd.read_csv('../input/test.csv')

column_target = [x for x in df_train.columns if x not in df_test.columns]
labels = df_train[column_target]

df_train.drop(column_target, axis=1, inplace=True)

print("Train shape: {}. Test shape: {}. Target shape {}".format(df_train.shape, df_test.shape, labels.shape))
df_train.head()


# In[3]:


image_size = df_train.shape[1]
print("Number of pixels for each image: {}".format(image_size))
print("Pixels ranges from {} to {}".format(df_train.values.min(), df_train.values.max()))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print("Image size: {}x{}".format(image_width, image_height))

num_colors = 1 # There are in black and white


# In[4]:


def display_image_with_label(index):
    """ Show and image and it's label """
    
    image = df_train.ix[index, :].values
    label = labels.values[index]
    
    plt.axis('off')
    plt.imshow(image.reshape(image_width, image_height), cmap=cm.binary)
    print("It is a {}".format(label))
    
display_image_with_label(42)


# ## Preprocess 
# #### Normalize
# For deep learning is a good practice to normalize inputs. Usually we will substract the mean and divide by the standard deviation for each column. But for images it is more common to use global min/max
# #### One hot encode
# We don't want the label itself, we want a matrix with all posible labels with all 0 except the correct one.
# 
# For example:
# * 0 => [1 0 0 0 0 0 0 0 0 0]
# * 1 => [0 1 0 0 0 0 0 0 0 0]
# * ...
# * 9 => [0 0 0 0 0 0 0 0 0 1]

# In[6]:


def normalize(x):
    """ Normalize a list of sample image data in the range of 0 to 1 """
    
    return (x-x.values.min())/(x.values.max()-x.values.min())

def one_hot_encode(x):
    """ One hot encode a list of sample labels. Return a one-hot encoded vector for each label. """
    
    # check if encoder has been previously created, if not make a global var an initialize it
    if 'encoder' not in globals():
        global encoder
        encoder = LabelBinarizer()
        encoder.fit(range(10))
    
    return encoder.transform(x)

df_train = normalize(df_train)
df_test = normalize(df_test)
labels = one_hot_encode(labels)

num_labels = labels.shape[1]


# ### Split dataset in train, test, val
# * Train --> for training the model
# * Val --> to check accuracy achived and prevent overfitting
# * Test --> final data where to perfome predictions
# 
# We will separate 10% of the training data to create the validation.

# In[7]:


cut_index = int(df_train.shape[0] * 0.1) # Since data is randomly distributed we can use that

x_train, y_train = df_train.head(-cut_index), labels[:-cut_index]
x_val, y_val = df_train.tail(cut_index), labels[-cut_index:]

x_test = df_test

print("x_train: {}, y_train: {}.".format(x_train.shape, y_train.shape))
print("x_val: {}, y_val: {}.".format(x_val.shape, y_val.shape))
print("x_test: {}".format(x_test.shape))


# ## Build the Neural Network
# 
# ### Auxiliar functions for the neural network
# #### Get_batch
# It will split the whole train dataset in smaller groups of images. This increases the training performance. We use yield instead of return to minimize the preprocessing time
# 
# #### Displayer class
# It will be use to show how the neural network is learning

# In[8]:


def get_batch(x, y, batch_size):
    """ Send smaller groups of images """
    
    n_batches = x.shape[0]//batch_size
    
    for step in range(0, n_batches):
        idx_low, idx_high = (step*batch_size, (step+1)*batch_size)
        yield x[idx_low:idx_high], y[idx_low:idx_high]
    
    # If there is more data, yield the remaining
    if (step + 1)*batch_size + 1 < x.shape[0]:
        idx_low, idx_high = (idx_high, x.shape[0])
        yield x[idx_low:idx_high], y[idx_low:idx_high]
        
class Displayer():

    def __init__(self):
        self.accuracies = {'train':[], 'validation':[]}
        
    def show_stats(self, e, epochs, accuracy_train, accuracy_val):

        sys.stdout.write("\rProgress: {:.2f}%  \tTrain accuracy: {:.4f}\t\tValidation accuracy: {:.4f}\t   ".format(
                                100 * e/float(epochs), accuracy_train, accuracy_val))
        
        # This will be useful when plotting for knowing in which epoch we are
        self.accuracies['train'].append(accuracy_train)
        self.accuracies['validation'].append(accuracy_val)


# ### Define NN architecture

# In[9]:


tf.reset_default_graph()

alpha = 0.1

x = tf.placeholder(tf.float32, [None, image_size], name='x')
y = tf.placeholder(tf.float32, [None, num_labels], name='y')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

images = tf.reshape(x, [-1, image_width, image_height, num_colors])

nn = tf.layers.conv2d(images, 6, 5, padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer())
nn = tf.nn.avg_pool(nn, [1,2,2,1], [1,2,2,1], padding='VALID')

nn = tf.layers.conv2d(nn, 16, 5, padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer())
nn = tf.nn.avg_pool(nn, [1,2,2,1], [1,2,2,1], padding='VALID')

nn = tf.contrib.layers.flatten(nn)

nn = tf.layers.dense(nn, 120, kernel_initializer=tf.contrib.layers.xavier_initializer())
nn = tf.layers.batch_normalization(nn)
nn = tf.maximum(nn, nn*alpha)
nn = tf.nn.dropout(nn, keep_prob)

nn = tf.layers.dense(nn, 84, kernel_initializer=tf.contrib.layers.xavier_initializer())
nn = tf.layers.batch_normalization(nn)
nn = tf.maximum(nn, nn*alpha)
nn = tf.nn.dropout(nn, keep_prob)

logits = tf.layers.dense(nn, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="logits")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

pred = tf.argmax(logits, 1)


# ## Train the Neural Network

# In[10]:


def train(epochs, batch_size, keep_probability, lr):
    
    disp = Displayer()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs + 1):

            for x_batch, y_batch in get_batch(df_train, labels, batch_size):  
                sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: keep_probability, learning_rate:lr})

            accuracy_train = accuracy.eval({x: x_batch, y: y_batch, keep_prob: 1.0})
            accuracy_val = accuracy.eval({x: x_val, y: y_val, keep_prob: 1.0})

            disp.show_stats(e, epochs, accuracy_train, accuracy_val)
            
        output = pred.eval({x: x_test, keep_prob: 1.0})
                     
    data = [go.Scatter(y=disp.accuracies[c], name=c) for c in disp.accuracies]
    py.iplot(data, show_link=False)
    
    return output


# In[11]:


epochs = 50
batch_size = 1024
keep_probability = 0.5
lr = 0.001

predictions = train(epochs, batch_size, keep_probability, lr)


# ## Export and view predictions for the test data

# In[19]:


df_out = pd.DataFrame(predictions)
df_out.index = [x + 1 for x in df_out.index]
df_out.columns = column_target
df_out.index.name = "ImageId"

df_out.to_csv("predictions.csv")
df_out.head()


# In[14]:


for index in range(5):
    plt.figure()
    image = df_test.ix[index,:].values
    plt.imshow(image.reshape(image_width, image_height), cmap=cm.binary)


# In[ ]:




