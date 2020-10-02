#!/usr/bin/env python
# coding: utf-8

# #Paint your catdog in TensorFlow
# 
# Reference:
# https://github.com/pkmital/CADL/blob/master/session-2/lecture-2.ipynb
# 
# ###Predict color to paint based on row/column position of data.

# In[50]:


TRAIN_DIR = '../input/train/'
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


# Create "files" variable which is list of path names for the files in a directory. Choose only the first 100 files. Read all rows, columns and 3 dimensions in Z axis for the first 100 path names (files) and make it into a new list (imgs). If you prefer dogs, you can play around with the index to choose a file or group of files further down the "files" list. Resize all the images in the imgs variable to certain size (depending on the quality you want and time you are willing to wait you can increase/decrease this size). Finally, create a numpy array out of the "imgs" variable.

# In[51]:


files = [os.path.join(TRAIN_DIR, fname)
        for fname in os.listdir(TRAIN_DIR)]

cats = files[:3]
dogs = files[-4:-1]

cat_imgs = [plt.imread(fname)[..., :3] for fname in cats]
cat_imgs = [resize(img_i, (100, 100)) for img_i in cat_imgs]
cat_imgs = np.array(cat_imgs).astype(np.float32)

dog_imgs = [plt.imread(fname)[..., :3] for fname in dogs]
dog_imgs = [resize(img_i, (100, 100)) for img_i in dog_imgs]
dog_imgs = np.array(dog_imgs).astype(np.float32)


# In[52]:


plt.imshow(cat_imgs[1])


# In[53]:


plt.imshow(dog_imgs[1])


# In[54]:


imgs = np.concatenate((cat_imgs[1], dog_imgs[1]), axis=1)
plt.imshow(imgs)


# Checking chosen image shape

# Function that loops over rows (img.shape[0]) and columns (img.shape[1]) of an image and creates inputs (xs) and outputs (ys) where inputs are the row/column coordinate and outputs are the three RGB color channel numbers at that coordinate (img[row_i, col_i])

# In[55]:


def split_image(img):
    # positions, ie row/column tuple
    xs = []

    # 3 rgb colors
    ys = []

    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])
            
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


# Splitting chosen image into inputs(coordinates) and targets(RGB color numbers).

# In[56]:


xs, ys = split_image(imgs)

xs.shape, ys.shape


# Normalize input incase image array numbers in 0-255 range.

# In[57]:


xs = ((xs - np.mean(xs)) / np.std(xs))


# In[58]:


print(np.min(ys), np.max(ys))


# Input tensor placeholder X(coordinates) needs arbitrary amount of 2D inputs.

# In[59]:


X = tf.placeholder(name='X', shape=(None, 2), dtype=tf.float32)


# Linear function does matrix multiply with n_input (row, columns coordinates in this case) and n_outputs (size of second dimension in matrix multiply) and adds a bias value to the matrix and then optionally applies an activation function, returning the output("h" whilst moving through the neural network and Y_pred on its final step as well as the resulting weight matrix "W".

# In[60]:


def linear(x, n_output, name=None, activation=None, reuse=None):

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fully_connected", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


# Return "h" output and "W" weight matrix, the result of passing tensor "X" through the linear function with 20 outputs per 1 input, giving the operation the name 'linear' and using the tf.nn.relu activation function.

# In[61]:


h, W = linear(x=X, n_output=20, name='linear', activation=tf.nn.relu)


# Create input X and output Y tensor placeholders that correspond to 2 coordinate inputs for X and 3 RGB color outputs for Y.

# In[62]:


tf.reset_default_graph()
X = tf.placeholder(name='X', shape=(None, 2), dtype=tf.float32)
Y = tf.placeholder(name='Y', shape=(None, 3), dtype=tf.float32)


# Specify our network structure, which will take in X and pass it through a number of layers. On each layer a multiplication with a weight matrix takes place, an addition of the bias and a pass through an activation function (except for the final layer where there is no activation and instead of passing "n_neurons" we specify the number of outputs we want for our predicted Y, which is 3 numbers corresponding to 3 RGB color values. Different activation functions might yield different final results.

# In[63]:


n_neurons = 100
h1, W1 = linear(X, n_neurons, name='layer1', activation=tf.nn.relu)
h2, W2 = linear(h1, n_neurons, name='layer2', activation=tf.nn.relu)
h3, W3 = linear(h2, n_neurons, name='layer3', activation=tf.nn.relu)
h4, W4 = linear(h3, n_neurons, name='layer4', activation=tf.nn.relu)
h5, W5 = linear(h4, n_neurons, name='layer5', activation=tf.nn.relu)

Y_pred, W6 = linear(h5, 3, activation=None, name='pred')


# We define our error as the squared difference between our true Y and our predicted, which gives us an absolute error metric. We sum all the error values for our three RGB values to get a single error value per image. We then define our cost as the mean summed error value across all images.

# In[64]:


error = tf.squared_difference(Y, Y_pred)
sum_error = tf.reduce_sum(input_tensor=error, axis=1)
cost = tf.reduce_mean(input_tensor=sum_error)


# Define the optimizer to use, giving it a learning rate and specifying what loss function (cost in this case) it should minimize. Specify number of iterations and batch size, which can be used to tune the model and get different varieties of final result. We also initialize the session.

# In[65]:


optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)
n_iterations = 300
batch_size = 50
sess = tf.Session()


# Check the shape of the image we chose to paint and store that shape in a variable.

# In[66]:


print(imgs.shape)
img_shape = imgs.shape


# Initialize all the variables we created. Create imgs list for storing resulting images during training. Loop through iterations, choosing random xs for training. Loop through batches splitting chosen xs into batches and train on that data, trying to determine what xs(row/column) coordinates should return what 3 RGB color values. 
# 
# If our current iteration is divisible by our display step, make a Y prediction(3 RGB color values) for current xs(row/column coordinates), clip those values to be between 0 and 1 and reshape it to be the same shape as our chosen input. We append that image to our "imgs" list variable which can later be used to access images produced during training.
# 
# If training takes long, decrease your network size/number of neurons, increase learning rate or decrease image resize in the second cell to something smaller. To get nice results at higher resolution requires larger initial images, larger number of neurons and longer training time usually.

# In[67]:


sess.run(tf.global_variables_initializer())

imgs = []
display_step = n_iterations // 10

for it_i in range(n_iterations):
    
    idxs = np.random.permutation(range(len(xs)))
    
    n_batches = len(idxs) // batch_size
    
    for batch_i in range(n_batches):
        
        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
        
        training_cost = sess.run([cost, optimizer],
                                feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]
    
    if (it_i + 1) % display_step == 0:
        
        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
        img = np.clip(ys_pred.reshape(img_shape), 0, 1)
        imgs.append(img)
        
        ax = plt.imshow(img)
        plt.title('Iteration {}'.format(it_i))
        plt.show()


# Using -1 we select the final image produced by our training process, we can use 0, 1 etc to view images from earlier in the training process.

# In[69]:


plt.figure(figsize=(8, 8))
plt.imshow(imgs[-1])


# We save the final image produced by our model as a .png

# In[70]:


plt.imsave(fname='my_catdog.png', arr=imgs[-1])


# Reference:
# https://github.com/pkmital/CADL/blob/master/session-2/lecture-2.ipynb

# In[ ]:




