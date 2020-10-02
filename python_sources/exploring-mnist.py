#!/usr/bin/env python
# coding: utf-8

# # Exploring MNIST
# Jake Lee, TA for COMS4701 Fall 2019
# 
# ## Introduction
# In this notebook, we're going to play around with the MNIST dataset a little so that we can understand exactly what we're working with.
# 
# Let's start by importing some basic packages and the dataset itself.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

# These paths are unique to Kaggle, obviously. Use your local path or colab path, depending on which you're using.
train_x = np.load('/kaggle/input/f2019-aihw7/mnist-train-images.npy')
train_y = np.load('/kaggle/input/f2019-aihw7/mnist-train-labels.npy')
val_x = np.load('/kaggle/input/f2019-aihw7/mnist-val-images.npy')
val_y = np.load('/kaggle/input/f2019-aihw7/mnist-val-labels.npy')

# Verify that their shapes are what we expect
print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
print("val_x shape:", val_x.shape)
print("val_y shape:", val_y.shape)


# ## Visualizing Some Numbers
# What's the point of classifying images if we don't know what the images look like? Let's plot some out:

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

# show each image, and make each title the label
# these are grayscale images so use appropriate heatmap
ax1.imshow(train_x[4701], cmap=plt.get_cmap('gray'))
ax1.set_title(str(train_y[4701]))
ax2.imshow(train_x[4702], cmap=plt.get_cmap('gray'))
ax2.set_title(str(train_y[4702]))
ax3.imshow(train_x[4703], cmap=plt.get_cmap('gray'))
ax3.set_title(str(train_y[4703]))

fig.show()


# Cool! We've confirmed that the images look like what we expect, and that the labels match up. You might be surprised that the images seem inverted - that is, it's white strokes on a black background instead of black ink on a white background. This makes more sense if you think about it in terms of pixel values - 255 (white) means there's information there, 0 (black) means it's the background. Some classifiers might care about this, others might not.
# 
# Speaking of pixel values, there are two ways to represent said values - **np.int** (from 0 to 255) or **np.float** (from 0 to 1). Which one is this one?

# In[ ]:


# print data type
print("Data type:", train_x.dtype)
# just to make sure, print the min/max too
print("Data min:", np.amin(train_x[4701]))
print("Data max:", np.amax(train_x[4701]))


# Great, we see that it's uint8, ranging from 0 to 255. Keep this in mind - some classifiers **require** that the data be normalized from 0 to 1, or with the mean at 0 with a stddev of 1. The most important part is to stay **consistent**. Your training, validation, and test data all need to be normalized the same way.
# 
# What about the labels?

# In[ ]:


print("Data type:", train_y.dtype)


# The labels are also integers - you'll probably end up one-hot encoding this anyways, so this doesn't matter much. See the Sklearn tutorial for more information.
# 
# ## Class Distribution
# 
# When performing classification, the makeup of your training data is very important. For instance, if you have 1 image of a dog and 100 images of a cat, no matter how complex your model is, it probably won't do a good job. A balanced class distribution (where each class has an equal number of training examples) is ideal.
# 
# So, is the MNIST training set balanced? We can just look at the labels to find out:

# In[ ]:


fig, ax = plt.subplots()
ax.hist(train_y, bins=range(11))
ax.set_xticks(range(10))
ax.set_title("MNIST Training Set Class Distribution")

fig.show()


# Great, looks like we won't have to worry about that.
# 
# ## Conclusion
# 
# That's all you need to get started! There are some other great descriptions of MNIST out there, here are some of my favorites:
# 
# - https://colah.github.io/posts/2014-10-Visualizing-MNIST/
# - http://varianceexplained.org/r/digit-eda/
# 
# Finally, a fixture in every public notebook: if you enjoyed the writeup, click below to upvote!
