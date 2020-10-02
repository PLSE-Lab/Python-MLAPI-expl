#!/usr/bin/env python
# coding: utf-8

# This notebook does the following:
#  1. Extracts the data from the files, 
#  2. Gives some insight on image size,
#  3. Resizes the images so they can be used as input for NNs, 
#  4. Saves the results and drinks a coffee

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[ ]:


TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'


# In[ ]:


# On the kaggle notebook
# we only take the first 2000 from the training set
# and only the first 1000 from the test set
# REMOVE [0:2000] and [0:1000] when running locally
train_image_file_names = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)][0:2000] 
test_image_file_names = [TEST_DIR+i for i in os.listdir(TEST_DIR)][0:1000]


# In[ ]:


# Slow, yet simple implementation with tensorflow
# could be rewritten to be much faster
# (which is not really needed as it takes less than 5 minutes on my laptop)
def decode_image(image_file_names, resize_func=None):
    
    images = []
    
    graph = tf.Graph()
    with graph.as_default():
        file_name = tf.placeholder(dtype=tf.string)
        file = tf.read_file(file_name)
        image = tf.image.decode_jpeg(file)
        if resize_func != None:
            image = resize_func(image)
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()   
        for i in range(len(image_file_names)):
            images.append(session.run(image, feed_dict={file_name: image_file_names[i]}))
            if (i+1) % 1000 == 0:
                print('Images processed: ',i+1)
        
        session.close()
    
    return images


# In[ ]:


train_images = decode_image(train_image_file_names)
test_images = decode_image(test_image_file_names)
all_images = train_images + test_images


# In[ ]:


# Check mean aspect ratio (width/height), mean width and mean height
width = []
height = []
aspect_ratio = []
for image in all_images:
    h, w, d = np.shape(image)
    aspect_ratio.append(float(w) / float(h))
    width.append(w)
    height.append(h)


# In[ ]:


print('Mean aspect ratio: ',np.mean(aspect_ratio))
plt.plot(aspect_ratio)
plt.show()


# **Having aspect ratio vary so much is not good :( , lets take a closer look...**

# In[ ]:


print('Mean width:',np.mean(width))
print('Mean height:',np.mean(height))
plt.plot(width, height, '.r')
plt.show()


# **Some images are horizontally stretched, others are vertically stretched.** If you are to crop them to particular size WxH, then W/H should be around 1.15 ( the mean aspect ratio). Yet, I decided to go for the long shot and not crop at all...

# In[ ]:


print("Images widther than 500 pixel: ", np.sum(np.array(width) > 500))
print("Images higher than 500 pixel: ", np.sum(np.array(height) > 500))


# If you use all the data locally, you will see there are only two images in the training set that are bigger than 500x500...

# **Instead of cropping I will use padding and resize all images to 500x500... Which will make models run much slower (then cropping to i.e. 64x64), yet be more accurate, as no information is lost or distorted...**

# In[ ]:


# Free up some memory
del train_images
del test_images
del all_images


# In[ ]:


WIDTH=500
HEIGHT=500
resize_func = lambda image: tf.image.resize_image_with_crop_or_pad(image, HEIGHT, WIDTH)


# In[ ]:


processed_train_images = decode_image(train_image_file_names, resize_func=resize_func)
processed_test_images = decode_image(test_image_file_names, resize_func=resize_func)


# In[ ]:


# Chech the shapes
print(np.shape(processed_train_images))
print(np.shape(processed_test_images))


# In[ ]:


# Let's check how the images look like
for i in range(10):
    plt.imshow(processed_train_images[i])
    plt.show()


# In[ ]:


def create_batch(data, label, batch_size):
    i = 0
    while i*batch_size <= len(data):
        with open(label+ '_' + str(i) +'.pickle', 'wb') as handle:
            content = data[(i * batch_size):((i+1) * batch_size)]
            pickle.dump(content, handle)
            print('Saved',label,'part #' + str(i), 'with', len(content),'entries.')
        i += 1


# In[ ]:


# Create one hot encoding for labels
labels = [[1., 0.] if 'dog' in name else [0., 1.] for name in train_image_file_names]


# In[ ]:


# TO EXPORT DATA WHEN RUNNING LOCALLY - UNCOMMENT THIS LINES
# a batch with 5000 images has a size of around 3.5 GB
#create_batch(labels, 'data/train_labels', 5000)
#create_batch(processed_train_images, 'data/train_images', 5000)
#create_batch(processed_test_images, 'data/test_images', 5000)


# **All you need now is powerful cluster of servers that can handle so much data :)**

# In[ ]:




