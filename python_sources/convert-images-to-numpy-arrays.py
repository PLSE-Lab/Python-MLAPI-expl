#!/usr/bin/env python
# coding: utf-8

# # Goal
# To do basic image analysis in this competition, it will be necessary to convert the TIFF images into more friendly array formats, especially if we want to make use of some of NumPy and other machine learning libraries with more advanced functions. This can help inform our decisions later, even if we opt to go with packages that can take TIFF files as input directly
# 
# Here, we'll convert the training images, training masks, and test images into compressed numpy array formats that can be easily loaded later.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt  # Just so we can visually confirm we have the same images
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load TIFF into NumPy array
# 
# Let's start by displaying the first training image, its mask, and the first test image so we know what they should look like.
# 
# I used the `viridis` colormap because it helps interpret the images visually because of its more dynamic color range. The yellow regions help us see where the images are saturated better.
# 
# As a demonstration, this cell also shows the difference between `matplotlib`'s standard `imshow` function and `matshow`. You can see the latter gives a more granular, seemingly noisier display. This is present in the original data and somewhat hidden by `imshow` because interpolation is done automatically, so I believe using imshow gives a more "real" visualization of the data and its noise. You can also achieve the same visual effect by adding `interpolation='none'` when calling `imshow`. Note that this is different from `interpolation=None`, which is `imshow`'s default.

# In[ ]:


train_image1 = plt.imread('../input/train/1_1.tif')
train_mask1 = plt.imread('../input/train/1_1_mask.tif')
test_image1 = plt.imread('../input/test/1.tif')

fig, ax = plt.subplots(2, 3)
for a in ax.reshape(6): a.axis('off')
fig.set_size_inches(16, 8)  # Just some magic numbers that looked nice on my display

ax[0, 0].imshow(train_image1, cmap='viridis')
ax[0, 1].imshow(train_mask1, cmap='Greys_r')
ax[0, 2].imshow(test_image1, cmap='viridis')
ax[1, 0].matshow(train_image1, cmap='viridis')
ax[1, 1].matshow(train_mask1, cmap='Greys_r')
ax[1, 2].matshow(test_image1, cmap='viridis')


# # Saving to compressed .npz file
# 
# We'll use `np.savez_compressed` to store our images as loadable numpy arrays. We'll make the key for each image the same as its filename, sans the *.tif* suffix. This can be done easily thanks to the fact that `.tif` is always the last four characters of a filename.
# 
# We should also decide what data type we would like to use. Because these are intensity images, and each value is a positive integer, it makes sense computationally that we create arrays of unsigned integers. `np.uint8` should suffice here. Fortunately, `plt.imread` is smart enough to figure this out for us.
# 
# **Executing the following cells will put a lot of data into your main memory. Make sure you have enough RAM to spare before creating your archives. Running this on Kaggle Scripts probably won't work since there isn't enough available memory.**

# In[ ]:


# Get the filenames, which will be used as our keys
training = get_ipython().getoutput('ls ../input/train')
test = get_ipython().getoutput('ls ../input/test')
training_names = [name[:-4] for name in training]
test_names = [name[:-4] for name in test]


# In[ ]:


# First let's save the training data, including masks
training_dict = {name:plt.imread('../input/train/'+f) for name, f in zip(training_names, training)}
np.savez_compressed('../input/training_data.npz', **training_dict)
del training_dict  # Just to keep memory usage low


# In[ ]:


# Now the test images
test_dict = {name:plt.imread('../input/test/'+f) for (name, f) in zip(test_names, test)}
np.savez_compressed('../input/test_data.npz', **test_dict)
del test_dict


# And now we have two compressed archives containing all the materials we need for testing and training. These archieves have an *.npz* suffix. Not only does this give a bit more convenience for accessing the data and working with scientific python libraries, these archives are actually smaller than the zipped archives provided by Kaggle. You'll need to run this on your local machine for this to show the file sizes.

# In[ ]:


get_ipython().system('ls ../input/*.zip -s -h')


# In[ ]:


get_ipython().system('ls ../input/*.npz -s -h')


# # Loading the compressed .npz files
# 
# Working with these archives is a lot like opening other kinds of files. The major difference is that all of the data is in the same archive file. As such, when loaded, the archive functions like a dictionary and is indexed by the keys we assigned above.
# 
# At this point, this may sound like a terrible thing to do, because during training and testing we shouldn't need to load every image at once. This would be very taxing on computers with less memory. We can get around this by using memory-mapping, conveniently handled for us by the NumPy `load` function. When reading in the archive, we supply the `mmap_mode='r'` keyword-argument. When this argument is specified, instead of reading the entire file into memory, the archive will be kept *on disk,* allowing us to access the chunks that we want (i.e. the individual images) without actually loading them all into memory at once. Neat, huh?
# 
# To prove it all works, let's plot the same training image, training mask, and test image we started with.

# In[ ]:


# Training data
with np.load('data/training_data.npz', mmap_mode='r') as data:
    # Note that the keys are the names we created above (i.e. filename sans .tif suffix)
    train_image1, train_mask1 = data['1_1'], data['1_1_mask']

# Test data
with np.load('data/test_data.npz', mmap_mode='r') as data:
    test_image1 = data['1']

# Display the images
fig, ax = plt.subplots(1, 3)
for a in ax: a.axis('off')
fig.set_size_inches(20, 10)  # Just some magic numbers that looked nice on my display

ax[0].matshow(train_image1, cmap='viridis')
ax[1].matshow(train_mask1, cmap='Greys_r')
ax[2].matshow(test_image1, cmap='viridis')


# Note: I'm not entirely sure the `mmap_mode='r'` argument is required to take advantage of memory mapping, because when I run the `np.load` command without it, my memory profiler doesn't show a 900+ MB file being loaded into memory.
