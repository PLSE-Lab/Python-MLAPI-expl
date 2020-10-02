#!/usr/bin/env python
# coding: utf-8

# # **Importing**

# In[67]:


import sys
import os
import subprocess
import gc

from six import string_types

# Make sure you have all of these packages installed, e.g. via pip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy

from sklearn.metrics import accuracy_score
from skimage import io
from scipy import ndimage
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Setup
# Set `PLANET_KAGGLE_ROOT` to the proper directory where we've got the TIFF and JPEG zip files, and accompanying CSVs.

# In[68]:


ROOT = os.path.abspath("../input/")
REPRODUCE_ROOT =  os.path.join(ROOT, "reproduce-results-of-report/")
REPRODUCE_OUTLIER_CALIBRATION=(os.path.join(REPRODUCE_ROOT, "outliers1.csv"))
REPRODUCE_SAMPLESETS=(os.path.join(REPRODUCE_ROOT, "data_sample1.csv"))
PLANET_KAGGLE_ROOT = os.path.join(ROOT, "planet-understanding-the-amazon-from-space/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')

assert os.path.exists(REPRODUCE_OUTLIER_CALIBRATION)
assert os.path.exists(REPRODUCE_SAMPLESETS)
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)


# To obtain the results mentioned in the report the following data has been used. These data sets were randomly generated or a result of an earlier computation. 

# In[69]:


# Data used to reproduce the results mentioned in the report
SAMPLE_IMAGE_CALIBRATION = "train_33143.tif"
OUTLIER_REFERENCE_LIST = pd.read_csv(REPRODUCE_OUTLIER_CALIBRATION)['0']


# # Data inspection

# In[70]:


#printing the number of tif files in the data set
tifs = 0
for elem in os.listdir(os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, "train-tif-v2"))): 
    if(".tif" in elem): tifs += 1
print(tifs)


# In[71]:


# Reading the provided data set.
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

# Getting the number of entries and attributes in the dataset
print(labels_df.shape)
# Checking for missing values
print(labels_df.any() == np.nan )


# In[72]:


# Build list with unique labels
label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

# The number of unique labels in the data set.
print(len(label_list))
# The data set contains the following unique labels.
print(label_list)


# In[73]:


# Building a list of all atmospheric labels
weather_label_list = [ 'haze', 'clear', 'cloudy', 'partly_cloudy'] 


# In[74]:


def check_labels(verbose = 0):
    #Get the list of all tags of each image
    labels_imgs = labels_df['tags'].apply(lambda x: np.asarray(x.split(' ')))

    #Check if each image has at least one weather label.
    outliers = []
    err = 0
    for i in range(len(labels_imgs)):
        if sum([j in weather_label_list for j in labels_imgs[i]]) ==0:
            err +=1
            outliers.append(i)
    if(err>0 and verbose == 1):
        print("Not all labels have one weather label!")
        print("There are "+str(err)+" images that don't have one weather label!")
    return outliers

o = check_labels()


# # preprocessing

# In[75]:


# Add onehot features for every label
for label in label_list:
    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Display head of the data frame to see how it now looks.
labels_df.head()


# In[76]:


#Remove all image that don't have any weather labels
outliers = check_labels(verbose = 1)
print("The following images don't have at least one weather label:")
print(labels_df.iloc[outliers])
labels_df = labels_df.drop(outliers)
print("Removed "+str(len(outliers))+" outliers.")


# In[77]:


# Histogram of label instances
plt.figure()
plt.title("Histogram of labels occurneces ")
labels_df[label_list].sum().sort_values(ascending=False).plot.bar()


# In[78]:


# Function to plot a cooccurrence matrix. 
def make_cooccurence_matrix(labels):
    numeric_df = labels_df[labels]; 
    c_matrix = numeric_df.T.dot(numeric_df)
    sns.heatmap(c_matrix)
    return c_matrix


# In[79]:


# Each image should have exactly one weather label:
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
make_cooccurence_matrix(weather_labels)


# In[80]:


# All cloudy images should only have the label cloudy:
a = make_cooccurence_matrix(label_list)

# creating a output file to import into the report later on
np.savetxt("mydata.csv", a, delimiter=' & ',  newline=' \\\\\n')


# ## Inspect images
# Let's display an image and visualize the pixel values. Here we will pick an image, load every single single band, then create RGB stack. These raw images are 16-bit (from 0 to 65535), and contain red, green, blue, and [Near infrared (NIR)](https://en.wikipedia.org/wiki/Infrared#Regions_within_the_infrared) channels. In this example, we are discarding the NIR band just to simplify the steps to visualize the image. However, you should probably keep it for ML classification.
# 
# The files can be easily read into numpy arrays with the skimage.

# In[81]:


# Randomly sample n images with the specified tags. 
# If no tags are specified than just sample over all images.
def sample_images(tags, n=None):
    condition = True
    if (tags == []):
        return labels_df.sample(n)
    if isinstance(tags, string_types):
        raise ValueError("Pass a list of tags, not a single tag.")
    for tag in tags:
        condition = condition & labels_df[tag] == 1
    if n is not None:
        return labels_df[condition].sample(n)
    else:
        return labels_df[condition]


# In[82]:


# returns the file of the image if present
def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    for dirname in os.listdir(PLANET_KAGGLE_ROOT):
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))
        if os.path.exists(path):
            #print('Found image {}'.format(path))
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))

# Given a dataframe of sampled images, get the corresponding filename
def sample_to_fname(sample_df, row_idx, suffix='tif'):
    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
    return '{}.{}'.format(fname, suffix)


# In[83]:


# Plots the color histogram of an image.
def plot_rgbn_histo(r, g, b, n):
    for slice_, name, color in ((r,'r', 'red'),(g,'g', 'green'),(b,'b', 'blue'), (nir, 'nir', 'magenta')):
        plt.hist(slice_.ravel(), bins=100, 
                 range=[0,rgb_image.max()], 
                 label=name, color=color, histtype='step')
    plt.legend()


# In[84]:


# return a rgb image and all individual bands of the given image
def get_rgb_r_g_b_nir(bgrn_image):
    # extract the rgb values
    bgr_image = bgrn_image[:,:,:3]
    rgb_image = bgr_image[:, :, [2,1,0]]

    # extract the different bands
    b, g, r, nir = bgrn_image[:, :, 0], bgrn_image[:, :, 1], bgrn_image[:, :, 2], bgrn_image[:, :, 3]
    
    return rgb_image, r, g, b, nir


# In[85]:


# Plot the bands indifidualy
def plot_r_g_b_nir(r, g, b, nir):
    fig = plt.figure()
    fig.set_size_inches(12, 4)
    for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):
        a = fig.add_subplot(1, 4, i+1)
        a.set_title(c)
        plt.imshow(x)


# Let's look at an individual image. First, we'll plot a histogram of pixel values in each channel. Note how the intensities are distributed in a relatively narrow region of the dynamic range

# In[86]:


# get a random image
s = sample_images(['water','road'], n=1)
# get the file name of this image
# fname = sample_to_fname(s, 0)

# To reproduce the same results as in the report uncomment the following line
fname = SAMPLE_IMAGE_CALIBRATION

# find the image in the data directory and load it
# note the initial bgrn band ordering
bgrn_image = load_image(fname)

rgb_image, r, g, b, nir = get_rgb_r_g_b_nir(bgrn_image)
# plot a histogram of rgbn values
plot_rgbn_histo(r, g, b, nir)


# We can look at each channel individually:

# In[87]:


plot_r_g_b_nir(r, g, b, nir)


# But, when we try to look at the RGB image, something funny's going on!|

# In[88]:


plt.imshow(rgb_image)


# ## Calibrate colors for visual inspection

# This image cann't be viewed by us as the colors are not yet calibrated. This is not a problem for analytic purposes, but we can try some transformations to make the image look better for visual inspection as is explained in the report. 
# 
# One way of doing this is to normalize the image channels to a reference color curve. We'll show here how to estimate a reference color curve from other normalized images. We could choose a third party aerial image of a canopy , but here we will employ the JPEG images provided in the data set, which have already been color-corrected.  
# 
# In essence, the idea is to transform the pixel values of the test image so that their average and variance match the reference image data.
# 
# Get a list of reference images to extract data from:

# In[ ]:


# Pull a list of 100 random jpeg image names
jpg_list = np.random.choice(np.asarray(os.listdir(PLANET_KAGGLE_JPEG_DIR)), 100, replace=False)

#Read each image (8-bit RGBA) and dump the pixels values to ref_colors, which contains buckets for R, G and B
ref_colors = [[],[],[]]
for _file in jpg_list:
    # keep only the first 3 bands, RGB
    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]
    # Flatten 2-D to 1-D
    _data = _img.reshape((-1,3))
    # Dump pixel values to aggregation buckets
    for i in range(3): 
        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()
    
ref_colors = np.array(ref_colors)

#Compute the mean and variance for each channel in the reference data
ref_means = [np.mean(ref_colors[i]) for i in range(3)]
ref_stds = [np.std(ref_colors[i]) for i in range(3)]

del jpg_list
q = gc.collect()


# In[ ]:


# A function that can calibrate any raw image reasonably well:
def calibrate_image(rgb_image):
    # Transform test image to 32-bit floats to avoid 
    # surprises when doing arithmetic with it 
    calibrated_img = rgb_image.copy().astype('float32')

    # Loop over RGB
    for i in range(3):
        # Subtract mean 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]-np.mean(calibrated_img[:,:,i])
        # Normalize variance
        calibrated_img[:,:,i] = calibrated_img[:,:,i]/np.std(calibrated_img[:,:,i])
        # Scale to reference 
        calibrated_img[:,:,i] = calibrated_img[:,:,i]*ref_stds[i] + ref_means[i]
        # Clip any values going out of the valid range
        calibrated_img[:,:,i] = np.clip(calibrated_img[:,:,i],0,255)

    # Convert to 8-bit unsigned int
    return calibrated_img.astype('uint8')


# Visualize the color histogram of the newly calibrated test image, and note that it's more evenly distributed throughout the dynamic range, and is closer to the reference data.

# In[ ]:


test_image_calibrated = calibrate_image(rgb_image)
for i,color in enumerate(['r','g','b']):
    plt.hist(test_image_calibrated[:,:,i].ravel(), bins=30, range=[0,255], 
             label=color, color=color, histtype='step')
plt.legend()
plt.title('Calibrated image color histograms')


# And now we have something we can recognize!

# In[ ]:


plt.imshow(test_image_calibrated)


# ## Outlier detection
# Now that we have a way to construct a rgb image from a tif file we can start the outlier detection. 
# As mentioned in the report we will get the least similar image from a list of images as this is most likely an outlier.
# We will then visualy inspect this image if it indeed is an outlier. If so we will use this image as a reference for the outlier detection. 
# To find the least similar image we will make use of a solution proposed by Paul Smith

# In[ ]:


# A way of computing the mean image from a sample of 600 images.
# Proposed solution by Paul Schmidt

import cv2

"""
ALTERNATE THE NUMBER OF N
"""
n_imgs = 500
all_imgs = []

# get n_imgs of random images 
# fnames = labels_df['image_name'].sample(n_imgs).as_matrix()

fnames = OUTLIER_REFERENCE_LIST

print("Getting "+str(n_imgs)+" images...")
for i in range(n_imgs):
    img = load_image( fnames[i]+".tif")
    img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR).astype('float')
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float')
    img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img = img.reshape(1, -1)
    all_imgs.append(img)

img_matrix = np.vstack(all_imgs)
all_imgs = None

from scipy.spatial.distance import pdist, squareform
#Computing distances to each of the images im a efficient manner
sq_dists = squareform(pdist(img_matrix))
print("All done")

mask = np.zeros_like(sq_dists, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# upper triangle of matrix set to np.nan
sq_dists[np.triu_indices_from(mask)] = np.nan
sq_dists[0, 0] = np.nan

fig = plt.figure(figsize=(6,4))
# maximal dissimilar image
maximal_dissimilar_image_idx = np.nanargmax(np.nanmean(sq_dists, axis=1))
print(fnames[maximal_dissimilar_image_idx]+".tif")

outlier_ref = load_image(fnames[maximal_dissimilar_image_idx]+".tif")

rgb, r, g, b, nir = get_rgb_r_g_b_nir(outlier_ref)
plt.imshow(calibrate_image(rgb))
plt.title('Maximal dissimilar image')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("Image is considered to be reference for outliers.")

pd.DataFrame(fnames).to_csv("outlier_detection.csv",index = False)
del all_imgs, mask
gc.collect();


# In[ ]:


# Finding the right threshold for similarity to find all outliers. 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity as coss
# Load all the images in the set used earlier
images = np.zeros((len(fnames), (256* 256* 4)) )
for i in range(len(images)):
        images[i] = load_image(fnames[i]+".tif").flatten() #flatten is needed to get the similarity measure later on
        
# Function that eliminates outliers based on the value of THRESHOLD
def eliminate_outliers (dataset, labelset=None, verbose=1):
    if verbose: print("calculating the simillarities...")
    sims = coss(np.expand_dims(outlier_ref.flatten(), axis=0),dataset)
    a = len(dataset)
    if verbose: print("Size of set before removing outliers: "+ str(a))
    
    #eliminating all images that are classified as outliers
    outlier_mask= (np.where(sims[0]>=THRESHOLD)[0])
    outliers = dataset[outlier_mask]
    not_outliers = dataset[list(set(range(len(dataset)))- set(outlier_mask))]
    if(not(labelset is None)):
        not_outliers_labels = labelset[list(set(range(len(dataset)))- set(outlier_mask))]
    if verbose: 
        print("Size of set after removing outliers: "+str(len(not_outliers)))
        print("Eliminated " +str(len(outliers))+" outliers from the set")
        print("These are the outliers removed from the dataset with threshold="+str(THRESHOLD)+" :")
        for i in range(len(outliers)):
            rgb, r, g, b, nir = get_rgb_r_g_b_nir(outliers[i].reshape( 256, 256, 4))
            plt.imshow(calibrate_image(rgb))
            plt.title('Outlier '+str(i))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        
    del sims, outliers, dataset
    gc.collect()
    if(labelset is None):
        return not_outliers
    return not_outliers, not_outliers_labels

#Eliminating all outliers from the sets
THRESHOLD = 0.95
eliminate_outliers(images)

#Eliminating all outliers from the sets
THRESHOLD = 0.990
eliminate_outliers(images)

#Eliminating all outliers from the sets
THRESHOLD = 0.995
eliminate_outliers(images)

#Eliminating all outliers from the sets
THRESHOLD = 0.999
eliminate_outliers(images)

del images
q = gc.collect()


# In[ ]:


# Setting Threshold value
THRESHOLD = 0.995


# In[ ]:


# Function to get a train and test set
def  get_train_and_test_set(size_of_train_set, size_of_test_set, verbose=0):
    
    # To reproduce the same results in the report
    samples = pd.read_csv(REPRODUCE_SAMPLESETS)
    # To get a random sample of the data
    print("Getting a training set of "+str(size_of_train_set)+" images and test set of "+str(size_of_test_set)+" images.")
    #samples = labels_df.sample(size_of_train_set + size_of_test_set)
    
    fnames = samples['image_name'].as_matrix()
    labels = samples[label_list].as_matrix()
    X_train_fnames, X_test_fnames = np.split(fnames, [size_of_train_set])
    y_train, y_test = np.split(labels, [size_of_train_set])
    
   
    X_train = np.zeros((size_of_train_set,(256*256*4)))
    for i in range(size_of_train_set):
        X_train[i] = load_image(X_train_fnames[i]+".tif").flatten() #flatten is used in order to normalize the data later on
    X_test = np.zeros((size_of_test_set,(256*256*4)))
    for i in range(size_of_test_set):
        X_test[i] = load_image(X_test_fnames[i]+".tif").flatten() #flatten is used in order to normalize the data later on

    print("Normalizing the data...")
    if(verbose):
        samples.to_csv("data_sample.csv")
    
    #normalization of the data
    X_train = normalize(X_train, norm="l2")
    X_test = normalize(X_test,norm="l2")
    
    # Reshape the data back to it's original shape
    train = np.zeros((len(X_train),256,256,4))
    for i in range(len(X_train)):
        train[i] = X_train[i].reshape(256,256,4)
    # Reshape the data back to it's original shape
    test = np.zeros((len(X_test),256,256,4))
    for i in range(len(X_test)):
        test[i] = X_test[i].reshape(256,256,4)
    print("All done")
    

    return train, y_train, test, y_test

X_train, y_train, X_test, y_test = get_train_and_test_set(2000,500,verbose=1)
X_train, y_train = eliminate_outliers(X_train, y_train,verbose=0)
X_test, y_test = eliminate_outliers(X_test, y_test,verbose=0)
q = gc.collect()


# # k-NN

# In[ ]:


# Flatten the data before fitting the classifier to it
X_train_flat = np.zeros((len(X_train),(256*256*4)))
for i in range(len(X_train)):
    X_train_flat[i] = X_train[i].flatten()

# Flatten the data before fitting the classifier to it   
X_test_flat = np.zeros((len(X_test),(256*256*4)))
for i in range(len(X_test)):
    X_test_flat[i] = X_test[i].flatten()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# The number of neigbohrs to compute the accuracy for
ks = np.arange(5,51,5)

scores_e = []
scores_c = []

print("Fitting the classifier...")
for k in ks:
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean") 
    print(k)
    print("euclidean")
    clf.fit(X_train_flat[:1000], y_train[:1000])
    y_pred = clf.predict(X_test_flat[:250])
    scores_e.append(accuracy_score(y_test[:250], y_pred))
    
    clf = KNeighborsClassifier(n_neighbors=k, metric="cityblock") 
    print("cityblock")
    clf.fit(X_train_flat[:1000], y_train[:1000])
    y_pred = clf.predict(X_test_flat[:250])
    scores_c.append(accuracy_score(y_test[:250], y_pred))

print("Accuracy with euclidean distance measure:")
print(scores_e)
print("Accuracy with cityblock distance measure:")
print(scores_c)

plt.figure()
plt.title("Plot of the accuracy over all labels as a function of k using calibrated images")
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(ks, scores_e, label='euclidean')
plt.plot(ks, scores_c, label='cityblock')
plt.legend()
plt.show()
print("Figure: Plot of the accuracy over all labels as a function of k using calibrated images using the k nearest neighbors with k on the x-axis and the accuracy on the y-axis")


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=5, metric="cityblock") 
clf.fit(X_train_flat, y_train)
y_pred = clf.predict(X_test_flat)
acc = accuracy_score(y_test, y_pred)

print(acc)


# # Extra trees classifier

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

# Please note that the flattend data is used!

# Determine the number of estimators to use with a small part of the data

# The numbers of estimators to compute the accuracy for
ks = np.arange(5,51,5)

scores_g = []
scores_e = []

print("Fitting the classifier...")
for k in ks:
    print(k)
    print("gini")
    clf = ExtraTreesClassifier(criterion='gini',n_estimators=k) 
    clf.fit(X_train_flat[:1000], y_train[:1000])
    y_pred = clf.predict(X_test_flat[:250])
    scores_g.append(accuracy_score(y_test[:250], y_pred[:250]))
    
    print("entropy")
    clf = ExtraTreesClassifier(criterion='entropy',n_estimators=k) 
    clf.fit(X_train_flat[:1000], y_train[:1000])
    y_pred = clf.predict(X_test_flat[:250])
    scores_e.append(accuracy_score(y_test[:250], y_pred[:250]))

print("Accuracy with gini:")
print(scores_g)
print("Accuracy with entropy:")
print(scores_e)
    
plt.figure()
plt.title("Plot of the accuracy over all labels as a function of k using calibrated images")
plt.xlabel('k')
plt.ylabel('accuracy')
plt.plot(ks, scores_g, label='gini')  
plt.plot(ks, scores_e, label='entropy')
plt.legend()
plt.show()
print("Figure: Plot of the accuracy over all labels as a function of k using calibrated images using the extra trees classifier with k on the x-axis and the accuracy on the y-axis")


# In[ ]:


clf = ExtraTreesClassifier(criterion='gini',n_estimators=35) 
clf.fit(X_train_flat, y_train)


# In[ ]:


y_pred = clf.predict(X_test_flat)
print(accuracy_score(y_test, y_pred))


# # CNN 

# In[ ]:


from keras.models  import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam

#Setting up the model
chanDim=-1
model = Sequential()

#Creatin the input layer
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256,256,4)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

#Creating the hidden layers
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#Creating the output layer
# use a *softmax* activation for single-label classification
# and *sigmoid*  activation for multi-label classification
model.add(Dense(17))
model.add(Activation("sigmoid"))

#compiling the model to bring it all together
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4, decay=1e-4 / 75), metrics=["accuracy"])
print("Compiled the model")


# In[ ]:


# Fitting the model to the data with shape (256,256,4)
model.fit(X_train, y_train, epochs = 3)


# In[ ]:


# save the prodictions based on the rgbnir images
y_pred_rgbn = model.predict(X_test)
score, acc = model.evaluate(X_test, y_test)
print(acc)


# # Optimize the solution

# In[ ]:


# Crwating a new model as the input shape for NDVI and NDWI is (256,256,1)
#Setting up the model
chanDim=-1
model_NDWI_NDVI= Sequential()

#Creatin the input layer
model_NDWI_NDVI.add(Conv2D(32, (3, 3), padding="same", input_shape=(256,256,1)))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization(axis=chanDim))
model_NDWI_NDVI.add(MaxPooling2D(pool_size=(3, 3)))
model_NDWI_NDVI.add(Dropout(0.25))

#Creating the hidden layers
model_NDWI_NDVI.add(Conv2D(64, (3, 3), padding="same"))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization(axis=chanDim))
model_NDWI_NDVI.add(Conv2D(64, (3, 3), padding="same"))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization(axis=chanDim))
model_NDWI_NDVI.add(MaxPooling2D(pool_size=(2, 2)))
model_NDWI_NDVI.add(Dropout(0.25))
 
# (CONV => RELU) * 2 => POOL
model_NDWI_NDVI.add(Conv2D(128, (3, 3), padding="same"))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization(axis=chanDim))
model_NDWI_NDVI.add(Conv2D(128, (3, 3), padding="same"))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization(axis=chanDim))
model_NDWI_NDVI.add(MaxPooling2D(pool_size=(2, 2)))
model_NDWI_NDVI.add(Dropout(0.25))

model_NDWI_NDVI.add(Flatten())
model_NDWI_NDVI.add(Dense(1024))
model_NDWI_NDVI.add(Activation("relu"))
model_NDWI_NDVI.add(BatchNormalization())
model_NDWI_NDVI.add(Dropout(0.5))

#Creating the output layer
# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
model_NDWI_NDVI.add(Dense(17)) #we want one label as output
model_NDWI_NDVI.add(Activation("sigmoid"))

#compiling the model to bring it all together
model_NDWI_NDVI.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4, decay=1e-4 / 75), metrics=["accuracy"])
print("Compiled the model")


# ### NDVI images 

# In[ ]:


# Function to create the NDVI image of an image
def NDVI(image):
    get_rgb_r_g_b_nir(image)
    NDVI = (nir-r)/(nir+r)
    NDVI = np.expand_dims(NDVI,axis=2)    
    return NDVI

# transform all images in the data set to NDVI images.
NDVI_train = np.zeros((len(X_train),256,256,1))
for i in range(len(X_train)):
    NDVI_train[i] = NDVI(X_train[i])
    
NDVI_test =np.zeros((len(X_test),256,256,1))
for i in range(len(X_test)):
    NDVI_test[i] = NDVI(X_test[i])


# In[ ]:


model_NDWI_NDVI.fit(NDVI_train, y_train, epochs = 3)


# In[ ]:


y_pred_NDVI = model_NDWI_NDVI.predict(NDVI_test)
score, acc = model_NDWI_NDVI.evaluate(NDVI_test, y_test)


# In[ ]:


acc_primary = accuracy_score(y_test[:,1],([int(x>0.5) for x in y_pred_NDVI[:,1]]))
acc_agriculture = accuracy_score(y_test[:,2],([int(x>0.5) for x in y_pred_NDVI[:,2]]))
acc_cultivation = accuracy_score(y_test[:,7],([int(x>0.5) for x in y_pred_NDVI[:,7]]))
print("Training the model with the NDVI images results in an overal accuracy of: "+str(acc))
print("Training the model with the NDVI images results in an accuracy of: "+str(acc_primary) + " for the primary label")
print("Training the model with the NDVI images results in an accuracy of: "+str(acc_agriculture) + " for the argiculture label")
print("Training the model with the NDVI images results in an accuracy of: "+str(acc_cultivation) + " for the cultivation label")


del NDVI_train, NDVI_test
q = gc.collect()


# ### NDWI images

# In[ ]:


# NDWI images

# Function to create the NDWI image of an image
def NDWI(image):
    get_rgb_r_g_b_nir(image)
    NDWI = (nir-r)/(nir+r)
    NDWI = np.expand_dims(NDWI,axis=2) 
    return NDWI

# transform all images in the data set to NDVI images.
NDWI_train = np.zeros((len(X_train),256,256,1))
for x in range(len(X_train)):
    NDWI_train[i] = NDWI(X_train[i])
NDWI_test = np.zeros((len(X_test),256,256,1))
for x in range(len(X_test)):
    NDWI_test[i] = NDWI(X_test[i])


# In[ ]:


model_NDWI_NDVI.fit(NDWI_train, y_train, epochs = 3)


# In[ ]:


y_pred_NDWI = model_NDWI_NDVI.predict(NDWI_test)
score, acc = model_NDWI_NDVI.evaluate(NDWI_test, y_test)

acc_water = accuracy_score(y_test[:,1],([int(x>0.5) for x in y_pred_NDWI[:,4]]))
print("Training the model with the NDVI images results in an overal accuracy of: "+str(acc))
print("Training the model with the NDVI images results in an accuracy of: "+str(acc_water) + " for the water label")

del NDWI_train, NDWI_test, X_train, X_test
q = gc.collect()


# ### Data generator

# In[ ]:


# data generator to fit all data to the CNN

def generator(fnames, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    #batch_images = np.empty((batch_size, 256, 256, 4))
    batch_labels = np.empty((batch_size,len(label_list)))
    while True:
        indexes = np.random.choice(range(len(fnames)), batch_size, replace=False)
        batch_images = np.asarray([load_image(str(fnames[k])+".tif").flatten() for k in indexes])
        batch_labels = labels[indexes]

        #for i in range(batch_size):
            # choose random index in features
        #    batch_features[i] = (features[index])
        #    batch_labels[i] = labels[index]
        #    index+=1
        batch_images, batch_labels = eliminate_outliers(batch_images, batch_labels, verbose=0)
        res_imgs = []
        for i in batch_images:
            res_imgs.append(i.reshape(256,256,4))
        yield np.asarray(res_imgs), batch_labels


# In[ ]:


X = labels_df['image_name'].as_matrix()
y = labels_df[label_list].as_matrix()

train_size = np.floor(len(X)*0.8)
X_train, X_test = np.split(X, [int(train_size)])
y_train, y_test = np.split(y, [int(train_size)])


# In[ ]:


training_generator = generator(X_train, y_train, 64)
model.fit_generator(generator=training_generator, steps_per_epoch=np.floor(len(X_train)/64), nb_epoch=1)


# In[ ]:


test_generator = generator(X_test, y_test, 64)
score, acc = model.evaluate_generator(generator=test_generator,steps=np.floor(len(X_test)/64), verbose=1)
print(acc)


# ### Augmenting the data

# In[ ]:


# data generator to fit all data to the CNN

def generator_aug(fnames, labels, batch_size):
    y = 0
    while True:
        indexes = fnames[y:y+(int(batch_size/4))]
        batch_images = np.asarray([load_image(str(k)+".tif").flatten() for k in indexes])
        batch_labels = labels[y:y+(int(batch_size/4))]
        """
        for i in range(batch_size):
            # choose random index in features
            batch_features[i] = (features[index])
            batch_labels[i] = labels[index]
            index+=1"""
        batch_images, batch_labels = eliminate_outliers(batch_images, batch_labels, verbose=0)
        res_imgs = []
        res_labels = []
        for i in batch_images:
            res_imgs.append(i.reshape(256,256,4))
            res_imgs.append(scipy.ndimage.rotate(i.reshape(256,256,4),90))
            res_imgs.append(scipy.ndimage.rotate(i.reshape(256,256,4),180))
            res_imgs.append(scipy.ndimage.rotate(i.reshape(256,256,4),270))
        for i in batch_labels:
            res_labels.append(i)
            res_labels.append(i)
            res_labels.append(i)
            res_labels.append(i)
        yield np.asarray(res_imgs),np.asarray(res_labels)

X = labels_df['image_name'].as_matrix()
y = labels_df[label_list].as_matrix()

train_size = np.floor(len(X)*0.8)
X_train, X_test = np.split(X, [int(train_size)])
y_train, y_test = np.split(y, [int(train_size)])


# In[ ]:


training_generator = generator_aug(X_train, y_train, 64)
model.fit_generator(generator=training_generator, steps_per_epoch=np.floor(len(X_train)*4/64), nb_epoch=1)


# In[ ]:


test_generator = generator(X_test, y_test, 64)
score, acc = model.evaluate_generator(generator=test_generator,steps=np.floor(len(X_test)/64), verbose=1)
print(acc)


# END
# 
