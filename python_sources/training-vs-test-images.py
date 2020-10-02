#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel is a based on [this one](https://www.kaggle.com/gomezp/complete-beginner-s-guide-eda-keras-lb-0-93) (and the referenced ones).
# 
# It will look in detail at differences between the training and test data pixel value distributions and, in particular, very bright images.
# 
# # Highlights
# * The number of pixels with value 255 seems to matter
# * Training and test data have different numbers of very bright images
# * Mean image brightness is not as important

# In[ ]:


#Load necessary modules
import os, cv2
from scipy import stats
from glob import glob 
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook,trange
import matplotlib.pyplot as plt


# We will load 50k randomly selected images from the training and test set and compare them.

# In[ ]:


def load_data(N,df):
    """ This functions loads N images using the data df
    """
    # allocate a numpy array for the images (N, 96x96px, 3 channels, values 0 - 255)
    X = np.zeros([N,96,96,3],dtype=np.uint8) 
    #if we have labels for this data, also get them
    if 'label' in df.columns:
        y = np.squeeze(df.as_matrix(columns=['label']))[0:N]
    else:
        y = None
    #read images one by one, tdqm notebook displays a progress bar
    for i, row in tqdm_notebook(df.iterrows(), total=N):
        if i == N:
            break
        X[i] = cv2.imread(row['path'])
          
    return X,y


# In[ ]:


#set paths to training and test data
path = "../input/" #adapt this path, when running locally
train_path = path + 'train/'
test_path = path + 'test/'

df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))}) # load the filenames
df_test = pd.DataFrame({'path': glob(os.path.join(test_path,'*.tif'))}) # load the test set filenames
df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0]) # keep only the file names in 'id'
labels = pd.read_csv(path+"train_labels.csv") # read the provided labels
df = df.merge(labels, on = "id") # merge labels and filepaths
df.head(3) # print the first three entrys


# In[ ]:


#shuffle the dataframes to a representative sample
df = df.sample(frac=1,random_state = 42).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state = 4242).reset_index(drop=True)

# Load N images from the training set
N = 50000
print("Loading training data samples...")
X,y = load_data(N=N,df=df) 
print("Loading test data samples...")
X_test,_ = load_data(N=N,df=df_test) 
print("Done.")


# # Exploratory Data Analysis (EDA)
# 
# As we have no labels for the test set, this analysis will first focus on the raw image data.
# 
# In detail, we will now look at:
# * Mean image brightness distribution
# * Individual channel brightness distributions
# * Peculiar images such as very bright ones

# ## Mean image brightness distributions
# Let's start with the mean image brightness and its distribution in the training and test data.

# In[ ]:


nr_of_bins = 256
fig,axs = plt.subplots(1,2,sharey=True, sharex = False, figsize=(8,2),dpi=150)
training_brightness = np.mean(X,axis=(1,2,3))
test_brightness = np.mean(X_test,axis=(1,2,3))
axs[0].hist(training_brightness, bins=nr_of_bins, density=True)
axs[1].hist(test_brightness, bins=nr_of_bins, density=True)
axs[0].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(training_brightness),np.std(training_brightness)),),loc=2,prop={'size': 6})
axs[1].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(test_brightness),np.std(test_brightness)),),loc=2,prop={'size': 6})
axs[0].set_title("Mean brightness, training samples")
axs[1].set_title("Mean brightness, test samples")
axs[0].set_xlabel("Image mean brightness")
axs[1].set_xlabel("Image mean brightness")
axs[0].set_ylabel("Relative frequency")
axs[1].set_ylabel("Relative frequency");


# We can also look at a cumulative distribution.

# In[ ]:


fig = plt.figure(figsize=(6,3),dpi=150)
plt.hist(training_brightness, bins=nr_of_bins, density=True,cumulative=True, alpha = 0.5)
plt.hist(test_brightness, bins=nr_of_bins, density=True,cumulative=True, alpha = 0.5);
plt.legend(("Training","Test"),loc=2,prop={'size': 6})
# axs[1].legend(("Mean={:2.2f}|Std={:2.2f}".format(np.mean(test_brightness),np.std(test_brightness)),),loc=2,prop={'size': 6})
plt.title("Cumulative mean brightness, training vs test")
plt.xlabel("Image mean brightness")
plt.ylabel("Cumulative frequency")
plt.show()


# Overall, these look pretty similar, which is good as the test and training set seem to stem from, at least, a similar distribution in that regard. There seem to be a bit more images with a mean brightness of 140 - 180 in the test samples and more very bright ones in the training data. 
# 
# Let's have a more detailed look at the image data. We will now investigate the distribution of pixel values in each channel separately and all combined.

# In[ ]:


nr_of_bins = 256 #each possible pixel value will get a bin in the following histograms
N_hist = 10000 #we will limit the nr of images we look at, because otherwise this tends to kill the kernel...
fig,axs = plt.subplots(4,2,sharey=True,figsize=(8,8),dpi=150)

#RGB channels
axs[0,0].hist(X[0:N_hist,:,:,0].flatten(),bins=nr_of_bins,density=True)
axs[0,1].hist(X_test[0:N_hist,:,:,0].flatten(),bins=nr_of_bins,density=True)
axs[1,0].hist(X[0:N_hist,:,:,1].flatten(),bins=nr_of_bins,density=True)
axs[1,1].hist(X_test[0:N_hist,:,:,1].flatten(),bins=nr_of_bins,density=True)
axs[2,0].hist(X[0:N_hist,:,:,2].flatten(),bins=nr_of_bins,density=True)
axs[2,1].hist(X_test[0:N_hist,:,:,2].flatten(),bins=nr_of_bins,density=True)

#All channels
axs[3,0].hist(X[0:N_hist].flatten(),bins=nr_of_bins,density=True)
axs[3,1].hist(X_test[0:N_hist].flatten(),bins=nr_of_bins,density=True)

#Set image labels
axs[0,0].set_title("Training samples (N =" + str(X.shape[0]) + ")");
axs[0,1].set_title("Test samples (N =" + str(X_test.shape[0]) + ")");

axs[0,1].set_ylabel("Red",rotation='horizontal',labelpad=35,fontsize=12)
axs[1,1].set_ylabel("Green",rotation='horizontal',labelpad=35,fontsize=12)
axs[2,1].set_ylabel("Blue",rotation='horizontal',labelpad=35,fontsize=12)
axs[3,1].set_ylabel("RGB",rotation='horizontal',labelpad=35,fontsize=12)
for i in range(4):
    axs[i,0].set_ylabel("Relative frequency")
axs[3,0].set_xlabel("Pixel value")
axs[3,1].set_xlabel("Pixel value")
fig.tight_layout()


# These also look relatively similar. There is a difference in the green channels for high pixel values and, note, how both, the training and test set, feature a high number of pixels with values of 255. These are likely artifacts. 
# 
# Let's have a look at images, that have an unusual number of pixels with values of 255 in particular. They are likely outliers. For the training set, we can also inspect their labels to get more insight.
# 
# Just a brief reminder: In the [last kernel](https://www.kaggle.com/gomezp/complete-beginner-s-guide-eda-keras-lb-0-93) we found that about 60% of labels are negative overall.
# 
# We will now look at some basic statistics and plot some examples.
# 
# First off, let's check how many image have more than a specific number of pixels of value 255 in them and, for the training set, how many of those have positive labels.

# In[ ]:


#first count pixels with value 255 in train and test data per image
bright_pixels_train = (X == 255).sum(axis=(1,2,3))
bright_pixels_test = (X_test == 255).sum(axis=(1,2,3))


# In[ ]:


N_bright_train,N_bright_test,N_bright_positive_labels = [],[],[]
xtics = range(0,5000,100)
for threshold in xtics:
    #count images with more than threshold 255 pixels
    N_bright_train.append((bright_pixels_train > threshold).sum() / N) 
    N_bright_test.append((bright_pixels_test > threshold).sum() / N) 
    #count positive samples
    N_bright_positive_labels.append(y[bright_pixels_train > threshold].sum() / (bright_pixels_train > threshold).sum())
    
fig = plt.figure(figsize=(6,3),dpi=150)
plt.plot(xtics,N_bright_train)
plt.plot(xtics,N_bright_test)
plt.plot(xtics,N_bright_positive_labels)
plt.legend(("Training images over threshold","Test images over threshold","Positive samples portion (Training images)"),loc=1,prop={'size': 6})
plt.title("Frequency of bright pixels in training and test with positive rate for training")
plt.xlabel("Pixel Number Threshold (how many pixels have to be 255)")
plt.ylabel("Relative frequency")
plt.show()


# So, this is getting quite interesting. Let *B* be the number of pixels with value 255. Then, we can see:
# * There are more images in the training data than in the test data for ~300 < *B* < ~2000 relatively speaking
# * Unlike the rest of the training data, which has ~40% positive samples, images with *B* > 500 have about 32% positive rate only (in the training data)
# * Images with *B* > 1700 are increasingly likely to be labeled negative (in the training data)
# 
# Let's have a look at example images with *B* close to 1500 (1475 < *B* < 1500) to see, if they stand out in any way.

# In[ ]:


#let's take those images where between 1475 and 1525 pixels have values of 255
bright_train_imgs = X[np.logical_and(bright_pixels_train > 1475,bright_pixels_train < 1525)] 
bright_test_imgs = X_test[np.logical_and(bright_pixels_test > 1475,bright_pixels_test < 1525)]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(100) #we can use the seed to get a different set of random images
fig.suptitle("Images with 1475 < B < 1500 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image

#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# So, these look relatively normal to me (no expert though), but some seem to contain areas, where no tissue was (the white blobs afaik). Sometimes these are in the center, but the label was positive, that might be a mislabel. In general, these images should not break things, but one might want to clean the data. Let's look at cases with even brighter images (2400 < *B* < 2600) !

# In[ ]:


#let's take those images where between 2400 and 2600 pixels have values of 255
bright_train_imgs = X[np.logical_and(bright_pixels_train > 2400,bright_pixels_train < 2600)] 
bright_test_imgs = X_test[np.logical_and(bright_pixels_test > 2400,bright_pixels_test < 2600)]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(42) #we can use the seed to get a different set of random images
fig.suptitle("Images with 2400 < B < 2600 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image
    
#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# This is where I am mostly at a loss. These are much more likely to be negative samples, however to me they look relatively similar to the ones with *B* =~ 1500. Any experts weighing in on this would be very appreciated. 
# 
# We'll return to numbers to avoid being biased by these visual samples. Let's inspect how many pixels are actually 255 and if there are any broken images.

# In[ ]:


#calculate relative and absolute incidence of 255 pixels
nr_train = (X == 255).sum() / N
nr_test = (X_test == 255).sum() / N
freq_train = nr_train*100 / (96*96*3)
freq_test = nr_test*100 / (96*96*3)

print("Nr of pixels with value 255 per image \nTraining data - {:.4f}% ; avg. Nr = {:.0f}; maxval = {} \nTest - {:.4f}% ; avg. Nr = {:.0f}; maxval = {}".format(freq_train,nr_train,np.max(bright_pixels_train),freq_test,nr_test,np.max(bright_pixels_test)))


# So, there is a higher average number of very bright pixels in the training than in the test data.
# 
# We will now look at images that have a mean image brightness instead, not ones with many bright pixels. We will start with images that have mean brightness of more than 220.

# In[ ]:


#let's take those images with high mean values (> 220)
bright_train_imgs = X[np.mean(X,axis=(1,2,3)) > 220] 
bright_test_imgs = X_test[np.mean(X_test,axis=(1,2,3)) > 220]

#train
fig = plt.figure(figsize=(8, 5), dpi=100)
np.random.seed(100) #we can use the seed to get a different set of random images
fig.suptitle("Images with mean brightness over 220 \n Training samples (N =" + str(bright_train_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_train_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_train_imgs[idx]) #plot image
    ax.set_title('Label: ' + str(y[idx])) #show the label corresponding to the image
    
#test
fig = plt.figure(figsize=(8, 4), dpi=100)
fig.suptitle("Test samples (N =" + str(bright_test_imgs.shape[0]) + ")")
for plotNr,idx in enumerate(np.random.randint(0,bright_test_imgs.shape[0],8)):
    ax = fig.add_subplot(2, 4, plotNr+1, xticks=[], yticks=[]) #add subplots
    plt.imshow(bright_test_imgs[idx]) #plot image


# So, these look much more broken to my (non-expert) eye. In some cases the center region seems to be completely white and the first one of the displayed test images is defintely broken. Let's look at the labels for these images and how frequent these images are in the test and training set.

# In[ ]:


training_brightness = np.mean(X,axis=(1,2,3))
test_brightness = np.mean(X_test,axis=(1,2,3))

N_bright_train,N_bright_test,N_bright_positive_labels = [],[],[]
xtics = range(100,255,1)
for threshold in xtics:
    #count images with more than threshold 255 pixels
    N_bright_train.append((training_brightness > threshold).sum() / N) 
    N_bright_test.append((test_brightness > threshold).sum() / N) 
    #count positive samples
    N_bright_positive_labels.append(y[bright_pixels_train > threshold].sum() / (bright_pixels_train > threshold).sum())
    
fig = plt.figure(figsize=(6,3),dpi=150)
plt.plot(xtics,N_bright_train)
plt.plot(xtics,N_bright_test)
plt.plot(xtics,N_bright_positive_labels)
plt.legend(("Training images over threshold","Test images over threshold","Positive samples portion (Training images)"),loc=1,prop={'size': 6})
plt.title("Frequency of bright pixels in training and test with positive rate for training")
plt.xlabel("Pixel Number Threshold (how many pixels have to be 255)")
plt.ylabel("Relative frequency")
plt.show()


# So, once again, the distribution is slightly different. There is a higher portion of images with a high mean brightness in the training data it seems. Interestingly, however, unlike for the images with a high *B*, i.e. with many very bright pixels, images with a high mean brightness don't seem to have different distribution of positive samples.

# # Conclusions
# So, for *B* being the number of pixels with value 255 in an image
# 
# * Images with high *B* have a different positive/false distribution than the rest
# * But mean image brightness does not seem to have a strong relationship with the positve/false distribution
# * Images with high *B* and higher mean brightness are more often encountered in the training than in the test set
# 
# Overall, it seems there are quite some images with questionable quality in both datasets with very high mean image brightness
# Cleaning the data should help and a special treatment of the differently distributed images with high *B* might be warranted.
