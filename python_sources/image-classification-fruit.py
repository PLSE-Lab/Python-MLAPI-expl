#!/usr/bin/env python
# coding: utf-8

# 
# ## Intro
# 
# The fruits 360 dataset on Kaggle contains many images of fruit. I wanted to see if I could use Python to read in these images and build an algorithm to correctly label the fruit in each image. I have previously created a similar kernel on the MNIST fashion dataset, but this had already processed the images into dataframes for modelling. I wanted to see if instead I could use the actual images and run this processing myself.
# 
# Whilst researching this task, I struggled to find a kernal/tutorial which outlined this whole process. Hence I hope the below will act as a tutorial for anyone else wanting to learn how to do this. 
# 
# ## Outline of process
# 
# #### 1. Image processing
# 
# - Like any machine learning model, I need features (X variables) and a label to predict (Y variable). With image classification, the same concept applies. However, here, the features are each pixel in the image. The images I am going to read in are 100x100 pixels. Hence, there are 10,000 pixels in every image; each of these pixels will be a feature in my model. Our first step therefore is to read in each image, capturing the value of every pixel as our 10,000 X variables, and recording the image's label for our Y variable. If we do this for each image, we have created our dataset. 
# 
# - However, colour images like the fruit data aren't stored as a single 100X100 array. They are actually stored as three arrays; one for the value of Red, Blue and Green in the image. Hence, another stage in our processing of these images involves transforming the images from colour to grayscale. By doing this, we can get a single value for each pixel, which indicates how black/white this pixel is. We can store this as a value between 0 and 1. 
# 
# #### 2. Machine learning
# - Once we have transformed our images into grayscale and created our dataframe of 10,000 X variables with Y labels, we can begin our machine learning element. I will split my dataframe into a training and test set, building my algorithm on the training set, before seeing how well it performs on the test set. This enables me to understand how well my model will perform on images of fruit it has not seen before.
# 
# 
# ### 1. Image processing - reading in images
# 
# Firstly, I load in the libraries I will need, and take a look at the folder structure. The fruit 360 dataset contains pre defined training and test splits, divided into two folders. Instead of using these, I am going to read in fruit images from the training folders, and create my own train/test splits later.

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

os.listdir('../input/fruits-360_dataset/fruits-360/Training')[0:10]


# In the training folder, there is a folder for each fruit, containing images of that fruit. 
# 
# We can read in one of these images to take a look, using cv2.

# In[ ]:


path = '../input/fruits-360_dataset/fruits-360/Training/Clementine/206_100.jpg'

im = cv2.imread(path)
plt.imshow(im)


# At first to me, it seemed strange this fruit was blue as it is supposed to be a clementine! I then realised that cv2 reads images as Blue Green Red, when they actually need to be in the form Red Green Blue to plot them. Hence, we can rearrange this order and plot again.

# In[ ]:


b,g,r = cv2.split(im)

im2 = cv2.merge([r,g,b])

plt.imshow(im2)


# Much more like a clementine! I can now change this image to grayscale.

# In[ ]:


grayim = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)

plt.imshow(grayim,cmap = 'gray')


# So how is this image stored in terms of data? Essentially each pixel in this grayscale image is a data point, with the value indicating how black or white the pixel is. The image is 100x100 pixels and hence we have 10,000 data points per image. We can take a look at this. 

# In[ ]:


grayim


# In[ ]:


grayim.shape


# 255 indicates a fully white pixel. We can normalise the pixel values so we get values between 0 and 1, with 1 being a fully white pixel and 0 being a black pixel.

# In[ ]:


grayim = grayim.astype('float')/255
grayim


# So we can now flatten our 100x100 array into a single series of 10,000 datapoints.

# In[ ]:


grayim = grayim.flatten()
grayim = pd.Series(grayim)
grayim


# Each of these pixels is going to be a feature in our dataset. Hence we need to repeat this process for all the images to build our dataset of X variables, and record the fruit label for our Y variable.
#  
# To do this, I am going to write a loop which reads into each fruit folder, extracting each image in turn and running through the above process. I will then also create a label vector indicating what fruit is in the image. I am doing this as a tutorial, so I am just going to go through this process for the first ten fruits in the folder to speed up running time. 

# In[ ]:


path = '../input/fruits-360_dataset/fruits-360/Training/'

cols = np.arange(grayim.shape[0])
df = pd.DataFrame(columns = cols)
labelcol = []

fruitlist = os.listdir(path)
x = 0

for f in fruitlist[0:9] : 
    fruitpath = '%s%s' % (path,f)
    
    imagelist = os.listdir(fruitpath)
    
    for i in imagelist:
        imagepath = '%s/%s' % (fruitpath,i)
    
        image = cv2.imread(imagepath)
    
        b,g,r = cv2.split(image)
        image = cv2.merge([r,g,b])
    
        imagegray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
        imagegray = imagegray.astype('float')/255
    
        imagegray = imagegray.flatten()
    
        df.loc[x] = imagegray
    
        x = df.shape[0] + 1
        labelcol.append(f)
    


# In[ ]:


df['label'] = labelcol
df


# I now have a dataframe with each row representing a single fruit image, and the 10K columns holding the values of each pixel in the image. We have also added on a column of labels, indicating what fruit is shown in the image. 

# Let's see what types of fruit we have and what proportion they make up in our dataset.

# In[ ]:


df['label'].value_counts(normalize = True)


# The classes are not that well balanced, which could affect my model's performance. However, as this is a tutorial on image processing I am going to leave them as they are for now, but I could do some work to improve this balance if I wanted to build on this kernel later. 
# 
# At the moment, my dataset is ordered by fruit type (due to how I created the dataset in the loop). Before moving onto the learning algorithm, we should shuffle our dataset. In the modelling part of the kernel, I am going to be dividing my data up into a training and test set, and hence want to shuffle the data before doing this in case this ordering impacts the divide. 

# In[ ]:


df = shuffle(df).reset_index(drop = True)
df


# I should also shuffle the columns in case there is any pattern here also. 

# In[ ]:


# transpose data set and shuffle
df_t = shuffle(df.transpose())
# transpose back to normal
df = df_t.transpose()


# ### 2. Modelling
# 
# We can now train a model to see if we can label images of fruit. 
# 
# - The first step is to split the dataframe into training and test sets, so we can build the model on the training data, and then test how well it performs on new data (test set). 
# - The next step will be to fit the model. There are a number of models which could be used here, but I am going to use Kernalised Support Vectors and Random Forests, comparing their performance. 
# 
# First of all, let's split the data into X and Y variables and create the training and test sets, as we will need this for both models.

# In[ ]:


from sklearn.model_selection import train_test_split

# Create X and Y variables
X = df.drop('label',axis = 1)
y = df['label']

# create our test and training set
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,stratify = y)


# #### Model 1 - SVM
# 
# Let's fit our model, using the default settings for SVM. We can print out both the training and test scores. 

# In[ ]:


from sklearn.svm import SVC

svm_model = SVC().fit(X_train,y_train)

trainscore = svm_model.score(X_train,y_train)
testscore = svm_model.score(X_test,y_test)
print('Training score: {:.3f}\nTest score: {:.3f}'.format(trainscore,testscore))


# Pretty good!
# 
# Now let's compare the SVM performance to a Random Forest. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 5).fit(X_train,y_train)

train = rf.score(X_train,y_train)
test = rf.score(X_test,y_test)

print('Training score: {:.3f}\nTest Score: {:.3f}'.format(train,test))


# Random forest appears to perform better. 
# 
# 
# If we wanted to extend this further, we could now perform gridsearch to see if we could tune the parameters to improve our fit further. We could also use more complex models like Neural Networks. I could also look at the confusion matrices for each model. 
