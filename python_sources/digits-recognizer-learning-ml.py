#!/usr/bin/env python
# coding: utf-8

# # Digits Recognizer
# ## Introduction
# This kernel is a humble start to learn the beautiful world of machine learning. The dataset doesn't require much introduction for individuals who have been introduced to machine learning. My intention here is not to describe the dataset. More information can be obtained online. A quick reading can be done on wikipedia - https://en.wikipedia.org/wiki/MNIST_database.
# 
# But no efforts are worth without a citation to Modified National Institute of Standards and Technology for this wonderful dataset. This kernel will initally focus on how to do a first pass of prediction using knn and I shall try to develop the kernel along the way. Request the reader to correct or point out my mistakes and provide suggestions that could help to improve the kernel. 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # personal choise for viz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('List of files in the input directory:')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Set seaborn
sns.set()


# ## Load data and inspect it
# We see that there are 2 datasets in csv format. As is always the case, the train dataset has labels and we need to predict the labels for the test dataset. Let us try to import into a pandas dataset and have a look at the structure of the data.

# In[2]:


# Load train and test dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[3]:


# Inspect the shape of the datasets
print('Shape of train dataset - Rows: %d; Columns: %d' % train.shape)
print('Shape of test dataset - Rows: %d; Columns: %d' % test.shape)


# There is one column less in the train dataset which is the column that needs to be predicted. Let us now look at the column names of these datasets.

# In[4]:


print('Training dataset columns:')
print(train.columns)
print('\nTesting dataset columns:')
print(test.columns)


# The first column in the training dataset is the label and followed by 784 columns viz. pixel0, pixel1, ... pixel783. The testing dataset is the same except that it doesn't have the label column which is the way it should be. Else there is no fun. :)
# 
# Lets us print out a few records of the training and testing datasets for 8 arbitrary columns to visually examine the content in the pixel columns.

# In[5]:


print('8 arbitrary columns of first 5 records in the training dataset:')
print(train[['label', 'pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].head())
print('8 arbitrary columns of first 5 records in the testing dataset:')
print(test[['pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].head())


# Let us look at the values of these columns with the statistical summaries to understand more.

# In[6]:


print('Training dataset - describe:')
print(train[['label', 'pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].describe())
print('\nTesting dataset - describe:')
print(test[['pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].describe())


# The label column ranges from 0 to 9. But it doesn't give us clue whether all digits are represented in the training dataset. We need to analyze that. The pixel values range from 0 to 255. There are some pixels that have only 0 values which could be the case depending on whether that pixel stores any handwriting on that pixel. A couple of things that we need to look out for 
# 1. Are there any NaN values in the datasets
# 1. Are all the digits represented in the training dataset

# In[7]:


# Are there any NaNs?
print('No. of null values in the training dataset: %d' % np.sum(np.sum(train.isnull())))
print('No. of null values in the testing dataset: %d' % np.sum(np.sum(test.isnull())))


# In[8]:


sns.countplot(train['label'])
plt.show()


# So good news. No NaNs and all digits are represented fairly in the training dataset.

# ## So what the heck is pixels?
# A good exercise is always to read descriptions in detail before you jump into the problem. Below are some details about the data provided as part of the competition.
# 
# ***Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.***
# 
# ***The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.***
# 
# ***Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).***
# 
# Should have read this initially. Anyways, lets reshape some of these images to 28 x 28 and visualize it using pyplot.imshow() and see if it relates to the label. There is no reason why it shouldn't. :)

# In[9]:


# pick 8 random digits in training dataset
np.random.seed(12345)
idx = np.random.randint(0, len(train), 8)

for n, i in enumerate(idx):
    ax = plt.subplot(2, 4, n+1)
    tmp = train.iloc[i]
    lbl = tmp['label']
    tmp = tmp.drop(['label'])
    tmp = tmp.values.reshape(28, 28)
    ax.imshow(tmp, cmap='gray', interpolation='gaussian')
    ax.set_title('Label: %d' % lbl)
    ax.axis('off')

plt.tight_layout()
plt.show()


# That's cool. We picked up 8 random digits from the training dataset. We then reshaped the digits to 28 x 28 matrix and plotted them with the titles as their corresponding labels. And all in all it matches perfectly fine. Let's do the same for the test dataset and see how it goes. Offcourse in this case we do not have the labels.

# In[10]:


# pick 4 random digits in testing dataset
np.random.seed(12345)
idx = np.random.randint(0, len(test), 4)

for n, i in enumerate(idx):
    ax = plt.subplot(2, 2, n+1)
    tmp = test.iloc[i]
    tmp = tmp.values.reshape(28, 28)
    ax.imshow(tmp, cmap='gray', interpolation='gaussian')
    ax.set_title('Label: ???')
    ax.axis('off')

plt.tight_layout()
plt.show()


# So that's 0, 2, 2 and 3. But our job is not plot it out and see the values and then predict. But hope the above illustrated how the pixels are reshaped and makes sense when plotted.

# ## Let's do some learning now
# I am assuming at this point we do not need any data cleansing or feature engineering.
# 
# *"Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. Feature engineering is fundamental to the application of machine learning, and is both difficult and expensive. The need for manual feature engineering can be obviated by automated feature learning."* - https://en.wikipedia.org/wiki/Feature_engineering
# 
# So let's jump straight onto training a model using knn. We shall look are hyperparameter tuning using GridSearchCV.

# In[11]:


# import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# split the training dataset to check for model performance
X = train.drop(['label'], axis=1)
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

knn = KNeighborsClassifier(n_neighbors=5)
_ = knn.fit(X_train, y_train)
print('The accurancy of the model: %f' % knn.score(X_test, y_test))


# The model is built and the score is a decent one for the first pass of ML.
# 
# ## Submission file
# The submission file should be a csv file with the below format. 
# 
# |ImageId|Label|
# |------|------|
# |1|3|
# |2|7|
# |3|8|
# 
# So we need to run the prediction on the dataset provided and save it as csv file. Let us it as submission.csv for submission.

# In[26]:


# make prediction on the set provided
submitLabel = knn.predict(test)
submitLabelDF = pd.DataFrame({'ImageId': range(1, len(submitLabel) + 1), 'Label': submitLabel})

# Let us print the first few lines of the submission file
print(submitLabelDF.head())


# In[29]:


# write the submission file
submitLabelDF.to_csv('submission.csv', index=False)


# ## Conclusion
# Hope you enjoyed the kernel and was useful. Please post your comment for any feedbacks or questions. It will be great to know your feedback. Also please upvote if you liked the kernel.
