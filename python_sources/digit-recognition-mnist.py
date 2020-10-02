#!/usr/bin/env python
# coding: utf-8

# # Disclaimer
# The data in this notebook is mostly copied from [https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification ](http://https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification). I intended to do modification later to the tutorial, so please permit me for using it. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Library Import
# For starter import any machine libary we wanted to use. SKLearn is good choice for beginner, the question is what the algorithm we interested to test. 
# Here's what we are going to need:
# 1. At least a classification algorithm (SVM or Decision Tree is a Good Choice)
# 2. Matplotlib
# 3. Preprocessing tools
# 4. Train test split
# And since we have been import numpy and panda no need to import them. 

# In[ ]:


import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
#%matplotlib inline


# # Load Data
# In case you haven't imported Digit Recognizer dataset from the competition, please do so. Then load the data with pandas. 
# For simplicity we'll only load first 5000 train images then split them again into our personal train & test set for testing.

# In[ ]:


# load the data
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)


# ### Q1
# Notice in the above we used _images.iloc?, can you confirm on the documentation? what is the role?

# In[ ]:


# now we gonna load the second image, reshape it as matrix than display it
i=1
img=train_images.iloc[i].values
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])


# ### Q2
# Now plot an image for each image class

# In[ ]:


# Todo: Put your code here


# Now plot the histogram within img

# In[ ]:


#train_images.iloc[i].describe()
#print(type(train_images.iloc[i]))
plt.hist(train_images.iloc[i])


# ### Q3
# Can you check in what class does this histogram represent?. How many class are there in total for this digit data?. How about the histogram for other classes

# In[ ]:


# create histogram for each class (data merged per class)
# Todo
#print(train_labels.iloc[:5])
data1 = train_images.iloc[1]
data2 = train_images.iloc[3]
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.append(data1,data2)
print(len(data3))
plt.hist(data3)


# ### Train the model
# Now we are ready to train the model, for starter let's use SVM. For the learning most model in SKLearn adopt the usual fit() and predict(). 

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ### Q4
# In above, did you see score() function?, open SVM.score() dokumentation at SKLearn, what does it's role?. Does it the same as MAE discussed in class previously?.
# Ascertain it through running the MAE. Now does score() and mae() prooduce the same results?. 

# In[ ]:


# Put your verification code here
# Todo
print(train_labels.values.ravel())
print(np.unique(test_labels)) # to see class number


# ### Improving Performance
# Did you noticed, that the performance is so miniscule in range of ~0.1. Before doing any improvement, we need to analyze what are causes of the problem?. 
# But allow me to reveal one such factor. It was due to pixel length in [0, 255]. Let's see if we capped it into [0,1] how the performance are going to improved.

# In[ ]:


test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].values.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])


# In[ ]:


# now plot again the histogram
plt.hist(train_images.iloc[i])


# # Retrain the model
# Using the now adjusted data, let's retrain our model to see the improvement

# In[ ]:


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)


# ### Q5
# Based on this finding, Can you explain why if the value is capped into [0,1] it improved the performance significantly?. 
# Perharps you need to do several self designed test to see why. 

# ### Prediction labelling
# In Kaggle competition, we don't usually submit the end test data performance on Kaggle. But what to be submitted is CSV of the prediction label stored in a file. 

# In[ ]:


# Test again to data test
test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])


# In[ ]:


# separate code section to view the results
print(results)
print(len(results))


# In[ ]:


# dump the results to 'results.csv'
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)


# In[ ]:


#check if the file created successfully
print(os.listdir("."))


# # Data Download
# We have the file, can listed it but how we are take it from sever. Thus we also need to code the download link. 

# In[ ]:


# from https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)


# # Q6
# Alhamdulillah, we have completed our experiment. Here's things to do for your next task:
# * What is the overfitting factor of SVM algorithm?. Previously on decision tree regression, the factor was max_leaf nodes. Do similar expriment using SVM by seeking SVM documentation!
# * Apply Decision Tree Classifier on this dataset, seek the best overfitting factor, then compare it with results of SVM. 
# * Apply Decision Tree Regressor on this dataset, seek the best overfitting factor, then compare it with results of SVM & Decision Tree Classifier. Provides the results in table/chart. I suspect they are basically the same thing. 
# * Apply Decision Tree Classifier on the same dataset, use the best overfitting factor & value.  But use the unnormalized dataset, before the value normalized to [0,1]
# 
# 

# 
