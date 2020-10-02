#!/usr/bin/env python
# coding: utf-8

# Hello, I'm glad to be making this kernel for you guys. Hope you understand the steps I have laid out down below.
# 
# Anyone with a small experience in image processing, would be aware that the pictures can be represented in the form the pixel intensitites of each pixel.
# 
# ![](https://thumbs.dreamstime.com/b/nucleolus-hepatocyte-liver-cells-hepatocytes-seen-light-microscope-their-nuclei-show-very-large-stained-red-93292624.jpg)
# 
# When trying to detect nuclei in a histology picture, it is clear that the nuclei is darker in color than its surroundings. So, the pixel intensities value will differ at the nuclei and the cell space.

# Each row of the file contains pixel intensity values and the final column contains a '0' or a '1' which denotes whether there is any nuclei present in that specific picture.

# Import the necessary libraries as always.

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


# Create a dataframe by importing all the values from the csv file.
# This is the main step to perform at the beginning of every data science or machine learning project. This enables us to acess the data at a faster speed and also making sure the original file is safe and unchanged.

# In[ ]:


df = pd.read_csv('../input/data.csv')


# Use the .head() command to confirm the importing of data to the dataframe.

# In[ ]:


df.head()


# Now, you might think which machine learning algorithms to use. It depends on the ouput. Check the unique entries of the output column which is in question. 

# In[ ]:


list(set(df.Label))


# Here, the output is binary. It is clear that classification algorithms should be employed to solve this task. For this dataset, I'll be using Logistic Regression. You can implement SVM, RandomForest Classifier etc. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #to split the dataset for training and testing


# We have just imported the libraries required for Logistic Regression and splitting the dataset for training and testing purpose.

# Let's split the data into training and test datasets as it is important to measure the accuracy of our model with the dataset (test) it has not learned from. 

# In[ ]:


train , test = train_test_split(df, test_size = 0.3)


# Let's create training and testing I/O.

# In[ ]:


X_train = train.drop(["Label"] , axis = 1)
Y_train = train.Label

X_test = test.drop(["Label"] , axis = 1)
Y_test = test.Label


# Create and define the classifier 

# In[ ]:


Logistic_regressor = LogisticRegression()
Logistic_regressor.fit(X_train, Y_train)


# Now when the classifier is fitted with the training data, it has learned the pattern from it. We can go ahead and start predicting with the test dataset input. 

# In[ ]:


Logistic_prediction = Logistic_regressor.predict(X_test)


# We can measure the performance of the classifier by using accuracy and confusion matrix. So let's import the library for it.

# In[31]:


from sklearn.metrics import confusion_matrix 
from sklearn import metrics #for checking the model accuracy


# Let's find out the confusion matrix and accuracy.

# In[32]:


Logistic_cm = confusion_matrix(Y_test, Logistic_prediction)
Logistic_accuracy = metrics.accuracy_score(Y_test, Logistic_prediction)


# In[ ]:


print("The accuracy of the classifier is: {}".format(Logistic_accuracy))


# In[ ]:


print("The Confusion matrix of the classifier is: {}".format(Logistic_cm))


# Here we can see the classifier is performing at a very high accuracy. This can be imporved further by using different classification algorithms or ensemble methods.
# 
# Thank you and let me know how I can improve my explainations.
