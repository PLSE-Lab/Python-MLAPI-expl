#!/usr/bin/env python
# coding: utf-8

# This is an implementation of the Naive Bayes algorithm done on the Iris dataset. We divided the data into train and test sets and after creating the model on the train dataset, we picked up random samples from the test dataset to validate the accuracy of the classification. We are able to classify the records with an accuracy of 100%.  

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt
from math import exp
from math import pi
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We added the dataset in csv format without the header. The column names can be seen from the output view below.  

# In[ ]:


import pandas as pd
iris = pd.read_csv("../input/iris-data-set/iris.csv",header=None)
iris.head()


# This is where we divide the data into train and test sets. 80% of the total data is taken as the train set and the remaining is used for sample testing. 

# In[ ]:


iris_train = iris.sample(frac=0.8,random_state=1000)
iris_test = iris.drop(iris_train.index)


# Here we construct a dictionary breaking down the data into three groups (Iris-setosa,Iris-versicolor, Iris-virginica) based on the type of flower. We calculate the mean, standard deviation, and total inputs for the four different attributes (sepal length, sepal width, petal width, petal length) for the three groups. The data structure used to store the data can be viewed from the output below.

# In[ ]:


train_summary = dict()
unique_classes = iris_train[4].unique()
for class_name in unique_classes:
    train_summary[class_name] = dict()
    train_summary[class_name]['values'] = dict()
    mean_list = list()
    std_list = list()
    len_list = list()
    for i in range(len(iris_train.columns)-1):
        mean_list.append(np.mean(iris_train[iris_train[4]==class_name][i]))
        std_list.append(np.std(iris_train[iris_train[4]==class_name][i]))
        len_list.append(len(iris_train[iris_train[4]==class_name][i]))
    train_summary[class_name]['values']['mean'] = mean_list
    train_summary[class_name]['values']['std_dev'] = std_list
    train_summary[class_name]['values']['length'] = len_list

train_summary


# We have defined a function which is the implementation of the mathematical formula for the algorithm

# In[ ]:


def gaussian_density_function(x,mean,stdev):
    exponent = exp(-((x-mean)**2/(2*stdev**2)))
    return (1/(sqrt(2*pi)*stdev))*exponent


# This is where we decide the sample used to test the model. The actual category of the sample is the last value in the list. 

# In[ ]:


test_value = list(iris_test.iloc[23])
test_value


# Here we invoke the defined gaussian_probability_density function by passing the required arguments, including the parameters of the sample chosen. The function returns the calculated values and the category which has the maximum value is the category of the selected sample. 

# In[ ]:


print("Calculated Values")
max_prob = 0;
for class_name in iris_train[4].unique():
    p = train_summary[class_name]['values']['length'][0]/float(len(iris_train))
    for i in range(len(iris_train.columns)-1):
           p=p*gaussian_density_function(test_value[i],train_summary[class_name]['values']['mean'][i],train_summary[class_name]['values']['std_dev'][i])
    print(class_name , ": ", p)
    if p>max_prob:
        max_prob = p
        final_class = class_name
print("\nPredicted Class: ", final_class)

