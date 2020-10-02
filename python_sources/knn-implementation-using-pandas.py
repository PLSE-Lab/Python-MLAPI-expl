#!/usr/bin/env python
# coding: utf-8

# In this experiment, we will use Wisconsin Breast Cancer data to detect malignant cells.
# 
# ### Data Source
# 
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
# 
# The data has been modified:
# 
# The id field has been removed
# The diagnosis field has been moved to the end

# ### Data Attributes
# Number of instances: 569
# 
# Number of attributes: 31 (diagnosis, 30 real-valued input features)
# 
# Ten real-valued features are computed for each cell nucleus:
# 
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)
# The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 1 is Mean Radius, field 11 is Radius SE, field 21 is Worst Radius. All feature values are recoded with four significant digits.
# 
# The last field is diagnosis: M for Malignant and B for Benign
# 
# Class distribution: 357 benign, 212 malignant

# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


import pandas as pd
CANCERDATA = "../input/DS_WDBC_NOIDFIELD.data"
td = pd.read_csv(CANCERDATA, header=None)
td.head()


# Implement the euclidean distance using math and collection libraries

# In[ ]:


import math
import collections
def dist(a, b):
    sqSum = 0
    for i in range(len(a)):
        sqSum += (a[i] - b[i]) ** 2
    return math.sqrt(sqSum)
# ------------------------------------------------ #
# We are assuming that the label is the last field #
# If not, munge the data to make it so!            #
# ------------------------------------------------ #
def kNN(k, train, given):
    distances = []
    for t in train:
        distances.append((dist(t[:-1], given), t[-1]))
    distances.sort()
    #print(distances[:k])
    return distances[:k]

def kNN_classify(k, train, given):
    tally = collections.Counter()
    for nn in kNN(k, train, given):
        tally.update(nn[-1])
    return tally.most_common(1)[0]


# In[ ]:


wdbc = pd.read_csv(CANCERDATA, header=None)
wdbc.values


# But where are the test data? In practice, as a designer of the algorithm you are given only one set of data. The real test data is with your teacher/examiner/customer. The standard way is to create a small validation data from your training data and use it for evaluating the performance and also for parameter tuning. Let us randomly split the data into 80:20 ratio. We will use 80% for training and the rest 20% for evaluating the performance.

# In[ ]:


import random
TRAIN_TEST_RATIO = 0.8
train = []
test = []
data = wdbc.values
for one in data:
    if random.random() < TRAIN_TEST_RATIO:
        train.append(one)  
    else:
        test.append(one)


# In[ ]:


print(kNN_classify(5, train, test[0])[0])

print(kNN_classify(5, train, test[0])[0], test[0][-1])
print(kNN_classify(5, train, test[4])[0], test[4][-1])


# In[ ]:


results=[]
for i,t in enumerate(test):
  results.append(kNN_classify(5,train,t)[0] == test[i][-1])
print(results.count(True), "are correct")
print(results.count(True)/len(test))


# In[ ]:




