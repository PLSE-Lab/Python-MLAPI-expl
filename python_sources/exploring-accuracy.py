#!/usr/bin/env python
# coding: utf-8

# # Exploring Accuracy
# 
# I didn't see many people experimenting with the accuracy so I was curious if I could find anything from it. My goal was to find some sort of relationship between accuracy and the variation in x and y. The purpose of this was so that I could eventually make a way to artificially populate the data set with 'ghosts' of the same data point in a localized radius (obtained from this relationship between accuracy and distance). Then use either KNN/KDTree or something similar to [ZFTurbo's method](https://www.kaggle.com/zfturbo/facebook-v-predicting-check-ins/mad-scripts-battle/code). 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data = pd.read_csv('../input/train.csv')


# In[ ]:


mean_train_data = train_data.groupby('place_id').mean()
std_train_data = train_data.groupby('place_id').std()

acc_df = pd.concat([mean_train_data['accuracy'],std_train_data['x'],std_train_data['y']], axis=1)
acc_df.rename(columns={'accuracy':'mean_accuracy','x':'std_x','y':'std_y'}, inplace=True)

acc_df.fillna(0, inplace=True)


# In[ ]:


p = plt.hist(acc_df.mean_accuracy, bins=np.arange(min(acc_df.mean_accuracy), max(acc_df.mean_accuracy) + 1, 1))
plt.xlabel('Mean Accuracy')
plt.ylabel('Count')
plt.title('Counts of mean accuracy for places')

plt.show()


# In[ ]:


p = plt.hist(acc_df.std_x, bins=np.arange(min(acc_df.std_x), max(acc_df.std_x) + .01, .01))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(x) for places')

plt.show()


# In[ ]:


p = plt.hist(acc_df.std_y, bins=np.arange(min(acc_df.std_y), max(acc_df.std_y) + .001, .001))
plt.xlabel('Std(y)')
plt.ylabel('Count')
plt.title('Counts of std(y) for places')

plt.show()


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place')

plt.show()


# Some outliers in this graph but for the most part this is about what was expected, higher accuracy generally leads to lower deviation in distance and the opposite for lower accuracy. The variation in this data is larger than what would be ideal, which could cause some distance approximations to be very inaccurate but it is a start.

# In[ ]:


p = plt.scatter(acc_df.mean_accuracy, acc_df.std_y)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs mean accuracy for each place')

plt.show()


# This looks better an more promising but that is also to be expected since there is less variation among y data

# [Reference for below](http://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing)

# In[ ]:


#Exponential Decay function
def func(x, a, b, c):
    return a*np.exp(-b*(x+c))

popt, pcov = scipy.optimize.curve_fit(func, acc_df.mean_accuracy, acc_df.std_x)
popt


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func(i, 1.09251753e+00,   4.33988866e-03,   2.86223906e+01) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (exp decay)')

plt.show()


# In[ ]:


#csch = 1/sinh and 
def func2(x,a,b,c):
    return a*1/(np.sinh(b*x)) + c

popt, pcov = scipy.optimize.curve_fit(func2, acc_df.mean_accuracy, acc_df.std_x)
popt


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func2(i, 108.75643807,   40.73096679,    0.67921053) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (csch)')

plt.show()


# In[ ]:


#1/x
def func4(x,a,b,c):
    return a*(1/(b*(x+c)))

popt, pcov = scipy.optimize.curve_fit(func4, acc_df.mean_accuracy, acc_df.std_x)
popt


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_x)
l = plt.plot([i for i in range(0,1000)], [func4(i, 97.59074952,   0.8234309 ,  95.34280786) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(x)')
plt.title('std(x) vs Mean accuracy for each place (1/x)')

plt.show()


# This wasn't what I was hoping to find though the predictions would be conservative for the accuracies with larger std(x) which I would prefer rather than overpredicting the distance. Perhaps someone better versed in numpy and/or statistics can come up with a better model.

# I wanted to make some attempt on the y data:

# In[ ]:


popt, pcov = scipy.optimize.curve_fit(func, acc_df.mean_accuracy, acc_df.std_y)
popt


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_y)
l = plt.plot([i for i in range(0,1000)], [func(i, -0.01277189,  1.4337799 ,  2.71943575) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs Mean accuracy for each place (exp decay)')

plt.show()


# In[ ]:


popt, pcov = scipy.optimize.curve_fit(func4, acc_df.mean_accuracy, acc_df.std_y)
popt


# In[ ]:


p = plt.scatter(acc_df.mean_accuracy,acc_df.std_y)
l = plt.plot([i for i in range(0,1000)], [func(i, 221.24964964,   12.02131292,  804.60004683) for i in range(0,1000)], 'r', linewidth=1)
plt.xlabel('Mean Accuracy')
plt.ylabel('Std(y)')
plt.title('std(y) vs Mean accuracy for each place (1/x)')

plt.show()


# Not what I was hoping for either but I may use some of these equations for now to get started on generating artificial data points. 

# In[ ]:




