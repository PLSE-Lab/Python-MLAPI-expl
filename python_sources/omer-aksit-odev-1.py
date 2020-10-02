#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   # visualisation library
import seaborn as sns             # visualisation library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[ ]:


data.info()   #examining the columns in general


# In[ ]:


data.head(10)  # examining first 10 rows


# In[ ]:


pd.set_option('display.max_columns', 1000)   #changing the default setting of hiding the middle columns


# In[ ]:


print(data.loc[0,:])    
#examining the first row to get an insight about how the independent variables could have affected the
#dependant variable. Unfortunately, I could not get anything out of this. There are too many variables and their values
#do not seem meaningful at all, probably due to the pca transformation which was used to transform the original data.


# In[ ]:


data.describe()   #I tried to get a general insight by examining the statistical values of columns. Most of them do not
                  #seem meaningful, so I decided to examine Time, Amount and Class columns.


# In[ ]:


data[['Time','Amount','Class']].describe()   #examining certain columns


# In[ ]:


data.corr()   #examining the correlations between variables


# In[ ]:


# finding significant correlation values

# Here I could not print which columns have those correlation values on x-axis. Yet we can see the pairs 
# should be 'Amount-V7' and 'Amount-V20'.
# Also I could not find how to change the values of data2 permanently.
data2=data.corr()
for index,row in data2.iterrows():
    for each in row:
        if(each >= 0.3 and each!=1):
            print(index,each)
        else:
            pass
    


# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),linewidth=0.5)
plt.show()


# There is significant correlation between Amount and V7, and also between Amount and V20. Let's take a closer look at them.

# In[ ]:


plt.plot(data.Amount, color='r', label='Amount',alpha=0.8)
plt.plot(data.V7, color='b', label='V7',alpha=0.8)
plt.legend(loc='upper left')
plt.show()


# We cannot see the correlation between Amount and V7, due to the big difference in their respective values. We need to normalize V7 to get a better graph. Below, I multiplied V7 by 100 to normalize it. Also had the graph got bigger.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
plt.plot(data.Amount, color='r', label='Amount',alpha=0.8)
plt.plot(data.V7*100, color='b', label='V7',alpha=0.8)
plt.legend(loc='upper left')
plt.show()


# Looks like we still do not have a meaningful graph, due to the large row size. Therefore I will reduce the row size to 100.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
plt.plot(data.Amount[0:100], color='r', label='Amount',alpha=0.8)
plt.plot(data.V7[0:100]*100, color='b', label='V7',alpha=0.8)
plt.legend(loc='upper left')
plt.show()


# Now we can observe the correlation between V7 and Amount. Let's take a different row group of data to make sure we were not deceived by the first 100 rows.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
plt.plot(data.Amount[10000:10100], color='r', label='Amount',alpha=0.8)
plt.plot(data.V7[10000:10100]*100, color='b', label='V7',alpha=0.8)
plt.legend(loc='upper left')
plt.show()


# Actually this correlation is not significant enough. It looked like  from the heatmap that the correlation between Amount and V7 was around 0.4. Let's check it out again.

# In[ ]:


print(data[['Amount','V7']].corr())


# In[ ]:


print(data[['Amount','V20']].corr())


# Let's make a scatter diagram too between Amount and V7, the most correlated values.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
data.plot(kind='scatter',x='Amount',y='V7',color='b',ax=ax,alpha=0.5)
plt.title('Scatter Diagram, Amount and V7')
plt.show()


# There are only a few amounts more than 5000. We can eliminate them to get a better scatter diagram.

# In[ ]:


data2=data[data.Amount<5000]
f,ax=plt.subplots(figsize=(15,8))
data2.plot(kind='scatter',x='Amount',y='V7',color='b',ax=ax,alpha=0.5)
plt.title('Scatter Diagram, Amount and V7')
plt.show()


# Let's see how Amount is distributed by using a histogram diagram.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
data.Amount.plot(kind='hist',color='r',alpha=0.5,bins=50)
plt.title('Distribution of Amount')
plt.show()


# Again we see most of the values are less than 2500. Let's get rid of the outlier values and check the distribution again.

# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
data.Amount[data.Amount<=2500].plot(kind='hist',color='r',alpha=0.5,bins=50)
plt.title('Distribution of Amount')
plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
data.Amount[data.Amount<=500].plot(kind='hist',color='r',alpha=0.5,bins=50)
plt.title('Distribution of Amount')
plt.show()


# Now, let's make some add, update and delete transactions on our data by using dictionary transactions.

# In[ ]:


data.describe()


# In[ ]:


dict1 = dict(data)
print(dict1.keys())


# In[ ]:


#add
dict1[284807]=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,1]
print(dict1[284807])


# In[ ]:


#change
dict1[284807]=[1,1,1,1,1,1,1,1,1,1,1,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,1]
print(dict1[284807])


# In[ ]:


#delete
print(dict1[284807])
del dict1[284807]
print(dict1[284807])


# In[ ]:


#clear the whole dictionary
dict1.clear()
print(dict1)


# In[ ]:


#delete the dictionary variable itself
del dict1
print(dict1)


# In[ ]:


#filtering with two constraints
filter1=np.logical_and(data.Amount>10000,data.Time<100000)
data3=data[filter1]
print(data3)

