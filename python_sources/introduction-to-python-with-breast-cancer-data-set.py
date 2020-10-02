#!/usr/bin/env python
# coding: utf-8

# **In this tutorial, we aim to perform basic data manipulation with the Breast Cancer Wisconsin (Diagnostic) Data Set provided by UCI.**
# 
# Firstly, we need to import our data and necessary libraries for our study. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualizing processes


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/data.csv") # you can add data with using pandas library easily

# Any results you write to the current directory are saved as output.


# When you import our data, we want to check information about  data. It is possible to do this with a single command as follows:

# In[ ]:


data.info()


# There are a few other commands that can be used to get detailed information about the data. 
# For example; you can reach first 5 rows in data with this command: 

# In[ ]:


data.head(5)


# Furthermore, you can see all columns with using this command:

# In[ ]:


data.columns


# Also, we may want to find out about the dependency of our predictions by seeing the correlation between the columns of our data.
# 
# The correlation between 2 columns is determined as;
# - If there is a correlation value which is close to 1, it is directly proportional,
# - If there is a correlation value which is close to -1, it is inversely proportional,
# - If there is a correlation value which is equal to 0, there is no relationship between two features.
# 
# For example; we can say that, the relationship between raduis_mean and perimeter_mean is directly proportional.

# In[ ]:


data.corr()


# After we have reached our data information, we may also want to access information visually on the whole data. 
# 
# We can easily visualize (plot) our data with **Matplotlib** library in Python.
# 
# The **Matplotlib**  library contains various graphic styles such as Line, Histogram, Scatter etc. For this reason, we can choose graphic styles that will be suitable for efficiency.
# 
# Let's try some plot drawings. 

# In[ ]:


data.texture_se.plot(kind='line', color='green', label='texture_se', linewidth=2, alpha=0.5, grid=False, linestyle='-')


# In[ ]:


data.texture_se.plot(kind='line', color='green', label='texture_se', linewidth=2, alpha=0.5, grid=False, linestyle='-')
data.smoothness_se.plot(kind='line', color='red', label='smoothness_se', linewidth=2, alpha=0.5, grid=False, linestyle=':')
plt.xlabel('x')              # label = name of label
plt.ylabel('y')
plt.title('Line Plot with Texture and Smoothness')            # title = title of plot
plt.show()


# In[ ]:


data.perimeter_mean.plot(kind='hist', bins=10, figsize=(10,10))


# In[ ]:


data.plot(kind='Scatter', x='area_mean', y='symmetry_worst', alpha=0.7, color='blue', grid=True)


# In[ ]:


plt.scatter(data.area_mean, data.symmetry_worst)


# As you can see above, we can make the same plot drawing with different commands and variables.
# 
# There is some variables for plot customizations.
# - kind determines the plot type.
# - alpha determines the opacity of the points.
# - color determines the color of the points.
# - label determines the flag of the points.
# - grid determines creates squares on plot.
# - figsize determines plot size.
# 
# 
# 

# After learning how to visualize the data, now we learn how to reach the desired data specifically.
# 
# **Dictionary** data type contains key and value pairs for every entry. We use the Dictionary because it works faster than the list structure.
# 
# Let's look at Dictionary syntax and processes.

# In[ ]:


dictionary = {'type': 'malignant', 'area': 'breast'}
print(dictionary)
print("\n")
print(dictionary.keys()) #prints keys
print(dictionary.values()) #prints values

dictionary['type'] = 'benign' #update dictionary
print("\n")
print(dictionary.keys())
print(dictionary.values())

del dictionary['area'] #deletes area's key & value
print("\n")
print(dictionary.keys())
print(dictionary.values())


# There are some operators that we use to perform mathematical comparison and filtering operations on our data.
# 
# Now, we will mention about these operators.

# In[ ]:


malignant = data['diagnosis'] =='M'
data[malignant]



# In[ ]:


data[(data['diagnosis']=='B') & (data['radius_mean']>15)]


# Generally, our data will not be small enough to reach each of them manually.
# 
# For this reason;  We use loop structures to reach the data faster.
# 
# You can examine several loop examples in below.

# In[ ]:


i = 0
while i != 5 :
    print(data['id'][i]) #print id's for first 5 rows
    i +=1 


# In[ ]:


dictionary = {'type': 'malignant', 'area': 'breast', 'tumor_radius': '14'}
for key,value in dictionary.items():
    print(key," : ",value)

