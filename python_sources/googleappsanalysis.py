#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt  #importing the necessary libraries
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the csv file

# In[ ]:


google_data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')  #read the csv file


# In[ ]:


google_data.head()  #inspecting first 5 rows


# In[ ]:


google_data.tail() #last 5 rows


# In[ ]:


type(google_data) #type of data


# In[ ]:


google_data.shape  #orientation of data


# In[ ]:


google_data.describe() #Gives only one column, because only rating is numerical, rest all are categorical


# In[ ]:


google_data.boxplot()


# In[ ]:


google_data.hist()


# In[ ]:


google_data.info()


# In[ ]:


google_data.isnull()


# In[ ]:


google_data.isnull().sum()


# In[ ]:


google_data[google_data.Rating > 5]


# In[ ]:


google_data.drop([10472], inplace = True)


# In[ ]:


google_data[10470: 10475]


# In[ ]:


google_data.boxplot() #Most of the data is focussed from 4 to 4.5


# In[ ]:


google_data.hist()


# If for some dataset, 90% of the columns are empty, we just take 10% of data out of it

# In[ ]:


threshold = len(google_data) * 0.1
threshold


# We are removing the columns that doesn't have even 1084 (or) threshold level or 10% of the overall data filled

# In[ ]:


print(google_data.isnull().sum())


# In[ ]:


google_data.shape #In this dataset, it remains the same 


# **Data Imputaion and Manipulation**
# 
# For categorical values, we are using mode and for numerical values we are using median and not mean, because based on the histogram, we can understand that the data is right skewed. 
# 
# 1. Right and left skewed data -> we will use median and not mode

# In[ ]:


def impute_median(series): #series means -> a column
    return series.fillna(series.median())


# In[ ]:


google_data.Rating = google_data['Rating'].transform(impute_median) #for each value in rating column, we are calling the impute_median function (passing each value to the function)


# In[ ]:


google_data.isnull().sum()


# Check the mode for the filling the missing values in the categorical columns

# In[ ]:


print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())


# Index[0] because, we will be using the first mode value in case there are any bi-moded values

# In[ ]:


google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace = True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace = True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace = True)
                                  


# In[ ]:


google_data.isnull().sum()


# In[ ]:


google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_data['Price'] = google_data['Price'].apply(lambda x: float(x)) 
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors = 'coerce') #coerce ignores the errors


# In[ ]:


google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))


# In[ ]:


google_data.head(10)


# In[ ]:


google_data.describe()


# Data Visualization

# In[ ]:


grp = google_data.groupby('Category') #Grouping apps based on cateory
x = grp['Rating'].agg(np.mean)  #agg=> aggregate, jus combines everything
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)


# In[ ]:


print(y)


# In[ ]:


print(z)


# In[ ]:


plt.figure(figsize = (15, 7))
plt.plot(x, 'ro')
plt.xticks(rotation = 90) #making the x label texts vertical
plt.title('Category wise rating')
plt.xlabel('Categories')
plt.ylabel('Rating')
plt.show()


# In[ ]:


plt.figure(figsize = (15,7))
plt.plot(y, 'b--')
plt.xticks(rotation = 90)
plt.title('Category wise pricing')
plt.xlabel('Categories')
plt.ylabel('Prices')
plt.show()


# In[ ]:


plt.figure(figsize = (15,6))
plt.plot(z, 'g^')
plt.xticks(rotation = 90)
plt.xlabel('Categories')
plt.ylabel('Reviews')
plt.title('Category wise Reviews')
plt.show()


# In[ ]:





# In[ ]:




