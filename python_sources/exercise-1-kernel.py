#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


credit = pd.read_csv('../input/weka-german-credit/credit-g.csv')


# In[ ]:


## check datatypes
credit.dtypes


# In[ ]:


## change class to category(nominal) type
credit['class'] = credit['class'].astype('category')

## smaller int can be used to save space
#credit['num_dependents'] = credit['num_dependents'].astype('int8')


# In[ ]:


## change all object(text) attributes to categorical(nominal) type
for k in credit:
    if credit[k].dtype == 'object':
        credit[k] = credit[k].astype('category')


# In[ ]:


## see if successful
credit.dtypes


# In[ ]:


## number of attributes
len(credit.dtypes)
# or since dtypes is Series
credit.dtypes.count()


# In[ ]:


#number of each type of attribute
credit.get_dtype_counts()


# In[ ]:


#possible values for 'housing' attribute
credit['housing'].unique()
#same for 'credit_history' and 'purpose'

#Categories (<unique>, object): ...


# In[ ]:


## I don't yet know an easier way to do this

##records with employment == unemployed
frame = credit[credit['employment'] == 'unemployed']
#counts of good and bad in frame
counts = unemp.groupby(['class']).count()['employment']
##percentage bad
float(counts['bad']) / sum(counts)

## so make a list of bad credit percentages 
arr = [[],[]]
for c in credit['employment'].unique():
    frame = credit[credit['employment'] == c]
    counts = frame.groupby(['class']).count()['employment']
    arr[0].append(c)
    arr[1].append(float(counts['bad']) / sum(counts))

##argmax gets index of largest value -> same index of category
arr[0][np.argmax(arr[1])]


# In[ ]:


## get counts for personal_status
credit.groupby(['personal_status']).count()['class']


# In[ ]:


##create pandas datapframe from csv file
cars = pd.read_csv('../input/ex1-cars/cars.csv')


# In[ ]:


##return first few records in dataframe
cars.head()


# In[ ]:


##return datatypes of all attributes
cars.dtypes


# In[ ]:


## return Series of values in column
data['Manufacturer']


# In[ ]:


## change datatype to categry and repeat logic in cell above
data['Manufacturer'] = data['Manufacturer'].astype('category')
data['Manufacturer']


# In[ ]:


## change other datatypes which should be categorical
data['Model'] = data['Model'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')


# In[ ]:


## count each pair
data.groupby(['Manufacturer','Model']).size()


# In[ ]:


## plot mean price of models produced by each manufacturer
data.groupby(['Manufacturer']).mean()['price'].plot.bar()


# <h1>Plot of average price for each manufacturer</h1>

# In[ ]:


data.groupby(['Manufacturer','Model']).size().loc['Honda'].plot.bar()


# In[ ]:




