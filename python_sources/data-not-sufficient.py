#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries

# for data munging
import numpy as np
import pandas as pd

#for viz
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# for ignoring the warnings while printing
import warnings
warnings.filterwarnings('ignore')

#for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


# Importing file


# File Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        


# In[ ]:


df = pd.read_csv('/kaggle/input/corruption-in-india/CorruptionInIndia.csv')
df.head()


# In[ ]:


#Shape of dataframe
print(" Shape of dataframe: ", df.shape)


# In[ ]:


# Drop duplicates
df.drop_duplicates()
print(" Shape of dataframe after dropping duplicates: ", df.shape)


# In[ ]:


#Variable inspection

print("Names of columns ", list(df.columns))


# In[ ]:


print(df.info())


# #### There are a lot of null values

# In[ ]:


#Null values

null= df.isnull().sum().sort_values(ascending=False)
total =df.shape[0]
percent_missing= (df.isnull().sum()/total).sort_values(ascending=False)

missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data= missing_data.rename(columns= { "index": " column name"})
 
print ("Null Values in each column:\n", missing_data)


# In[ ]:


#See see how null values look in dataframe
#Missing data as white lines 
import missingno as msno
msno.matrix(df,color=(0,0.3,0.9))


# In[ ]:


boxplot= df.boxplot(column=['Stock', "CommodityStock", "TotalStock"]);


# In[ ]:


df.describe()


# In[ ]:


#Normal distribution

sns.distplot(df['TotalStock'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['TotalStock'], plot=plt)

#skewness and kurtosis
print("Skewness: %f" % df['TotalStock'].skew())
print("Kurtosis: %f" % df['TotalStock'].kurt())


# # Data is not enough for any analysis!!!

# In[ ]:




