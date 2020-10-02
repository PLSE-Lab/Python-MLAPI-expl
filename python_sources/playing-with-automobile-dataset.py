#!/usr/bin/env python
# coding: utf-8

# **IMPLEMENTING DATA WRANGLING TECHNIQUES LEARNED FROM DATACAMP PYTHON PROGRAMMER TRACK**

# **Step1: **Importing Dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("../input/Automobile_data.csv")
print("Done")


# In[3]:


df.head()


# **Step2**: Data Wrangling

# In[4]:


import numpy as np
df.replace("?",np.nan,inplace=True)


# In[5]:


df.head()


# In[6]:


missing_data = df.isnull()
missing_data.head(5)


# In[7]:


df.isnull().sum()


# In[8]:


#df['normalized-losses'].astype("float").mean()
df['normalized-losses'].replace(np.nan,122.0,inplace=True)


# In[9]:


df.head()


# In[10]:


bore_mean= df['bore'].astype("float").mean()


# In[11]:


df['bore'].replace(np.nan,bore_mean,inplace=True)


# In[12]:


df.head()


# In[13]:


stroke_mean = df['stroke'].astype('float').mean()


# In[14]:


df['stroke'].replace(np.nan,stroke_mean,inplace=True)


# In[15]:


df.head()


# In[16]:


avg_4=df['horsepower'].astype('float').mean(axis=0)
df['horsepower'].replace(np.nan, avg_4, inplace= True)
avg_5=df['peak-rpm'].astype('float').mean(axis=0)
df['peak-rpm'].replace(np.nan, avg_5, inplace= True)


# In[17]:


df['num-of-doors'].value_counts()


# In[18]:


df['num-of-doors'].value_counts().idxmax()


# In[19]:


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace = True)


# In[20]:


df.dropna(subset=['price'],axis=0,inplace=True)


# In[21]:


df.head()


# In[22]:


df.isnull().sum()


# **DATASET CLEANING SUCCESSFULL!!!**

# *OOPS! forgot to check if data in correct format and normalization also applied afterwards for training*

# In[23]:


df.dtypes


# In[24]:



df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
print("Done")


# In[25]:


df.dtypes


# performing binning on horsepower due to continuous variable not needed

# In[26]:


df["horsepower"]=df["horsepower"].astype(float, copy=True)


# In[31]:


#Divide data in 3 equally sized bins
binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4
binwidth


# In[34]:


#1st bin 48-101 2nd bin 101.5-155 and 3rd bin 155-208.5 that's why in previous cell divided by 4
bins = np.arange(min(df["horsepower"]), max(df["horsepower"]), binwidth)
bins


# In[29]:


group_names = ['Low', 'Medium', 'High']


# In[30]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3,color=['green'],rwidth=0.75)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[40]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# replace (origianl value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()


# In[76]:


#dummy_variable_1 = pd.get_dummies(df["fuel-type"])
#dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
#df = pd.concat([df, dummy_variable_1], axis=1)
#df.drop("fuel-type", axis = 1, inplace=True)


# In[67]:


df.head()


# In[70]:


#MY MISTAKE SORRY TRIED TO RUN THE DROP STATMENT WITHOUT INPLACE AND THEN ADDED INPLACE RESULTED IN DUPLICATE COLUMNS
df = df.loc[:,~df.columns.duplicated()]


# In[71]:


df


# In[72]:


dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis = 1, inplace=True)


# In[74]:


df.head()


# In[75]:


df.to_csv('clean_automobile_data.csv')


# **Step3**: Exploratory Data Analysis

# In[77]:


import numpy as np


# In[78]:


dfclean = pd.read_csv('clean_automobile_data.csv')


# In[80]:


dfclean.head(20)


# In[82]:


dfclean.dtypes


# In[89]:


dfclean.drop(['Unnamed: 0'], axis=1,inplace=True)


# In[90]:


dfclean


# In[91]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[97]:


dfclean.corr()


# In[101]:


dfclean[['bore','stroke' ,'compression-ratio','horsepower','price']].corr()


# In[102]:


dfclean[["engine-size", "price"]].corr()


# In[103]:


dfclean['drive-wheels'].value_counts()


# In[105]:


sns.boxplot(x='drive-wheels',y='price',data=dfclean)


# In[109]:


plt.scatter(x=dfclean['engine-size'],y=dfclean['price'])


# In[ ]:




