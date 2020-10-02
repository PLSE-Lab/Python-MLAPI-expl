#!/usr/bin/env python
# coding: utf-8

# **Problem :**
# Predict the chances of admission of a student to a University fro Graduate program based on different parameters such as:
# 
# GRE Scores (290 to 340)
# TOEFL Scores (92 to 120)
# University Rating (1 to 5)
# Statement of Purpose (1 to 5)
# Letter of Recommendation Strength (1 to 5)
# Undergraduate CGPA (6.8 to 9.92)
# Research Experience (0 or 1)
# Chance of Admit (0.34 to 0.97)
# 
# **Solve :**
# using Fast.ai Tabular model

# In[36]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #For visualization
from matplotlib import rcParams #add styling to the plots
from matplotlib.cm import rainbow #for colors
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[37]:


# import everything we need for the tabular application
from fastai.tabular import *


# Graduate Admissions dataset which has information on individual GRE, TOEFL scores, Univeristy ratings, etc. We'll use it to train a model to predict whether a student will get admitted (Students whose chances of admit are greater than 80%) or not. 

# In[38]:


# read the data file
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# All the information that will form our input is in the first 8 columns, and the dependent variable is the last column. 
# 
# Continuous variables will be normalized and then directly fed to the model.
# Any missing values, NaNs:  Has to be removed  All of this preprocessing is done by TabularTransform objects and TabularDataset.
# Define a bunch of Transforms that will be applied to our variables.  Replace missing values for continuous variables by the median column value and normalize those.

# **Explore the data**

# In[39]:


df.shape # number of rows and columns


# In[40]:


df.info()


# In[41]:


df.describe()


# In[42]:


df.sample(5)


# In[43]:


df.columns


# In[50]:


df=df.rename(columns = {'Chance of Admit ':'Admit'})


# In[51]:


#Visualize
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True)


# GRE Score, TOEFL Score and CGPA has the highest correlation
# 

# GRE Score: Histogram shows the frequency for GRE scores.

# In[52]:


# Understanding the data
# Visualizations to better understand data and do any processing if needed.
df["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
plt.show()


# TOEFL Score: Histogram shows the frequency for TOEFL scores.

# In[53]:


df["TOEFL Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("TOEFL Scores")
plt.xlabel("TOEFL Score")
plt.ylabel("Frequency")
plt.show()


# In[54]:


s = df[df["Admit"] >= 0.80]["GRE Score"].value_counts().head(5)
plt.title("GRE Scores of Candidates with an 80% acceptance chance")
s.plot(kind='bar',figsize=(20, 10))
plt.xlabel("GRE Scores")
plt.ylabel("Candidates")
plt.show()


# In[55]:


# Process the data 
df = df.drop(['Serial No.'], axis=1)


# In[56]:


df.head()


# If a candidate's Chances of Admission is greater than 80%, the candidate will receive the 1 label.
# If a candidate's Chances of Admission is less than or equal  to 80%, the candidate will receive the 0 label.

# In[57]:


df.loc[df.Admit <= 0.8, 'Admitted'] = '0' 
df.loc[df.Admit > 0.8, 'Admitted'] = '1' 
#df


# In[58]:


df['Admit'] = df['Admitted']
df.Admit=df.Admit.astype(int)
df = df.drop(['Admitted'], axis=1)

df.head()


# Manually split our variables into categorical and continuous variables (ignore the dependent variable at this stage). fastai will assume all variables that aren't dependent or categorical are continuous, unless we explicitly pass a list to the cont_names parameter when constructing our DataBunch
# 
# 
# 
# 
# 
# 

# In[59]:


dep_var = 'Admit'
cat_names = []
cont_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA','Research']
procs = [FillMissing, Categorify, Normalize]


# In[60]:


#save the model
path="../kaggle/working"


# To split our data into training and validation sets, we use valid indexes

# In[75]:


#np.random.rand()


# In[70]:


#valid_idx = range(len(df)-100, len(df))
#valid_idx 

random.seed(33148690)
valid_idx = random.sample(list(df.index.values), int(len(df)*0.2) )
#valid_idx 


# Pass this information to TabularDataBunch.from_df to create the DataBunch that will be used for training.

# In[71]:



data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
#print(data.train_ds.cont_names)


# In[72]:


data.show_batch(rows=5)


# **Defining a model**
# Once the data is ready in a DataBunch, create a model to then define a Learner and start training. The fastai library has a flexible and powerful TabularModel in models.tabular. To use that function, we just need to specify the embedding sizes for each of our categorical variables. https://docs.fast.ai/tabular.html

# In[73]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[74]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:



#df.loc[df['Admit'] == 1]


# Use the Learner.predict method to get predictions. In this case, we need to pass the row of a dataframe that has the same continuous variables as our training or validation dataframe.

# In[76]:


learn.predict(df.iloc[150])


# In[77]:


learn.predict(df.iloc[50])

