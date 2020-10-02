#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install plotly_express')


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


train_df= pd.read_csv('../input/train.csv')
train_df.head()


# In[5]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[6]:


(train_df.shape, test_df.shape)


# In[7]:


gender_sub_df = pd.read_csv('../input/gender_submission.csv')
gender_sub_df.head()


# In[8]:


gender_sub_df.shape


# # Exploratory analysis

# Let us look at the data type of the columns present in this data set.

# In[9]:


# datatype of columns
train_df.info()


# ### Visualize missing data

# In[10]:


# find missing data
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# `Age` and `Cabin` do have some missing data, we will impute for these later.

# In[11]:


import warnings
warnings.filterwarnings("ignore")


# ### Visualize distributions and correlations

# In[12]:


sns.pairplot(train_df[['Survived','Pclass','Age','Fare']], dropna=True)


#  * Does age show any correlation with Fare? Do older people pay higher?
#  * Does passenger class show any correlation with Fare? Does first class cost more?
#  * Do older passenger prefer better class?
#  * Does the passenger class / fare / age / gender determine whether or not you survived?

# #### Visualize distribution of class vs Age

# In[15]:


plot1 = px.histogram(train_df, x='Age', color='Pclass')
plot1


# In[30]:


px.histogram(train_df, x='Age', color='Survived', facet_col='Pclass')


# The third class is most skewed to be younger than 2nd and 1st classes. First class has a mostly uniform distribution, while 2nd is centered around `30`. Further, kids (younger than 20) are most likely to be in second and third classes.
# 
# Its clear that most of 3rd class did not survive, whereas most of first class did. Majority of children survived, they are in 2nd class.

# #### Visualize fare distribution across passenger class

# In[39]:


px.histogram(train_df, x='Fare', color='Pclass')


# In[31]:


px.histogram(train_df, x='Fare', facet_col='Pclass', color='Survived', range_x=[0,200], nbins=50)


# It is surprising that a lot of 1st class passengers, did pay around the same as 2nd and 3rd. They survived well. 3rd class primarily paid under $`50` but only a few survived. Thus, your class is a better indicator than age or fare.

# #### What influence does gender play?
# Let us first, look at the distribution of `Age` across the genders.

# In[37]:


px.histogram(train_df, x='Age', color='Sex')


# The distribution of male and female age looks similar, but there are more men in just about any age group.
# Next let us see how many men survived

# In[38]:


px.histogram(train_df, x='Age', color='Sex', facet_col='Survived')


# Well, men of early 20th centry have been quite chivalrous. Most women survived immaterial of their age. Next let us see how survival by gender is linked with survival by passenger class.

# In[34]:


px.histogram(train_df, x='Age', color='Sex', facet_col='Survived', facet_row='Pclass')


# This is a detailed plot to unpack.
#  * In first class, almost all women survived, but there are a lot of men who died.
#  * In class 2, same way most women survived and little to no men survived
#  * In class 3, a lot of men and women died. About half the women survived. Even kids died.

# In[41]:


sns.countplot(x='Survived', hue='Pclass', data=train_df)


# ## Impute for missing age
# 

# In[ ]:


sns.heatmap(train_df[['Survived','Pclass']])


# In[ ]:




