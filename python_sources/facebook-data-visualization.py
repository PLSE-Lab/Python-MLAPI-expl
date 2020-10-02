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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#load the data set into df
df= pd.read_csv('../input/pseudo_facebook.csv')
#describe data 
df.describe()


# In[ ]:


#histogram
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
_ = plt.hist(df['age'])
_ = plt.xlabel("no of people")
_ = plt.ylabel("age")
_ = plt.title("people on facebook")
plt.show()


# In[ ]:


#histogram
sns.set()
_ = plt.hist(df['dob_day'])
_ = plt.xlabel("no of people")
_ = plt.ylabel("age")
_ = plt.title("Day wise register")
plt.show()


# In[ ]:


#histogram
sns.set()
_ = plt.hist(df['dob_month'])
_ = plt.xlabel("no of people")
_ = plt.ylabel("age")
_ = plt.title("Month wise register")
plt.show()


# In[ ]:


#histogram
sns.set()
_ = plt.hist(df['dob_month'])
_ = plt.xlabel("no of people")
_ = plt.ylabel("age")
_ = plt.title("Month wise register")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
_ = plt.hist(df['age'])
_ = plt.xlabel('percentage of age')
_ = plt.ylabel('no of people')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.boxplot(x ='friend_count', y ='likes', data = df)


# In[ ]:


bin_edges = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
_ = plt.hist(df['age'],bins = bin_edges)
_ = plt.xlabel('percentage of age')
_ = plt.ylabel('no of people')
plt.show()


# In[ ]:


plt.hist(df['age'])


# In[ ]:


sns.pairplot(df,x_vars=["age"],y_vars="likes",size=4)


# In[ ]:


#pair plot
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df,x_vars=["age"],y_vars="likes",size=4)
_ = plt.xlabel("age")
_ = plt.ylabel("likes")
_ = plt.title("Age based likes")


# In[ ]:


sns.pairplot(data = df, hue="gender")


# In[ ]:



sns.barplot(df['gender'],df['likes'])


# In[ ]:


sns.pairplot(dataframe,x_vars=["likes","friend_count"],y_vars="age",size=10)


# In[ ]:


import seaborn as sns; sns.set(color_codes=True)
sns.regplot(x="likes", y="tenure", data=df)


# In[ ]:


import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import numpy as np
ax = sns.regplot(x="likes_received", y="mobile_likes_received", data=df,x_estimator=np.mean)


# In[ ]:


sns.lmplot( x="mobile_likes_received", y="www_likes_received", data=df, fit_reg=False, hue='tenure', legend=False)


# In[ ]:



sns.lmplot( x="age", y="friendships_initiated", data=df, fit_reg=False, hue='dob_year', legend=False)

