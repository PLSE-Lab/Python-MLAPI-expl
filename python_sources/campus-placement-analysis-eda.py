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


# **PLEASE UPVOTE WHEN YOU FIND IT USEFUL**
# NEW LEARNER ,YOUR VOTE WILL ENCORAGE ME 

# In[ ]:


import pandas as pd


# Using pandas to import the files

# In[ ]:


df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# reading the top five values

# In[ ]:


df.head()


# we would require some other visuallization and mathematical libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# first we should check for the size and null vallues in data********

# In[ ]:


df.info()


# lets check the factors
# msking a countplot of gender first 
# 

# looks fine!!!
# 

# In[ ]:


sns.countplot(df['gender'],hue=df['status'])


#  looks like a typical engg college
#  we can see that fraction of student placed is almost equal for men and women.
#  more than half,close to 2/3

# In[ ]:


sns.scatterplot(y=df['ssc_p'],x=df['sl_no'],hue=df['status'])


# we can conclude from here that this is a very important criteria.
# students above ssc p 70% are mostly placed.

# In[ ]:


sns.countplot(df['ssc_b'],hue=df['status'])


# as of now we can argue that this factor doesnt matter much.

# In[ ]:


sns.scatterplot(y=df['hsc_p'],x=df['sl_no'],hue=df['status'])


# intuitively from here we see that at 60% chances are more than half but more than 70 % is better.

# In[ ]:


sns.countplot(df['hsc_b'],hue=df['status'])


# not much effect of board is seen here.

# In[ ]:


sns.countplot(df['hsc_s'],hue=df['status'])


# this is interesting,
# in Science and commerce the placement is little less than half.
# and in arts it is very high but considering we have only  5 -6 students it is not intutive.

# In[ ]:


sns.scatterplot(y=df['degree_p'],x=df['sl_no'],hue=df['status'])


# here we can say that this percent is not as evenly distributed as previous 2 percentages.
# so getting higher percent in degree is not cumpulsory but good.

# In[ ]:


sns.countplot(df['degree_t'],hue=df['status'])


# considering diff no of students in each stream ,
# we see than no of stundent placed are higher in comm and managment ,but no of stundent i also more.
# so does'nt effect much
# 

# In[ ]:


sns.countplot(df['workex'],hue=df['status'])


# this is very helpful!!!!!!!!
# students who have a prior workex are way more likely to be selected.
# means if you have a workexperience then you have a chance of 65/75*100 =86% to get placed

# In[ ]:


sns.scatterplot(y=df['etest_p'],x=df['sl_no'],hue=df['status'])


# considering this plot,we can say that this doesnt matter at all!!!!!!!
# it can be just a cut-off score.

# In[ ]:


sns.countplot(df['specialisation'],hue=df['status'])


# Clearly we see that more students are placed in marketing and finance .
# 

# In[ ]:


sns.scatterplot(y=df['mba_p'],x=df['sl_no'],hue=df['status'])


# no uniform distribution is seen so doesnt affect the stuatus much

# **Now as we have considered all criterias to be placed ,
# we can say that WORK EXPERIENCE affects the placement most and after that SPECIALISAION.
# the ssc % and hsc % are also important like greater than 70 % is good.******

# **IN CASE OF PERCENTAGE ,WE CAN SAY THAT THERE IS A ROUGH PERCENT,WHICH IS MORE LIKE A CUTOFF AND ONLY IS CASE OF HSC-P AND SSC-P**

# to observe the demand ,we have to consider the salary criteria also.
# 

# **clearly we can see that more students are places in mareting and finance**.****
# 

# Now lets see some means and highest lowest values

# In[ ]:


df[df['salary']==df['salary'].max()]


# here we have to observe that this stundent had a desent percentages but had a work experrinece.

# In[ ]:


df[df['gender']=='M']['salary'].mean()


# In[ ]:


df[df['gender']=='F']['salary'].mean()


# so salary for both is almost equal

# In[ ]:




