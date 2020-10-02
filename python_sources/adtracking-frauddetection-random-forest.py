#!/usr/bin/env python
# coding: utf-8

# In[46]:


# This is a simple data exploration notebook. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# It is an attempt to extend the thoughtful work of fellow kagglers - yulia (https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns) and anokas (https://www.kaggle.com/anokas/talkingdata-adtracking-eda). Some of the data structures I had to recreate to fit my code requirements. 

# 
# 

# In[5]:


print(os.listdir('../input'))


# I'm using the random training sample provided by the organizers (train_sample) which has 100000 click records. I'm choosing the random data since it'll give a relatively unbiased view of the data. Let's first have a peek into the dataset.

# In[47]:


train= pd.read_csv('../input/train_sample.csv')
train.head()


# In[48]:


train.info()


# Converting the datatypes to categorical to remove the notion of order from the variables with the exception of check_time and attributed_time which are converted to datetime. 

# In[49]:


cols=train.columns
cols_time=['click_time', 'attributed_time']
cols_categorical=[col for col in cols if col not in cols_time]

for col in cols_categorical:
    train[col]=train[col].astype('category')
for col in cols_time:
    train[col]=pd.to_datetime(train[col])


# In[50]:


train.describe()


# One way to look at the data - these are the entitites we are dealing with - 
# 
# * **user** : defined by ip, device, os
# * **mode** :  defined by app, channel
# * **user journey** : Start - click_time (user came in contact with the mode) and End - attributed_time (user was acquired)

# In[51]:


train['conversion_time']=pd.to_timedelta(train['attributed_time']-train['click_time']).astype('timedelta64[s]')
print(train['conversion_time'].quantile(0.9)/3600)
train.describe()


# Amongst the users which were acquired:
# *     Min time between user coming in contact with a mode and the user getting acquired is 4 seconds.
# *     50% of the users got converted within approx 5 mins.
# *     75% of the users took 1 hour to decide and download.
# *     90% of the users took 4 hours to decide and download.
# *     Max time taken by a user after coming in contact with an ad and going for download is approx 20 hours.
# Hence within a day of contact with the ad, the users who could have been acquired were acquired. This is the engagement duration. Let's look at the full distribution.

# In[11]:


ctimes=train['conversion_time'].dropna()


# In[12]:


sns.distplot(ctimes)


# In[13]:


sns.distplot(np.log10(ctimes))


# In[14]:


# min_time=np.nanmin(train['click_time'])
# max_time=np.nanmax(train['attributed_time'])
# print(min_time)
# print(max_time)


# In[15]:


# max_engagement_window_size=str(int(np.ceil(np.nanmax(ctimes)/3600)))+'H'
# max_engagement_window_size


# Now let's look at how the different variables are correlated in the 2 different categories - Successful Ad campgains versus Unsuccessful ones. 

# In[16]:


sns.pairplot(train[train['is_attributed']==1])


# Here are a few observations for the successful Ad campaigns:
# * The conversion times are failry distributed over the different ip addresses and the channels.
# * There are few devices which have succesful conversions, however there are many devices which don't have any conversions at all. Maybe the ads on those devices are not that engaging(say particular tablets or desktop versions). 

# In[17]:


col_list=[col for col in train.columns if col!='conversion_time']
# print(col_list)

sns.pairplot(train.loc[train['is_attributed']!=1,col_list])


# 

# Let's build a simple classifier

# In[24]:


from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# In[53]:


train['is_attributed'].value_counts()/train.shape[0]


# It's a very imabalanced dataset with 99.8% of the clicks not yielding to any downloads and only 0.2% of the data has downloads!!! We'll need evaluation metrics like roc_auc_score and precision_recall_curve

# In[54]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve


# First let's create some features for the click time. Attributed time is NaT for all those data points which have is_attributed=0, which is 99.8% of the data!

# In[55]:


train['chour']=train['click_time'].dt.hour
train['cminute']=train['click_time'].dt.minute
train['cday']=train['click_time'].dt.day


# In[56]:


del train['click_time']
del train['attributed_time']


# In[ ]:


Check how much of the data still has missing values


# In[58]:


train.isnull().any()


# Conversion time still has NaN values corresponding to all the cases where there was no attributed_time (which is 99.8%) of the cases). Hence it's better to drop this column as well.

# In[59]:


del train['conversion_time']


# In[60]:


X=train
Y=train['is_attributed']
del X['is_attributed']
X.head()


# In[61]:


Xtrain, Xtest, Ytrain, Ytest=train_test_split(X,Y,test_size=0.2, random_state=32)


# **Random Forest**

# In[62]:


param_grid={'n_estimators':np.arange(10,100,20), 
           'max_depth':[None, 5, 10],
           'max_features':['auto','sqrt']}
rfgrid=GridSearchCV(RandomForestClassifier(),param_grid,cv=StratifiedKFold(5))


# In[63]:


rfgrid.fit(Xtrain, Ytrain)


# In[68]:


rfgrid.score(Xtest,Ytest)


# Though the score is 99.76%, the class is highly imbalanced, hence other score measures are needed..
