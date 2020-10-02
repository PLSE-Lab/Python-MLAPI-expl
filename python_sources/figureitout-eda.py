#!/usr/bin/env python
# coding: utf-8

# # FigureItOut Complete EDA

# Importing Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier


# Setting some options

# In[ ]:


pd.set_option("max_columns",110)
plt.rcParams['figure.figsize'] = 50,50


# Reading data files

# In[ ]:


train = pd.read_csv("../input/train.csv",index_col="unique_id")
test = pd.read_csv("../input/test.csv",index_col="unique_id")


# Seperation of data

# In[ ]:


y = train.targets.values
x = train.drop(["targets"],axis=1).values


# ## BaseLine Classifier Training

# Splitting train set to train and validation set

# In[ ]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=0)
for train_index, test_index in sss.split(x, y):
    x_train,x_test,y_train,y_test = x[train_index],x[test_index],y[train_index],y[test_index]


# Training and testing the model

# In[ ]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict_proba(x_test)
log_loss(y_test,y_pred)


# # DataCleaning, EDA and Feature Engg

# In[ ]:


train_df = train.copy()
test_df = test.copy()


# In[ ]:


train_df.head()


# In[ ]:


train_df.hist()
plt.show()


# We can see some columns have very high values but they are more or less same we can subtract those columns with there smallest value to get the best result

# In[ ]:


train_df[["x_79"]] = train_df[["x_79"]]+2017
train_df["x_6"] = train_df["x_6"]+101.25
train_df["x_6"] = train_df["x_6"].astype("int64")
train_df[["x_51"]] = train_df[["x_51"]]-123456
train_df[["x_61"]] = train_df[["x_61"]]-2017
test_df[["x_79"]] = test_df[["x_79"]]+2017
test_df["x_6"] = test_df["x_6"]+101.25
test_df["x_6"] = test_df["x_6"].astype("int64")
test_df[["x_51"]] = test_df[["x_51"]]-123456
test_df[["x_61"]] = test_df[["x_61"]]-2017


# In[ ]:


train_df.head()


# If we see further we can also convert other colums to int by taking there log

# In[ ]:


train_df['x_31'] = np.log(train_df['x_31'])
test_df['x_31'] = np.log(test_df['x_31'])
train_df['x_31'] = train_df['x_31'].astype("int64")
test_df["x_31"] = test_df["x_31"].astype("int64")
train_df['x_90'] = np.log(train_df['x_90'])
test_df['x_90'] = np.log(test_df['x_90'])
train_df['x_90'] = train_df['x_90'].astype("int64")
test_df["x_90"] = test_df["x_90"].astype("int64")


# In[ ]:


train_df.head()


# There are some remaning columns with decimal values we can convert them to int by taking floor of them this also keep the same density accros the axis

# In[ ]:


plt.rcParams.update(plt.rcParamsDefault)


# In[ ]:


sns.kdeplot(train_df.x_94)


# In[ ]:


v = train_df.x_94
v = np.floor(v)
sns.kdeplot(v)


# In[ ]:


train_df.x_94 = train_df.x_94.apply(np.floor).astype("int64")
train_df.x_98 = train_df.x_98.apply(np.floor).astype("int64")
train_df.x_100 = train_df.x_100.apply(np.floor).astype("int64")
train_df.x_11 = train_df.x_11.apply(np.floor).astype("int64")
train_df.x_13 = train_df.x_13.apply(np.floor).astype("int64")
train_df.x_17 = train_df.x_17.apply(np.floor).astype("int64")
train_df.x_21 = train_df.x_21.apply(np.floor).astype("int64")

test_df.x_94 = test_df.x_94.apply(np.floor).astype("int64")
test_df.x_98 = test_df.x_98.apply(np.floor).astype("int64")
test_df.x_100 = test_df.x_100.apply(np.floor).astype("int64")
test_df.x_11 = test_df.x_11.apply(np.floor).astype("int64")
test_df.x_13 = test_df.x_13.apply(np.floor).astype("int64")
test_df.x_17 = test_df.x_17.apply(np.floor).astype("int64")
test_df.x_21 = test_df.x_21.apply(np.floor).astype("int64")


# In[ ]:


train_df.head()


# Most of the plots of the columns which contains the negative number are gaussin plots we can use the property of these convert them to positive

# In[ ]:


sns.kdeplot(train_df['x_13'])


# In[ ]:


def negpos(a):
    if(a<0):
        a=-a
    else:
        a=a
    return a


# In[ ]:


train_df.x_94 = train_df.x_94.apply(negpos)
train_df.x_98 = train_df.x_98.apply(negpos)
train_df.x_100 = train_df.x_100.apply(negpos)
train_df.x_11 = train_df.x_11.apply(negpos)
train_df.x_13 = train_df.x_13.apply(negpos)
train_df.x_17 = train_df.x_17.apply(negpos)
train_df.x_21 = train_df.x_21.apply(negpos)

test_df.x_94 = test_df.x_94.apply(negpos)
test_df.x_98 = test_df.x_98.apply(negpos)
test_df.x_100 = test_df.x_100.apply(negpos)
test_df.x_11 = test_df.x_11.apply(negpos)
test_df.x_13 = test_df.x_13.apply(negpos)
test_df.x_17 = test_df.x_17.apply(negpos)
test_df.x_21 = test_df.x_21.apply(negpos)


# In[ ]:


train_df.head()


# Finally for column 96 we can take log10 to convert that to its original value

# In[ ]:


def log_problem(a):
    if(a<=1):
        return 0
    else:
        return np.log10(a) 
    
train_df.x_96=train_df.x_96.apply(log_problem)
train_df.x_96=np.ceil(train_df.x_96).astype("int64")

test_df.x_96=test_df.x_96.apply(log_problem)
test_df.x_96=np.ceil(test_df.x_96).astype("int64")


# In[ ]:


train_df.head()


# ## Now we can start modelling as our data is properly cleanded and scaled

# We can also remove outliers for some more accuracy

# Splitting data for validation

# In[ ]:


y_new = train_df.targets.values
x_new = train_df.drop(["targets"],axis=1).values


# In[ ]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=0)
for train_index, test_index in sss.split(x, y):
    x_train,x_test,y_train,y_test = x_new[train_index],x_new[test_index],y_new[train_index],y_new[test_index]


# ### 1. Logistic Regression

# In[ ]:


lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train,y_train)
y_pred = lr.predict_proba(x_test)
log_loss(y_test,y_pred)


# Utility function for file submission

# In[ ]:


t = pd.read_csv("../input/test.csv")
def filesub(v,y_pred,test):
    a = pd.DataFrame(y_pred,columns=['proba_1', 'proba_2', 'proba_3', 'proba_4', 'proba_5', 'proba_6', 'proba_7', 'proba_8', 'proba_9'])
    a["unique_id"] = test["unique_id"]
    columns=["unique_id",'proba_1', 'proba_2', 'proba_3', 'proba_4', 'proba_5', 'proba_6', 'proba_7', 'proba_8', 'proba_9']
    a = a[columns]
    a.to_csv(v+"_sub.csv",index=False)


# In[ ]:


lr.fit(x_new,y_new)
y_pred = lr.predict_proba(test_df.values)


# In[ ]:


filesub("lr_first",y_pred,t)


# ## Things remaning

# 1. Differet models can be applied
# 2. Outliers can be removed
# 3. More feature engg can be done
