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


__author__ = 'Its Me!'


# In[ ]:


import xgboost as xgb

from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=2000)
test = pd.read_csv('../input/test.csv', nrows=2000)


# In[ ]:


# To get a quick understanding of data
summary = train.describe()
summary = summary.transpose()
summary.head()


# In[ ]:


# str(train)    # too big data for str


# In[ ]:


train_corr = train.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.7
#number of features considered
size = 15
#get the names of all the columns
cols=train.columns 
# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (train_corr.iloc[i,j] >= threshold and train_corr.iloc[i,j] < 1) or (train_corr.iloc[i,j] < 0 and train_corr.iloc[i,j] <= -threshold):
            corr_list.append([train_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
train_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in train_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# In[ ]:


train_corr_list


# In[ ]:


train_corr_list = np.array(train_corr_list)


# In[ ]:


train_corr_list = train_corr_list[:,[1,2]]


# In[ ]:


#np.unique(train_corr_list) 
#train_corr_list.ravel()
#np.median(train_corr_list)


# In[ ]:


from scipy.stats import mode
a=mode(train_corr_list.ravel(), axis=None)


# In[ ]:


a


# In[ ]:


test_corr = test.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (test_corr.iloc[i,j] >= threshold and test_corr.iloc[i,j] < 1) or (test_corr.iloc[i,j] < 0 and test_corr.iloc[i,j] <= -threshold):
            corr_list.append([test_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
test_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in test_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# In[ ]:


# http://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
def correlation_matrix(df,cols):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Some Feature Correlation')
    labels=cols
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

correlation_matrix(train,cols)


# In[ ]:


remove_cols=['cat12','cat11','cat13','cat10','cat7']   
train.drop(remove_cols, axis=1)
test.drop(remove_cols, axis=1)
train.head(5)


# In[ ]:


# review the coorelation with the correlated columns removed
correlation_matrix(train,cols)
correlation_matrix(test,cols)


# In[ ]:


train.columns.to_series().groupby(train.dtypes).groups


# In[ ]:


# if any object data type, then convert them to category
cat_train_cols = train.select_dtypes(['object']).columns
for x in cat_train_cols:
    train[x] = train[x].astype('category')


# In[ ]:


cat_train_cols = train.select_dtypes(['category']).columns


# In[ ]:


train[cat_train_cols] = train[cat_train_cols].apply(lambda x: x.cat.codes)


# In[ ]:


train.head(5)


# In[ ]:


train.columns.to_series().groupby(train.dtypes).groups


# In[ ]:


#list(train.columns.values)
#list(train)
#train.columns.values.tolist()
##X_cols = train.columns.difference(['id','loss']).values.tolist()
X_cols = train.select_dtypes(['float64']).columns.difference(['loss'])
#y_cols = train[['loss']].columns.values.tolist()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model = LogisticRegression()


# In[ ]:


# y_cols = np.ravel(y_cols)
y_cols = np.asarray(train['loss'], dtype="|S6")


# In[ ]:


model = model.fit(train[X_cols],y_cols)
#train[X_cols]
#y_cols


# In[ ]:


# check the accuracy on the training set
model.score(train[X_cols], y_cols)


# In[ ]:


from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier)
from sklearn.cross_validation import cross_val_score


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(train[X_cols],y_cols)


# In[ ]:


rf.score(train[X_cols], y_cols)


# In[ ]:


y_cols = rf.predict(test[X_cols])


# In[ ]:


rf.score(test[X_cols], y_cols)


# In[ ]:


scores = cross_val_score(LogisticRegression(), test[X_cols], y_cols, scoring='accuracy', cv=5)
print(scores)
print(scores.mean())


# In[ ]:


preds = pd.DataFrame({"id": test['id'], "loss": y_cols})
preds.to_csv('AllStateClaimsSeverity_yyyymmdd.csv', index=False)


# In[ ]:


print(check_output(["ls"]).decode("utf8"))

