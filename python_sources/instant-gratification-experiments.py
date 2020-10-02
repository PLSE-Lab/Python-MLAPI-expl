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


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# read the dataset
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


# print shape of datasets
print('Shape of the train dataset : ', train_data.shape)
print('Shape of the test dataset : ', test_data.shape)


# #### First things first, Let's handle the missing values from dataset.

# In[ ]:


# let's check that if there is missing values in the dataset
print('Missing Values in Train Dataset : ', train_data.isnull().sum().sum())
print('Missing Values in Test Dataset : ', test_data.isnull().sum().sum())


# #### Distribution Of Classes In Label Column

# In[ ]:


train_data.target.hist()


# From the graph it is clear that there is no class imbalance problem.

# In[ ]:


# There are 258 attributes in the train dataset, which is infact not desirable
# we need to select only a subset of columns (dimensionality reduction)
# for that let's check that if there strong correlation exist between them
f, ax = plt.subplots(figsize=(10, 8))
corr = train_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[ ]:


# handlng the categorical features
train_data.select_dtypes(exclude=["number","bool_","object_"]).columns


# In[ ]:


labels = train_data['target']
train_data = train_data.drop(['id','target'], axis=1)


# Which shows that there is no correlation at all. Hmm, so what are the hidden patterns that I'm looking at!
# 

# ### Logistic Regression

# In[ ]:


# let us try a logistic regression model on the dataset

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.25, random_state = 0)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred)
print('ROC AUC for LR =',round(auc,5))


# It is ~ 0.5, which says the model will predict an ouput with 50% confidence. <br>
# Infact a random value generator can perform better than this. <br>
# Which shows that this is the worst case that can happen. <br>
# Why do we get such a kind of result ? Does it have anything to do with the results of the correlation matrix ?

# #### Step back and Start again

# In[ ]:


train_data['target'] = labels


# In[ ]:


# the medians of columns when we group the data by 'target' feature
target_medians = train_data.groupby("target").median()
target_medians


# In[ ]:


# let's calculate the difference b/w row-1 and row-2
sorted_target_distance = np.abs(target_medians.iloc[0]-target_medians.iloc[1]).sort_values(ascending=False)


# In[ ]:


sorted_target_distance.head() # they do posses large difference b/w the medians


# In[ ]:


sorted_target_distance.tail()


# Zero! What's happening here?

# To understand the effect of these distribution let us comapare the 2nd attribute and last one to the attribute in 1st solution.

# In[ ]:


fig, ax = plt.subplots(2,2,figsize=(20,10))
sns.distplot(train_data.loc[train_data.target==0, "wheezy-myrtle-mandrill-entropy"], color="Blue", ax=ax[0,0])
sns.distplot(train_data.loc[train_data.target==1, "wheezy-myrtle-mandrill-entropy"], color="Red", ax=ax[0,0])
sns.distplot(train_data.loc[train_data.target==0, "wheezy-copper-turtle-magic"], color="Blue", ax=ax[0,1])
sns.distplot(train_data.loc[train_data.target==1, "wheezy-copper-turtle-magic"], color="Red", ax=ax[0,1])
ax[1,0].scatter(train_data["wheezy-myrtle-mandrill-entropy"].values,
                train_data["skanky-carmine-rabbit-contributor"].values, c=train_data.target.values,
                cmap="coolwarm", s=1, alpha=0.5)
ax[1,0].set_xlabel("wheezy-myrtle-mandrill-entropy")
ax[1,0].set_ylabel("skanky-carmine-rabbit-contributor")
ax[1,1].scatter(train_data["wheezy-myrtle-mandrill-entropy"].values,
                train_data["wheezy-copper-turtle-magic"].values, c=train_data.target.values,
                cmap="coolwarm", s=1, alpha=0.5)
ax[1,1].set_xlabel("wheezy-myrtle-mandrill-entropy")
ax[1,1].set_ylabel("wheezy-copper-turtle-magic");


# The 2nd and 4th graph shows that the distribution of <b>wheezy-copper-turtle-magic</b> is spread across the space around it.

# ### The effect of 'wheezy-copper-turtle-magic'

# In[ ]:


# consider the distribution of first three attributes from above list

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

feat1 = "wheezy-myrtle-mandrill-entropy"
feat2 = "skanky-carmine-rabbit-contributor"
feat3 = "thirsty-carmine-corgi-ordinal"

N = 10000

trace1 = go.Scatter3d(
    x=train_data[feat1].values[0:N], 
    y=train_data[feat2].values[0:N],
    z=train_data[feat3].values[0:N],
    mode='markers',
    marker=dict(
        color=train_data.target.values[0:N],
        colorscale = "Jet",
        opacity=0.3,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = 'The turtle place',
    scene = dict(
        xaxis = dict(title=feat1),
        yaxis = dict(title=feat2),
        zaxis = dict(title=feat3),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[ ]:


# consider the distribution of first two attributes and the last one 'wheezy-copper-turtle-magic' from above list

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

feat1 = "wheezy-myrtle-mandrill-entropy"
feat2 = "skanky-carmine-rabbit-contributor"
feat3 = "wheezy-copper-turtle-magic"

N = 10000

trace1 = go.Scatter3d(
    x=train_data[feat1].values[0:N], 
    y=train_data[feat2].values[0:N],
    z=train_data[feat3].values[0:N],
    mode='markers',
    marker=dict(
        color=train_data.target.values[0:N],
        colorscale = "Jet",
        opacity=0.3,
        size=2
    )
)

figure_data = [trace1]
layout = go.Layout(
    title = 'The turtle place',
    scene = dict(
        xaxis = dict(title=feat1),
        yaxis = dict(title=feat2),
        zaxis = dict(title=feat3),
    ),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    showlegend=True
)

fig = go.Figure(data=figure_data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# Previous LR model does not consider the interactions between the attributes. It treats the attributes as Independant Variables. <br> So we need to create an LR model by considering the interactions.

# In[ ]:


# let's check the distribution of 'wheezy-copper-turtle-magic'
train_data['wheezy-copper-turtle-magic'].hist()


# In[ ]:


train_data['wheezy-copper-turtle-magic'].describe()


# The distribution of <b>wheezy-copper-turtle-magic</b> is between (0, 512)

# ### Logistic Regression With Interactions

# We will create seperate 512 model for the unique values of the <b>wheezy-copper-turtle-magic</b>

# In[ ]:


# INITIALIZE VARIABLES
cols = [c for c in train_data.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
interactions = np.zeros((512,255))
oof = np.zeros(len(train_data))
preds = np.zeros(len(test_data))


# In[ ]:


from sklearn.model_selection import StratifiedKFold

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train_data[train_data['wheezy-copper-turtle-magic']==i]
    test2 = test_data[test_data['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    test2.reset_index(drop=True,inplace=True)
    
    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        # LOGISTIC REGRESSION MODEL
        clf = LogisticRegression()
        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]
        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0
        # RECORD INTERACTIONS
        for j in range(255):
            if clf.coef_[0][j]>0: interactions[i,j] = 1
            elif clf.coef_[0][j]<0: interactions[i,j] = -1
    if i%25==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train_data['target'],oof)
print('LR with interactions scores CV =',round(auc,5))


# In[ ]:


# submit results
test_data['target'] = preds
result_data = test_data[['id', 'target']]
result_data.head()


# In[ ]:


result_data.to_csv('sample_submission.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




