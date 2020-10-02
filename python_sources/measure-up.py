#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The main outcome of this competition is to uncover new insights in early childhood education and whether media can support learning outcomes. The data for this competition is provided by **PBS KIDS**, specifically from the **PBS KIDS Measure UP** app, a game based learning tool developed as part of CPB-PBS Ready to Learn initiative. The data used in this competition is anonymous, tabular data of interactions with the app. 

# ## About the Data
# 
# In PBS KIDS MeasureUp app, children navigate a map and complete various levels, which may be activities, video clips, games, or assessments. Each assessment is designed to test a child's comprehension of a certain set of measurement-related skills. There are five assessments: Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter.
# 
# The intent of the competition is to use the gameplay data to forecast how many attempts a child will take to pass a given assessment (an incorrect answer is counted as an attempt). Each application install is represented by an installation_id. This will typically correspond to one child, but there is some noise from issues such as shared devices. The training set, contains full history of gameplay data whereas the test set, has been truncated to the history after the start event of a single assessment, chosen randomly, for which we must predict the number of attempts. Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.
# 
# The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):
# 
#     3: the assessment was solved on the first attempt
#     2: the assessment was solved on the second attempt
#     1: the assessment was solved after 3 or more attempts
#     0: the assessment was never solved
# 
# The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains "correct":true.

# # Loading Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sns.set()


# # Loading and exploring necessary datasets

# In[ ]:


train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# In[ ]:


train_df.info()


# In[ ]:


cols_list = train_df.columns.tolist()
#for col in cols_list:
#    print(train_df[col].value_counts().sort_values(ascending=False))

cols_list


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.info()


# In[ ]:


specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

specs.head()


# In[ ]:


labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

labels.head()


# In[ ]:


#train_df = train_df.loc[(train_df['event_code'] == 4100) | (train_df['event_code'] == 4110)]
#train_df = train_df.loc[train_df['type'] == 'Assessment']

#train_df = pd.merge(train_df, specs, on='event_id', how='inner')

#train_df.head()


# # Data Wrangling
# 
# ## Training Dataframe
# ### Merging training and labels dataframes

# In[ ]:


train_df = pd.merge(train_df,labels,on=['game_session','installation_id'],how='inner')

train_df.info()


# ### Converting timestamp to datetime type and appending hour and dayofweek values to dataframes

# In[ ]:


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], infer_datetime_format=True)
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], infer_datetime_format=True)

train_df['event_hour'] = train_df['timestamp'].dt.hour
train_df['event_day'] = train_df['timestamp'].dt.dayofweek

test_df['event_hour'] = test_df['timestamp'].dt.hour
test_df['event_day'] = test_df['timestamp'].dt.dayofweek


# ### Converting categorical variables into indicator variables

# In[ ]:


train_df[['title_x','world','type']] = train_df[['title_x','world','type']].astype('category')

df=pd.get_dummies(train_df,columns=['title_x','world','type'], prefix=['title','world','type'])

df.info()


# ### Rename column names

# In[ ]:


df=df.rename(columns={"title_Bird Measurer (Assessment)":"BirdMeasurer","title_Cart Balancer (Assessment)":"CartBalancer",
                               "title_Cauldron Filler (Assessment)":"CauldronFiller","title_Chest Sorter (Assessment)":"ChestSorter",
                               "title_Mushroom Sorter (Assessment)":"MushroomSorter","world_CRYSTALCAVES":"CRYSTALCAVES",
                               "world_MAGMAPEAK":"MAGMAPEAK","world_TREETOPCITY":"TREETOPCITY","type_Assessment":"type"})


# ### Group training dataframe by installation_id and pick record with max timestamp value

# In[ ]:


train_df_gp = df.groupby('installation_id')['timestamp'].agg('max').reset_index()
df = pd.merge(df,train_df_gp,on=['installation_id','timestamp'],how='inner')

df.info()


# In[ ]:


df.head()


# In[ ]:


cols = ['game_session',
 'installation_id',
 'event_day',
 'event_hour',
 'BirdMeasurer',
 'CartBalancer',
 'CauldronFiller',
 'ChestSorter',
 'MushroomSorter',
 'CRYSTALCAVES',
 'MAGMAPEAK',
 'TREETOPCITY',
 'accuracy',
 'accuracy_group']

df = df[cols]
df.head()


# In[ ]:


#test_df = test_df.loc[(test_df['event_code'] == 4100) | (test_df['event_code'] == 4110)]
#test_df = test_df.loc[test_df['type'] == 'Assessment']


# ## Test Dataframe

# ### Converting categorical variables to indicator variables 

# In[ ]:


test_df[['title','world','type']] = test_df[['title','world','type']].astype('category')

df_test=pd.get_dummies(test_df,columns=['title','world','type'], prefix=['title','world','type'])


# ### Rename column names

# In[ ]:


df_test=df_test.rename(columns={"title_Bird Measurer (Assessment)":"BirdMeasurer","title_Cart Balancer (Assessment)":"CartBalancer",
                               "title_Cauldron Filler (Assessment)":"CauldronFiller","title_Chest Sorter (Assessment)":"ChestSorter",
                               "title_Mushroom Sorter (Assessment)":"MushroomSorter","world_CRYSTALCAVES":"CRYSTALCAVES",
                               "world_MAGMAPEAK":"MAGMAPEAK","world_TREETOPCITY":"TREETOPCITY","type_Assessment":"type"})


# ### Group test dataframe by installation_id and pick record with max timestamp value

# In[ ]:


test_df_gp = df_test.groupby('installation_id')['timestamp'].agg('max').reset_index()
test_df_merge = pd.merge(df_test,test_df_gp,on=['installation_id','timestamp'],how='inner')


# In[ ]:


test_df_merge.info()


# In[ ]:


cols_test = ['game_session',
 'installation_id',
 'event_day',
 'event_hour',
 'BirdMeasurer',
 'CartBalancer',
 'CauldronFiller',
 'ChestSorter',
 'MushroomSorter',
 'CRYSTALCAVES',
 'MAGMAPEAK',
 'TREETOPCITY',
]

test_df_subset = test_df_merge[cols_test]

test_df_subset.head()


# # EDA

# In[ ]:


cat1 = sns.catplot(x="BirdMeasurer", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat1


# In[ ]:



cat2 = sns.catplot(x="CartBalancer", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat2


# In[ ]:


cat3 = sns.catplot(x="CauldronFiller", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat3


# In[ ]:


cat4 = sns.catplot(x="ChestSorter", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat4


# In[ ]:


cat5 = sns.catplot(x="MushroomSorter", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat5


# In[ ]:


cat6 = sns.catplot(x="CRYSTALCAVES", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat6


# In[ ]:


cat7 = sns.catplot(x="MAGMAPEAK", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat7


# In[ ]:


cat8 = sns.catplot(x="TREETOPCITY", hue="accuracy_group", data=df,
                height=6, aspect=1.5, kind="count", palette="colorblind")

cat8


# # Model definitions

# In[ ]:


def perform_logistic_regression(df_X, df_Y, test_df_X):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(df_X, df_Y)
    pred_Y = logistic_regression.predict(test_df_X)
    accuracy = round(logistic_regression.score(df_X, df_Y) * 100,2)
    returnval = {'model':'Logistic Regression','accuracy':accuracy}
    return returnval


# In[ ]:


def perform_svc(df_X, df_Y, test_df_X):
    svc_clf = SVC()
    svc_clf.fit(df_X, df_Y)
    pred_Y = svc_clf.predict(test_df_X)
    accuracy = round(svc_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'SVC', 'accuracy':accuracy}
    return returnval


# In[ ]:


def perform_linear_svc(df_X, df_Y, test_df_X):
    svc_linear_clf = LinearSVC()
    svc_linear_clf.fit(df_X, df_Y)
    pred_Y = svc_linear_clf.predict(test_df_X)
    accuracy = round(svc_linear_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'LinearSVC', 'accuracy':accuracy}
    return returnval


# In[ ]:


def perform_rfc(df_X, df_Y, test_df_X):
    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)
    rfc_clf.fit(df_X, df_Y)
    pred_Y = rfc_clf.predict(test_df_X)
    accuracy = round(rfc_clf.score(df_X, df_Y) * 100, 2)
    returnval = {'model':'RandomForestClassifier','accuracy':accuracy}
    return returnval


# In[ ]:


def perform_knn(df_X, df_Y, test_df_X):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_X, df_Y)
    pred_Y = knn.predict(test_df_X)
    accuracy = round(knn.score(df_X, df_Y) *100,2)
    returnval = {'model':'KNeighborsClassifier','accuracy':accuracy}
    return returnval


# In[ ]:


def perform_gnb(df_X, df_Y, test_df_X):
    gnb = GaussianNB()
    gnb.fit(df_X, df_Y)
    pred_Y = gnb.predict(test_df_X)
    accuracy = round(gnb.score(df_X, df_Y)*100,2)
    returnval = {'model':'GaussianNB','accuracy':accuracy}
    return returnval


# In[ ]:


def perform_dtree(df_X, df_Y, test_df_X):
    dtree = DecisionTreeClassifier()
    dtree.fit(df_X, df_Y)
    pred_Y = dtree.predict(test_df_X)
    accuracy = round(dtree.score(df_X, df_Y)*100,2)
    returnval = {'model':'DecisionTreeClassifier','accuracy':accuracy}
    return returnval


# In[ ]:


def perform_linear_regression(df_X, df_Y, test_df_X):
    linear_regression = LinearRegression()
    linear_regression.fit(df_X, df_Y)
    pred_Y = linear_regression.predict(test_df_X)
    # size_y = pred_Y.size
    # cks = cohen_kappa_score(df_Y[:size_y], pred_Y, weights="quadratic")
    accuracy = round(linear_regression.score(df_X, df_Y)*100,2)
    returnval = {'model':'LinearRegression','accuracy':accuracy}
    return returnval


# # Preparing training and test dataframes for model evaluation

# In[ ]:


X = df.drop(['game_session','installation_id','accuracy','accuracy_group'],axis=1)
y = df['accuracy_group']

test_X = test_df_subset.drop(['game_session','installation_id'],axis=1)


# ## Evaluating models

# In[ ]:


linreg_val = perform_linear_regression(X, y, test_X)
lr_val = perform_logistic_regression(X, y, test_X)
svc_val = perform_svc(X, y, test_X)
svc_lin_val = perform_linear_svc(X, y, test_X)
rfc_val = perform_rfc(X, y, test_X)
knn_val = perform_knn(X, y, test_X)
gnb_val = perform_gnb(X, y, test_X)
dtree_val = perform_dtree(X, y, test_X)
    
model_accuracies = pd.DataFrame()
model_accuracies = model_accuracies.append([linreg_val, lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val])
# [linreg_val, lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val]
cols = list(model_accuracies.columns.values)
cols = cols[-1:] + cols[:-1]
model_accuracies = model_accuracies[cols]
model_accuracies = model_accuracies.sort_values(by='accuracy')
print(model_accuracies)
plt.figure()
plt.xticks(rotation=90)
sns.barplot(x='model', y='accuracy', data=model_accuracies)


# # Predicting output
# 
# Based on the model accuries from above section it is evident that RandomForestClassifier has highest accuracy value. So I am going to use it to predict output. 

# In[ ]:


lg = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None).fit(X, y)

y_pred = lg.predict(test_X)

y_pred = lg.predict(test_X)

test_X['accuracy_group'] = y_pred

test_X['accuracy_group'] = test_X['accuracy_group'].astype('int')

test_X['installation_id'] = test_df_subset['installation_id']

final_df = test_X[['installation_id','accuracy_group']]
final_df.to_csv('submission.csv',sep=',',index=False)

#final_df['accuracy_group'].value_counts()


# # Summary
# 
# Using RandomForestClassifier has given public score of 0.279 which is the highest so far I achieved. I know there is lot to improve on my part in terms of data manipulation, feature selection and picking the right model. Please provide any inputs that will help me to improve.
