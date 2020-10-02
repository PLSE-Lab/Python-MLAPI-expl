#!/usr/bin/env python
# coding: utf-8

# ## 2019 Data Science Bowl
# EXPLORATORY DATA ANALYSIS NOTEBOOK
# 
# Using use the gameplay data to forecast how many attempts a child will take to pass a given assessment (an incorrect answer is counted as an attempt). Each application install is represented by an installation_id. This will typically correspond to one child, but you should expect noise from issues such as shared devices. In the training set, you are provided the full history of gameplay data. In the test set, we have truncated the history after the start event of a single assessment, chosen randomly, for which you must predict the number of attempts. Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.
# 
# To see full data description click this link: https://www.kaggle.com/c/data-science-bowl-2019/data
# 
# *Author notes: The analysis will include gameplay, and specs (?)*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as train_valid_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import json

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# First we need to load the important dataset that will be included in the analysis

# In[ ]:


gameplay = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
gameplay_test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
gameplay.info()


# ## Gameplay
# 
# The competition provided variable description that can help us understand the dataset better.
# 
# * event_id - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
# * game_session - Randomly generated unique identifier grouping events within a single game or video play session.
# * timestamp - Client-generated datetime
# * event_data - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.
# * installation_id - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# * event_count - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# * event_code - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
# * game_time - Time in milliseconds since the start of the game session. Extracted from event_data.
# * title - Title of the game or video.
# * type - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# * world - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).
# 
# To see full data description click this link: https://www.kaggle.com/c/data-science-bowl-2019/data
# 
# *Author Notes: For the mean time we will not handle the unstructured data fromat of the event_data column, also will be removing the randomly generated identifier variables*

# We can see the distribution of event code and event count. 

# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot(x='event_code',y='event_count',data=gameplay)
plt.plot();


# It's also important to us to know the distribution of gameplay duration. Since there was a lot of zeros... we will try to see the distribution of non-zero game times.

# In[ ]:


g = gameplay.game_time.replace(0,np.nan).dropna().value_counts()
plt.hist(g.values,bins=100)
plt.xlabel('GameTime')
plt.ylabel('Count')
plt.show()


# We want to know the proportion of different worlds in the dataset. We can see below that majority of the worlds played fall into the MAGMAPEAK category

# In[ ]:


g = gameplay.world.value_counts()
plt.bar(g.index,g.values)
plt.xlabel('Worlds')
plt.ylabel('Count')
plt.show()


# We also want to know the proportion of media types. We can see below that majority of the media types played falls into the GAME category

# In[ ]:


g = gameplay.type.value_counts()
plt.bar(g.index,g.values)
plt.xlabel('Media Type')
plt.ylabel('Count')
plt.show()


# We can also see the distribution of Title of the game or video that was used. The top title was the "Bottle Filler (Activity)'

# In[ ]:


g = gameplay.title.value_counts()
plt.figure(figsize=(5,8))
plt.barh(g.index,g.values)
plt.xlabel('Count')
plt.ylabel('Media Title')
plt.show()


# ## Labels
# 
# The outcomes in this competition are grouped into 4 groups:
# 
# * 3: the assessment was solved on the first attempt <br/>
# * 2: the assessment was solved on the second attempt<br/>
# * 1: the assessment was solved after 3 or more attempts<br/>
# ![](http://)* 0: the assessment was never solved

# In[ ]:


labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
labels.info()


# We can see that majority of the accuracy groups falls into the #3 type. Which the subject was able to solve the assessment in the first assessment

# In[ ]:


labels['accuracy_group'].value_counts().plot(kind='bar');


# In[ ]:


labels['num_correct'].value_counts().plot(kind='bar');


# In[ ]:


labels['num_incorrect'].value_counts().plot(kind='bar');


# In[ ]:


labels['accuracy'].value_counts().plot(kind='hist',bins=100);


# ## Random Forest Modeling
# 
# This section will include modeling prep, modeling and modeling evaluation

# In[ ]:


gameplay = pd.merge(gameplay,labels[['game_session','accuracy_group']],on=['game_session'],how='left')
gameplay.head()


# ### Data Prep for Modeling

# In[ ]:


def dt_parts(df,dt_col):
    if(df[dt_col].dtype=='O'):
        df[dt_col] = pd.to_datetime(df[dt_col])
    df['year'] = df[dt_col].dt.year.astype(np.int16)
    df['month'] = df[dt_col].dt.month.astype(np.int8)
    df['day'] = df[dt_col].dt.day.astype(np.int8)
    df['hour'] = df[dt_col].dt.hour.astype(np.int8)
    df['minute'] = df[dt_col].dt.minute.astype(np.int8)
    df['second'] = df[dt_col].dt.second.astype(np.int8)
    df.drop(dt_col,axis=1,inplace=True)
    return df
    
def category_mapping(df,map_dict):
    for col in map_dict.keys():
        df[col] = df[col].map(map_dict[col])
        df[col] = df[col].astype(np.int16)
    return df


# In[ ]:


drop_cols = ['game_session','event_data','installation_id']
gameplay.drop(drop_cols,axis=1,inplace=True)
gameplay = dt_parts(gameplay,'timestamp')

gameplay_mapping = {}
cat_cols = gameplay.select_dtypes('object').columns
for col in cat_cols:
    values = list(gameplay[col].unique())+list(gameplay_test[col].unique())
    LE = LabelEncoder().fit(values)
    gameplay_mapping[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))
    
gameplay = category_mapping(gameplay,gameplay_mapping)
gameplay['accuracy_group'] = gameplay['accuracy_group'].fillna(method='bfill')
gameplay['accuracy_group'] = gameplay['accuracy_group'].fillna(method='ffill')
gameplay.head()


# In[ ]:


target_col = 'accuracy_group'
y = gameplay[target_col]
Xs = gameplay.drop(target_col,axis=1)

X_train, X_valid, y_train, y_valid = train_valid_split(Xs, y, test_size=0.2, random_state=0)
X_train.shape,X_valid.shape


# ### Modeling

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model = RandomForestClassifier(n_estimators=15,\n                              random_state=0,n_jobs=-1)\nmodel.fit(X_train,y_train)')


# In[ ]:


def get_evaluations(model):
    preds = model.predict(X_train)
    plt.hist(preds)
    plt.title('training predictions')
    plt.show();
    print('train_report',classification_report(y_train,preds))
    preds = model.predict(X_valid)
    plt.hist(preds)
    plt.title('validation predictions')
    plt.show();
    print('valid_report',classification_report(y_valid,preds))

get_evaluations(model)


# ## Submission Pipeline
# 
# *Author Notes: Still debugging the submission error for this code..*

# In[ ]:


# gameplay_test was already loaded at the start of this notebook
gameplay_test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
installation_id = gameplay_test['installation_id']
gameplay_test.head()


# In[ ]:


drop_cols = ['game_session','event_data','installation_id']
gameplay_test.drop(drop_cols,axis=1,inplace=True)
gameplay_test = dt_parts(gameplay_test,'timestamp')
gameplay_test = category_mapping(gameplay_test,gameplay_mapping)
gameplay_test.shape


# In[ ]:


preds_df = pd.DataFrame()
preds_df['installation_id'] = installation_id
preds_df['accuracy_group'] = model.predict(gameplay_test)

#this will be used to find which is the majority classif
preds_df['counter'] = 1
print(preds_df.shape)


# In[ ]:


preds_df = preds_df.groupby(['installation_id','accuracy_group'],as_index=False).sum()
preds_df['agg'] = preds_df.groupby(['installation_id'],as_index=False)['counter'].transform(np.mean)
preds_df = preds_df.sort_values('agg').drop_duplicates('installation_id')
preds_df = preds_df.sort_values('installation_id')
print(preds_df.shape)
preds_df.head()


# In[ ]:


sub_df = preds_df[['installation_id','accuracy_group']]
sub_df['accuracy_group'] = sub_df['accuracy_group'].astype(int)
sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv',index=False)
sub_df['accuracy_group'].hist()
plt.show()


# ## Notebook in progress
# 
# Do UPVOTE if this notebook is helpful to you in some way :) <br/>
# Comment below any suggetions that can help improve this notebook. TIA
