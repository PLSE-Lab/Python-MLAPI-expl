#!/usr/bin/env python
# coding: utf-8

# # FootBall Goal Prediction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ## Read Data

# In[ ]:


train=pd.read_csv('../input/google-ai-iiitdm/train.csv')


# In[ ]:


train.shape


# In[ ]:


train.head(10)


# In[ ]:


train.isna().sum()


# In[ ]:


train.dtypes


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
train = my_imputer.fit_transform(train.select_dtypes(exclude='object'))


# In[ ]:





# In[ ]:


train=train.drop(['shot_id_number','match_id','team_id','match_event_id'],axis=1)


# In[ ]:


train['location_x'].value_counts()


# In[ ]:


train['location_x']=train['location_x'].fillna(train['location_x'].mode()[0])


# In[ ]:


train['location_y'].value_counts()


# In[ ]:


train['location_y']=train['location_y'].fillna(train['location_y'].mode()[0])


# In[ ]:


train['remaining_min'].value_counts()


# In[ ]:


train['remaining_min']=train['remaining_min'].fillna(train['remaining_min'].mode()[0])


# In[ ]:


train['power_of_shot'].value_counts()


# In[ ]:


train['power_of_shot']=train['power_of_shot'].fillna(train['power_of_shot'].mode()[0])


# In[ ]:


train['knockout_match'].value_counts()


# In[ ]:


train=train.drop(['knockout_match'],axis=1)


# In[ ]:


train['game_season'].value_counts()


# In[ ]:


train['game_season']=train['game_season'].fillna(train['game_season'].mode()[0])


# In[ ]:


train['remaining_sec']=train['remaining_sec'].fillna(train['remaining_sec'].mean())


# In[ ]:


train['distance_of_shot'].value_counts()


# In[ ]:


train['distance_of_shot']=train['distance_of_shot'].fillna(train['distance_of_shot'].mode()[0])


# In[ ]:


train['area_of_shot'].value_counts()


# In[ ]:


train['area_of_shot']=train['area_of_shot'].fillna(train['area_of_shot'].mode()[0])


# In[ ]:


train['shot_basics'].value_counts()


# In[ ]:


train['shot_basics']=train['shot_basics'].fillna(train['shot_basics'].mode()[0])


# In[ ]:


train['range_of_shot'].value_counts()


# In[ ]:


train['range_of_shot']=train['range_of_shot'].fillna(train['range_of_shot'].mode()[0])


# In[ ]:


train['team_name'].value_counts()


# In[ ]:


train=train.drop(['team_name'],axis=1)


# In[ ]:


train=train.drop(['date_of_game'],axis=1)


# In[ ]:


train['lat/lng'].value_counts()


# In[ ]:


train['lat/lng']=train['lat/lng'].fillna(train['lat/lng'].mode()[0])


# In[ ]:


train['type_of_shot'].value_counts()


# In[ ]:


train['type_of_shot']=train['type_of_shot'].fillna(train['type_of_shot'].mode()[0])


# In[ ]:


train['type_of_combined_shot'].value_counts()


# In[ ]:


train['type_of_combined_shot']=train['type_of_combined_shot'].fillna(train['type_of_combined_shot'].mode()[0])


# In[ ]:


train['remaining_min.1'].value_counts()


# In[ ]:


train['remaining_min.1']=train['remaining_min.1'].fillna(train['remaining_min.1'].mean())


# In[ ]:


train['power_of_shot.1'].value_counts()


# In[ ]:


train['power_of_shot.1']=train['power_of_shot.1'].fillna(train['power_of_shot.1'].mean())


# In[ ]:


train['knockout_match.1'].value_counts()


# In[ ]:


train['knockout_match.1']=train['knockout_match.1'].fillna(train['knockout_match.1'].mode()[0])


# In[ ]:


train['remaining_sec.1'].value_counts()


# In[ ]:


train['remaining_sec.1']=train['remaining_sec.1'].fillna(train['remaining_sec.1'].mode()[0])


# In[ ]:


train['distance_of_shot.1'].value_counts()


# In[ ]:


train['distance_of_shot.1']=train['distance_of_shot.1'].fillna(train['distance_of_shot.1'].mode()[0])


# In[ ]:


train['home/away'].value_counts()


# In[ ]:


train=train.drop(['home/away'],axis=1)


# ## Text Features Preprocessing

# In[ ]:


preproc = []
for sent in train['game_season']:
    sent = sent.replace('-', '')
    preproc.append(sent)
train['game_season']=preproc


# In[ ]:





# In[ ]:


preproc = []
for sent in train['area_of_shot']:
    sent = sent.replace('(', '')
    sent = sent.replace(')', '')
    sent = sent.replace(' ', '')
    preproc.append(sent)
train['area_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['shot_basics']:
    sent = sent.replace(' ', '')
    preproc.append(sent)
train['shot_basics']=preproc


# In[ ]:


preproc = []
for sent in train['range_of_shot']:
    sent = sent.replace(' ', '')
    sent = sent.replace('.', '')
    sent = sent.replace('+', '')
    sent = sent.replace('-', '')
    preproc.append(sent)
train['range_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['type_of_shot']:
    sent = sent.replace(' - ', '')
    preproc.append(sent)
train['type_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['type_of_combined_shot']:
    sent = sent.replace(' - ', '')
    preproc.append(sent)
train['type_of_combined_shot']=preproc


# ## Train-Test Split

# In[ ]:


y=train['is_goal']


# In[ ]:


y.shape


# In[ ]:


y=y.astype(int)


# In[ ]:


train=train.drop(['is_goal'],axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, stratify=y)


# ## Make Data Model Ready: encoding numerical and categorical features

# ### Numerical Features Encoding

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
def numerical_norm(tr,te,col):
    scaler = MinMaxScaler()
    scaler.fit(tr[col].values.reshape(-1,1))

    X_train_norm = scaler.transform(tr[col].values.reshape(-1,1))
    X_test_norm = scaler.transform(te[col].values.reshape(-1,1))
    return X_train_norm,X_test_norm,scaler


# #### Column2

# In[ ]:


X_tr_location_x,X_test_location_x,scaler_x=numerical_norm(X_train,X_test,'location_x')


# In[ ]:


X_tr_location_x.shape


# #### Column3

# In[ ]:


X_tr_location_y,X_test_location_y,scaler_y=numerical_norm(X_train,X_test,'location_y')


# In[ ]:


X_tr_location_y.shape


# #### Column8

# In[ ]:


X_tr_sec,X_test_sec,scaler_sec=numerical_norm(X_train,X_test,'remaining_sec')


# In[ ]:


X_tr_sec.shape


# #### Column9

# In[ ]:


X_tr_dist,X_test_dist,scaler_dist=numerical_norm(X_train,X_test,'distance_of_shot')


# In[ ]:


X_tr_dist.shape


# In[ ]:


X_tr_min1,X_test_min1,scaler_min1=numerical_norm(X_train,X_test,'remaining_min.1')


# In[ ]:


X_tr_min1.shape


# In[ ]:


X_tr_pow1,X_test_pow1,scaler_pow1=numerical_norm(X_train,X_test,'power_of_shot.1')


# In[ ]:


X_tr_pow1.shape


# In[ ]:


X_tr_ko1,X_test_ko1,scaler_ko1=numerical_norm(X_train,X_test,'knockout_match.1')


# In[ ]:


X_tr_ko1.shape


# In[ ]:


X_tr_sec1,X_test_sec1,scaler_sec1=numerical_norm(X_train,X_test,'remaining_sec.1')


# In[ ]:


X_tr_sec1.shape


# In[ ]:


X_tr_dist1,X_test_dist1,scaler_dist1=numerical_norm(X_train,X_test,'distance_of_shot.1')


# In[ ]:


X_tr_dist1.shape


# ### Categorical Features Encoding

# In[ ]:


from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
def categorical_text_encoding(df,tr,te,col):
    my_counter = Counter()
    for word in df[col].values:
        my_counter.update(word.split())

    dictionary = dict(my_counter)
    sorted_dict = dict(sorted(dictionary.items(), key=lambda kv: kv[1]))
    
    vectorizer = CountVectorizer(vocabulary=list(sorted_dict.keys()), lowercase=False, binary=True)
    vectorizer.fit(tr[col].values)
    train = vectorizer.transform(tr[col].values)
    test = vectorizer.transform(te[col].values)
    print(vectorizer.get_feature_names())
    return train,test,vectorizer


# In[ ]:


X_tr_min,X_te_min=X_train['remaining_min'],X_test['remaining_min']


# In[ ]:


X_tr_min=X_tr_min.values.reshape(-1,1)
X_te_min=X_te_min.values.reshape(-1,1)


# In[ ]:


X_tr_min.shape


# In[ ]:


X_tr_pow,X_te_pow=X_train['power_of_shot'],X_test['power_of_shot']


# In[ ]:


X_tr_pow=X_tr_pow.values.reshape(-1,1)
X_te_pow=X_te_pow.values.reshape(-1,1)


# In[ ]:


X_tr_pow.shape


# In[ ]:


X_tr_game,X_te_game,vectorizer_game=categorical_text_encoding(train,X_train,X_test,'game_season')


# In[ ]:


X_tr_game.shape


# In[ ]:


X_tr_area,X_te_area,vectorizer_area=categorical_text_encoding(train,X_train,X_test,'area_of_shot')


# In[ ]:


X_tr_area.shape


# In[ ]:


X_tr_basics,X_te_basics,vectorizer_basics=categorical_text_encoding(train,X_train,X_test,'shot_basics')


# In[ ]:


X_tr_basics.shape


# In[ ]:


X_tr_range,X_te_range,vectorizer_range=categorical_text_encoding(train,X_train,X_test,'range_of_shot')


# In[ ]:


X_tr_range.shape


# In[ ]:


X_tr_type,X_te_type,vectorizer_type=categorical_text_encoding(train,X_train,X_test,'type_of_shot')


# In[ ]:


X_tr_type.shape


# In[ ]:


X_tr_type_comb,X_te_type_comb,vectorizer_type_comb=categorical_text_encoding(train,X_train,X_test,'type_of_combined_shot')


# In[ ]:


X_tr_type_comb.shape


# ## Preparing Data Matrix

# In[ ]:


from scipy.sparse import hstack
from scipy import sparse


# In[ ]:


X_tr=hstack([X_tr_dist,X_tr_min1,X_tr_pow1,X_tr_ko1,X_tr_sec1,X_tr_dist1,X_tr_min,X_tr_pow,X_tr_area,X_tr_basics,X_tr_range,X_tr_type,X_tr_type_comb]).tocsr()


# In[ ]:


X_tr.shape


# In[ ]:


X_te=hstack([X_test_dist,X_test_min1,X_test_pow1,X_test_ko1,X_test_sec1,X_test_dist1,X_te_min,X_te_pow,X_te_area,X_te_basics,X_te_range,X_te_type,X_te_type_comb]).tocsr()


# In[ ]:


X_te.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


def my_score(y_actual, y_predicted):
    epsilon = 1.0/(1.0+mean_absolute_error(y_actual,y_predicted))
    return epsilon


# In[ ]:


from sklearn.metrics import make_scorer
ftwo_scorer = make_scorer(my_score)


# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import GridSearchCV\nfrom sklearn.tree import DecisionTreeClassifier\n\ntree = DecisionTreeClassifier()\n\nparameters = {'max_depth': [1,5,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], 'min_samples_split': [50,60,100,150,200,250,300,350,400,450,500,550,600,650,700]}\n\nclassifier = GridSearchCV(tree, parameters, cv=3, return_train_score=True,scoring=ftwo_scorer)\nselect = classifier.fit(X_tr, y_train)")


# In[ ]:


select.best_params_


# In[ ]:


dt = DecisionTreeClassifier(max_depth= select.best_params_['max_depth'], min_samples_split= select.best_params_['min_samples_split'])
dt.fit(X_tr, y_train)

y_train_pred = dt.predict(X_tr)    
y_test_pred = dt.predict(X_te)


# In[ ]:


epsilon_train=1.0/(1.0+mean_absolute_error(y_train,y_train_pred))


# In[ ]:


epsilon_test=1.0/(1.0+mean_absolute_error(y_test,y_test_pred))


# In[ ]:


epsilon_train


# In[ ]:


epsilon_test


# ## Read Data

# In[ ]:


train=pd.read_csv('../input/google-ai-iiitdm/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


train.isna().sum()


# In[ ]:


train=train.drop(['match_id','team_id','match_event_id'],axis=1)


# In[ ]:


train['location_x'].value_counts()


# In[ ]:


train['location_x']=train['location_x'].fillna(train['location_x'].mode()[0])


# In[ ]:


train['location_y'].value_counts()


# In[ ]:


train['location_y']=train['location_y'].fillna(train['location_y'].mode()[0])


# In[ ]:


train['remaining_min'].value_counts()


# In[ ]:


train['remaining_min']=train['remaining_min'].fillna(train['remaining_min'].mode()[0])


# In[ ]:


train['power_of_shot'].value_counts()


# In[ ]:


train['power_of_shot']=train['power_of_shot'].fillna(train['power_of_shot'].mode()[0])


# In[ ]:


train['knockout_match'].value_counts()


# In[ ]:


train=train.drop(['knockout_match'],axis=1)


# In[ ]:


train['game_season'].value_counts()


# In[ ]:


train['game_season']=train['game_season'].fillna(train['game_season'].mode()[0])


# In[ ]:


train['remaining_sec']=train['remaining_sec'].fillna(train['remaining_sec'].mean())


# In[ ]:


train['distance_of_shot'].value_counts()


# In[ ]:


train['distance_of_shot']=train['distance_of_shot'].fillna(train['distance_of_shot'].mode()[0])


# In[ ]:


train['area_of_shot'].value_counts()


# In[ ]:


train['area_of_shot']=train['area_of_shot'].fillna(train['area_of_shot'].mode()[0])


# In[ ]:


train['shot_basics'].value_counts()


# In[ ]:


train['shot_basics']=train['shot_basics'].fillna(train['shot_basics'].mode()[0])


# In[ ]:


train['range_of_shot'].value_counts()


# In[ ]:


train['range_of_shot']=train['range_of_shot'].fillna(train['range_of_shot'].mode()[0])


# In[ ]:


train['team_name'].value_counts()


# In[ ]:


train=train.drop(['team_name'],axis=1)


# In[ ]:


train=train.drop(['date_of_game'],axis=1)


# In[ ]:


train['lat/lng'].value_counts()


# In[ ]:


train['lat/lng']=train['lat/lng'].fillna(train['lat/lng'].mode()[0])


# In[ ]:


train['type_of_shot'].value_counts()


# In[ ]:


train['type_of_shot']=train['type_of_shot'].fillna(train['type_of_shot'].mode()[0])


# In[ ]:


train['type_of_combined_shot'].value_counts()


# In[ ]:


train['type_of_combined_shot']=train['type_of_combined_shot'].fillna(train['type_of_combined_shot'].mode()[0])


# In[ ]:


train['remaining_min.1'].value_counts()


# In[ ]:


train['remaining_min.1']=train['remaining_min.1'].fillna(train['remaining_min.1'].mean())


# In[ ]:


train['power_of_shot.1'].value_counts()


# In[ ]:


train['power_of_shot.1']=train['power_of_shot.1'].fillna(train['power_of_shot.1'].mean())


# In[ ]:


train['knockout_match.1'].value_counts()


# In[ ]:


train['knockout_match.1']=train['knockout_match.1'].fillna(train['knockout_match.1'].mode()[0])


# In[ ]:


train['remaining_sec.1'].value_counts()


# In[ ]:


train['remaining_sec.1']=train['remaining_sec.1'].fillna(train['remaining_sec.1'].mode()[0])


# In[ ]:


train['distance_of_shot.1'].value_counts()


# In[ ]:


train['distance_of_shot.1']=train['distance_of_shot.1'].fillna(train['distance_of_shot.1'].mode()[0])


# In[ ]:


train['home/away'].value_counts()


# In[ ]:


train=train.drop(['home/away'],axis=1)


# ## Text Features Preprocessing

# In[ ]:


preproc = []
for sent in train['game_season']:
    sent = sent.replace('-', '')
    preproc.append(sent)
train['game_season']=preproc


# In[ ]:


preproc = []
for sent in train['area_of_shot']:
    sent = sent.replace('(', '')
    sent = sent.replace(')', '')
    sent = sent.replace(' ', '')
    preproc.append(sent)
train['area_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['shot_basics']:
    sent = sent.replace(' ', '')
    preproc.append(sent)
train['shot_basics']=preproc


# In[ ]:


preproc = []
for sent in train['range_of_shot']:
    sent = sent.replace(' ', '')
    sent = sent.replace('.', '')
    sent = sent.replace('+', '')
    sent = sent.replace('-', '')
    preproc.append(sent)
train['range_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['type_of_shot']:
    sent = sent.replace(' - ', '')
    preproc.append(sent)
train['type_of_shot']=preproc


# In[ ]:


preproc = []
for sent in train['type_of_combined_shot']:
    sent = sent.replace(' - ', '')
    preproc.append(sent)
train['type_of_combined_shot']=preproc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Make Data Model Ready: encoding numerical and categorical features

# ### Numerical Features Encoding

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
def numerical_norm(tr,scaler,col):
    X_train_norm = scaler.transform(tr[col].values.reshape(-1,1))
    return X_train_norm


# #### Column2

# In[ ]:


X_tr_location_x=numerical_norm(train,scaler_x,'location_x')


# In[ ]:


X_tr_location_x.shape


# #### Column3

# In[ ]:


X_tr_location_y=numerical_norm(train,scaler_y,'location_y')


# In[ ]:


X_tr_location_y.shape


# #### Column8

# In[ ]:


X_tr_sec=numerical_norm(train,scaler_sec,'remaining_sec')


# In[ ]:


X_tr_sec.shape


# #### Column9

# In[ ]:


X_tr_dist=numerical_norm(train,scaler_dist,'distance_of_shot')


# In[ ]:


X_tr_dist.shape


# In[ ]:


X_tr_min1=numerical_norm(train,scaler_min1,'remaining_min.1')


# In[ ]:


X_tr_min1.shape


# #### Column19

# In[ ]:


X_tr_pow1=numerical_norm(train,scaler_pow1,'power_of_shot.1')


# In[ ]:


X_tr_pow1.shape


# #### Column20

# In[ ]:


X_tr_ko1=numerical_norm(train,scaler_ko1,'knockout_match.1')


# In[ ]:


X_tr_ko1.shape


# #### Column21

# In[ ]:


X_tr_sec1=numerical_norm(train,scaler_sec1,'remaining_sec.1')


# In[ ]:


X_tr_sec1.shape


# #### Column22

# In[ ]:


X_tr_dist1=numerical_norm(train,scaler_dist1,'distance_of_shot.1')


# In[ ]:


X_tr_dist1.shape


# ### Categorical Features Encoding

# In[ ]:


from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
def categorical_text_encoding(tr,vectorizer,col):
    train = vectorizer.transform(tr[col].values)
    print(vectorizer.get_feature_names())
    return train


# #### Column4

# In[ ]:


X_tr_min=train['remaining_min']


# In[ ]:


X_tr_min=X_tr_min.values.reshape(-1,1)


# In[ ]:


X_tr_min.shape


# #### Column5

# In[ ]:


X_tr_pow=train['power_of_shot']


# In[ ]:


X_tr_pow=X_tr_pow.values.reshape(-1,1)


# In[ ]:


X_tr_pow.shape


# #### Column7

# In[ ]:


X_tr_game=categorical_text_encoding(train,vectorizer_game,'game_season')


# In[ ]:


X_tr_game.shape


# #### Column10

# In[ ]:


X_tr_area=categorical_text_encoding(train,vectorizer_area,'area_of_shot')


# In[ ]:


X_tr_area.shape


# #### Column11

# In[ ]:


X_tr_basics=categorical_text_encoding(train,vectorizer_basics,'shot_basics')


# In[ ]:


X_tr_basics.shape


# #### Column12

# In[ ]:


X_tr_range=categorical_text_encoding(train,vectorizer_range,'range_of_shot')


# In[ ]:


X_tr_range.shape


# #### Column16

# In[ ]:


X_tr_type=categorical_text_encoding(train,vectorizer_type,'type_of_shot')


# In[ ]:


X_tr_type.shape


# #### Column17

# In[ ]:


X_tr_type_comb=categorical_text_encoding(train,vectorizer_type_comb,'type_of_combined_shot')


# In[ ]:


X_tr_type_comb.shape


# #### Column23

# ## Preparing Data Matrix

# In[ ]:


X_tr=hstack([X_tr_dist,X_tr_min1,X_tr_pow1,X_tr_ko1,X_tr_sec1,X_tr_dist1,X_tr_min,X_tr_pow,X_tr_area,X_tr_basics,X_tr_range,X_tr_type,X_tr_type_comb]).tocsr()


# ## Applying Models

# In[ ]:


output1=dt.predict(X_tr)


# In[ ]:


train['is_goal']=output1


# In[ ]:


my_submission = pd.DataFrame({'shot_id_number': train['shot_id_number'], 'is_goal': train['is_goal']})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:


my_submission.shape


# In[ ]:





# In[ ]:





# In[ ]:




