#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")\ntrain_labels = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")\ntrain = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")\nspecs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")\nsubmission = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")\nprint("Data Loaded!")')


# In[ ]:


train.head()


# In[ ]:


train_labels.head(4)


# In[ ]:


specs.head()


# In[ ]:


specs['info'][1]


# In[ ]:


test.sample(5)


# In[ ]:


train['title'].nunique()


# In[ ]:


plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
sns.set()
sns.set(style="darkgrid")
ax = sns.countplot(x=train['title'], data=train)


# In[ ]:


countsT = train["type"].value_counts()
values = list(range(4))
labels = 'Game' ,'Activity', 'Assessment', 'Clip'
sizes = countsT.values
explode = (0.1, 0.1, 0.1, 0.9)  # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()


# In[ ]:


countsT = train_labels["accuracy_group"].value_counts()
values = list(range(4))
labels = '3' ,'2', '1', '0'
sizes = countsT.values
explode = (0.1, 0.1, 0.1, 0.9)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()


# In[ ]:


countsT = train["world"].value_counts()
values = list(range(4))
labels = 'MAGMAPEAK' ,'CRYSTALCAVES', 'TREETOPCITY', 'NONE'
sizes = countsT.values
explode = (0.1, 0.1, 0.1, 0.9)  
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
ax1.set_title('Title to which Game/Video belongs to')


# In[ ]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train['date'] = train['timestamp'].dt.date
group2 = train.groupby(['date'])['event_id'].agg('count')
fig = go.Figure([go.Scatter(x=group2.index, y=group2.values, line_color= "#B22222", )])
fig.update_layout(title_text='Time Series for all Events')
fig.show()


# **DATA PREPARATION**

# **This part is taken from this kernel,check it out and upvote it if you like.**
# 
# https://www.kaggle.com/shahules/xgboost-starter-dsbowl

# In[ ]:


train_labels.drop(['num_correct','num_incorrect','accuracy','title'],axis=1,inplace=True)


# In[ ]:


train.drop(['event_data','date'],axis=1,inplace=True)


# In[ ]:


not_req=(set(train.installation_id.unique()) - set(train_labels.installation_id.unique()))
train = train[~train['installation_id'].isin(not_req)]
print(train.shape)


# In[ ]:


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear
    return df


# In[ ]:


def prepare_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day']=df['timestamp'].map(lambda x : int(x.hour))
    #one hot encoding on event code
    join_one=pd.get_dummies(df[['event_code','installation_id','game_session']],
                            columns=['event_code']).groupby(['installation_id','game_session'],as_index=False,sort=False).agg(sum)
    
    #dictionary to perform some aggregate functions after grouping
    agg={'event_count':sum,'hour_of_day':'mean','game_time':['sum','mean'],'event_id':'count'}
    
    join_two=df.drop(['timestamp'],axis=1).groupby(['installation_id','game_session'],as_index=False,sort=False).agg(agg)
    
    join_two.columns= [' '.join(col).strip() for col in join_two.columns.values]

    join_three=df[['installation_id','game_session','type','world','title']].groupby(['installation_id','game_session'],as_index=False,sort=False).first()
    
    join_four=(join_one.join(join_two.drop(['installation_id','game_session'],axis=1))).                         join(join_three.drop(['installation_id','game_session'],axis=1))
    return join_four


# In[ ]:


join_train=prepare_data(train)
cols=join_train.columns.to_list()[2:-3]
join_train[cols]=join_train[cols].astype('int16')


# In[ ]:


join_test=prepare_data(test)
cols=join_test.columns.to_list()[2:-3]
join_test[cols]=join_test[cols].astype('int16')


# In[ ]:


cols=join_test.columns[2:-8].to_list()
cols.append('event_id count')
cols.append('installation_id')


# In[ ]:


df=join_test[['hour_of_day mean','event_count sum','game_time mean','game_time sum',
    'installation_id']].groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=join_test[cols].groupby('installation_id',as_index=False,
                               sort=False).agg('sum').drop('installation_id',axis=1)

df_three=join_test[['title','type','world','installation_id']].groupby('installation_id',
         as_index=False,sort=False).last().drop('installation_id',axis=1)
        


# In[ ]:


final_train=pd.merge(train_labels,join_train,on=['installation_id','game_session'],
                                         how='left').drop(['game_session'],axis=1)

#final_test=join_test.groupby('installation_id',as_index=False,sort=False).last().drop(['game_session','installation_id'],axis=1)
final_test=(df.join(df_two)).join(df_three).drop('installation_id',axis=1)


# In[ ]:


df=final_train[['hour_of_day mean','event_count sum','game_time mean','game_time sum','installation_id']].     groupby('installation_id',as_index=False,sort=False).agg('mean')

df_two=final_train[cols].groupby('installation_id',as_index=False,
                                 sort=False).agg('sum').drop('installation_id',axis=1)

df_three=final_train[['accuracy_group','title','type','world','installation_id']].         groupby('installation_id',as_index=False,sort=False).         last().drop('installation_id',axis=1)

final_train=(df.join(df_two)).join(df_three).drop('installation_id',axis=1)


# In[ ]:


#concat train and test and Label Encode Categorical Columns

final=pd.concat([final_train,final_test])
encoding=['type','world','title']
for col in encoding:
    lb=LabelEncoder()
    lb.fit(final[col])
    final[col]=lb.transform(final[col])
    
final_train=final[:len(final_train)]
final_test=final[len(final_train):]


# In[ ]:


X_train=final_train.drop('accuracy_group',axis=1)
y_train=final_train['accuracy_group']


# **Model**

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npars = {\n    'colsample_bytree': 0.5,                 \n    'learning_rate': 0.01,\n    'max_depth': 10,\n    'subsample': 0.5,\n    'objective':'multi:softprob',\n    'num_class':4\n}\n\nkf = KFold(n_splits=10, shuffle=True, random_state=42)\ny_pre=np.zeros((len(final_test),4),dtype=float)\nfinal_test=xgb.DMatrix(final_test.drop('accuracy_group',axis=1))\n\n\nfor train_index, val_index in kf.split(X_train):\n    train_X = X_train.iloc[train_index]\n    val_X = X_train.iloc[val_index]\n    train_y = y_train[train_index]\n    val_y = y_train[val_index]\n    xgb_train = xgb.DMatrix(train_X, train_y)\n    xgb_eval = xgb.DMatrix(val_X, val_y)\n    \n    xgb_model = xgb.train(pars,\n                  xgb_train,\n                  num_boost_round=10000,\n                  evals=[(xgb_train, 'train'), (xgb_eval, 'val')],\n                  verbose_eval=False,\n                  early_stopping_rounds=100\n                 )\n    \n    val_X=xgb.DMatrix(val_X)\n    pred_val=[np.argmax(x) for x in xgb_model.predict(val_X)]\n    \n    print('choen_kappa_score :',cohen_kappa_score(pred_val,val_y,weights='quadratic'))\n    \n    pred=xgb_model.predict(final_test)\n    y_pre+=pred\n    \npred = np.asarray([np.argmax(line) for line in y_pre])")


# In[ ]:


sub=pd.DataFrame({'installation_id':submission.installation_id,'accuracy_group':pred})
sub.to_csv('submission.csv',index=False)

