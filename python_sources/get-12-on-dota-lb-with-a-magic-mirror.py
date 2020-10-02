#!/usr/bin/env python
# coding: utf-8

# # Get +12 on LB on Dota2 with a magic mirror

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os


# **The main idea is that we have not one but two train data sets to fit our models!**

# ## An example:

# You have some data like this:

# In[ ]:


train_data = pd.DataFrame(data=np.array([[100,0,0,200,1],[0,300,400,500,0]]), columns=['f1','f2', 'f3', 'f4', 'target'])
train_data


# Please predict target for this test data:

# In[ ]:


test_data = pd.DataFrame(data=np.array([[0,200,100,0,'?'],[400,500,0,300,'?']]), columns=['f1','f2', 'f3', 'f4', 'target'])
test_data


# Do you know target? Of course you don't! We didn't see combination like these in the train data!
# 
# Now. I say that 'f1' and 'f2' are 'gold' and 'xp' of the team0,  'f3' and 'f4' are the same features but for the team1. And target shows that the team0 is winner.
# 
# So! In the first match we see that the team0 with gold=0 and xp=100 win the team1 with gold=200 and xp=0.
# 
# But teams are equal! So it means that any team with 0,100 ussually win any team with 200,0.
# 
# Obviosly, target of **[0,200 against 100,0]** is 0 (fault), because target of it's mirror in train data **[100,0 against 0,200]** is 1 (win).
# 
# And now we have very confident answers for test data!

# But our model does not have this information.
# 
# So, the idea is to make mirror train data for our model. Then our train will be like this:

# In[ ]:


train_data_mirror = pd.DataFrame(data=train_data[['f3','f4', 'f1', 'f2', 'target']].values, columns=['f1','f2', 'f3', 'f4', 'target'])
train_data_mirror.target = 1- train_data.target
train_data_double = pd.concat([train_data,train_data_mirror], axis=0, sort=False)
train_data_double


# Now our model can predict our test target easy!

# ## So, let's do it for our base features in Dota

# In[ ]:


PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'
df_train= pd.read_csv(PATH_TO_DATA + 'train_features.csv', index_col='match_id_hash')
df_train['target']= pd.read_csv(PATH_TO_DATA + 'train_targets.csv', index_col='match_id_hash').radiant_win.astype('int')
df_train.head(5)


# Be carefully! We can't mirror x and y features. Because our teams are in different corners. Let's prepare new x and y for d-team.

# In[ ]:


d_xy_cols = ['d'+str(i)+'_x' for i in range(1,6)] + ['d'+str(i)+'_y' for i in range(1,6)]
df_train[d_xy_cols] = 256 - df_train[d_xy_cols]
# Now they are equal.


# Let's make team features/

# In[ ]:


r_pref = ['r'+str(i)+'_' for i in range(1,6)]
d_pref = ['d'+str(i)+'_' for i in range(1,6)]
team_fts = [c[3:] for c in df_train.columns if c[:3]=='r1_']
for ft in team_fts:
    r_fts = [pref + ft for pref in r_pref]
    d_fts = [pref + ft for pref in d_pref]
    df_train['r_' + ft+'_mean'] = df_train[r_fts].mean(1)
    df_train['d_' + ft+'_mean'] = df_train[d_fts].mean(1)
    df_train.drop(r_fts+d_fts, axis=1, inplace=True)


# In[ ]:


df_train.columns


# In[ ]:


# We have several columns for our magic mirror...
# Let's mark them.
r_cols = [c for c in df_train.columns if c[:2]=='r_']
d_cols = [c for c in df_train.columns if c[:2]=='d_']


# In[ ]:


change_col_dict  = dict(zip(r_cols + d_cols, d_cols+r_cols))


# In[ ]:


df_train['is_mirror']=0 
df_train_mirror = df_train.copy()
df_train_mirror.columns = df_train.columns.map(lambda x: change_col_dict.get(x, x)) # We are changing our columns
df_train_mirror['target'] = 1- df_train['target'] # We are flipping our targets
df_train_mirror['is_mirror'] = 1 # It could be useful to let our model know, that it is synthetic data
df_train_mirror.index = df_train.index.map(lambda x: x+'_wv') # We are making new index for mirror rows


# In[ ]:


df_train_double = pd.concat([df_train, df_train_mirror], axis=0, sort=False)
df_train.shape, df_train_double.shape # So we have new train dataset. It has double count of different instances.


# ## So, we have mirror. But where magic is? Let's see a score!

# ### Plain dataset

# In[ ]:


df_train.shape


# In[ ]:


n=len(df_train)*2 // 3
X_train, y_train, X_valid, y_valid = df_train.drop('target', axis=1)[:n].values, df_train.target[:n].values, df_train.drop('target', axis=1)[n:].values, df_train.target[n:].values
model = LogisticRegression(solver='lbfgs', random_state=1, max_iter=5000)
model.fit(X_train, y_train)
score_train = roc_auc_score( y_valid, model.predict( X_valid))
print('Plain dataset score', score_train)


# ### Double dataset

# In[ ]:


df_train_double.shape


# In[ ]:


X_train, y_train, X_valid, y_valid = pd.concat([df_train.drop('target', axis=1)[:n],df_train_mirror.drop('target', axis=1)[:n]], axis=0, sort=False).values,                                      pd.concat([df_train.target[:n],df_train_mirror.target[:n]], axis=0, sort=False).values,                                      pd.concat([df_train.drop('target', axis=1)[n:],df_train_mirror.drop('target', axis=1)[n:]], axis=0, sort=False).values,                                      pd.concat([df_train.target[n:],df_train_mirror.target[n:]], axis=0, sort=False).values
model = LogisticRegression(solver='lbfgs', random_state=1, max_iter=5000)
model.fit(X_train, y_train)
score_train_double = roc_auc_score( y_valid, model.predict( X_valid))
print('Double dataset score', score_train_double)


# In[ ]:


print('Plain dataset score', score_train)
print('Double dataset score', score_train_double)
print('Growth +{0:10.2f}%%'.format((score_train_double/score_train-1)*100))


# We have 0.48% better result! Is it big improvement? Let's see on leaderbord! The 1st place has 0.86428, minus 0.48%...is...  

# In[ ]:


0.86428/(score_train_double/score_train)


# .. is 0.0.86011... It is the 13d place!

# **Waw!!!! Our magic mirror can get us +12 improvement on the leaderboard in Dota2!!! It's really magic!!! **

# Do not give thanks ... Although..
