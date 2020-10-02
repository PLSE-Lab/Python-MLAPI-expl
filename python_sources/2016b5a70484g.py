#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.columns = ['id', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'type', 'J', 'K', 'rating']
# df['type'] = df['type'].map({'new':1, 'old':0})
# df.fillna(value = df.mean(), inplace = True)
# df = pd.get_dummies(data = df, columns = ['type'])
df.head()


# In[ ]:


new_mean = df[df.type == 'new'].C.mean()
old_mean = df[df.type == 'old'].C.mean()
for i in range(len(df)):
    if df.loc[:,'C'].isnull().loc[i]:
        if df.loc[i, 'type'] == 'new':
            df.loc[i, 'C'] = new_mean
        else:
            df.loc[i, 'C'] = old_mean

new_mean = df[df.type == 'new'].D.mean()
old_mean = df[df.type == 'old'].D.mean()
for i in range(len(df)):
    if df.loc[:,'D'].isnull().loc[i]:
        if df.loc[i, 'type'] == 'new':
            df.loc[i, 'D'] = new_mean
        else:
            df.loc[i, 'D'] = old_mean
            
new_mean = df[df.type == 'new'].J.mean()
old_mean = df[df.type == 'old'].J.mean()
for i in range(len(df)):
    if df.loc[:,'J'].isnull().loc[i]:
        if df.loc[i, 'type'] == 'new':
            df.loc[i, 'J'] = new_mean
        else:
            df.loc[i, 'J'] = old_mean


# In[ ]:


df = df.fillna(df.mean())
df = pd.get_dummies(data = df, columns = ['type'])
df.head()


# In[ ]:


df = df.drop( df.index[df.K > 1.75], axis = 0 )
df = df.drop( df.index[df.G > 10.07], axis = 0 )
df = df.drop( df.index[df.I > 27], axis = 0 )
df = df.drop( df.index[ (df.E > 10) & (df.rating == 0) ] )
# df = df.drop( df.index[ (df.E > 30) & (df.rating == 4) ] )
# df = df.drop( df.index[ (df.E > 30) & (df.rating == 3) ] )
df = df.drop( df.index[ (df.E > 30) ] )
df = df.drop( df.index[ (df.C > 12) ] )
df = df.drop( df.index[ (df.H > 1) ] )
df = df.drop( df.index[ (df.A > 30) ] )
len(df)
sns.scatterplot(x = 'A', y = 'B', hue = 'rating', data = df)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler


# In[ ]:


def test_RFC(df, test):
    df = df.drop('id', axis = 1)
#     test = test.drop('id', axis = 1)
    
    X = df.drop('rating', axis = 1)
    y = df['rating']
    
#     scaler = RobustScaler()
#     scaler.fit(X)
#     X = scaler.transform(X)
#     test = scaler.transform(test)
    
    clas = ExtraTreesRegressor(n_estimators = 4000, random_state = 4)
    cl = clas.fit(X, y)
    y_pred = cl.predict(test)
    
    return y_pred


# In[ ]:


test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
test.columns =  ['id', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'type', 'J', 'K']
# test['type'] = test['type'].map({'new':1, 'old':0}) 
test = pd.get_dummies(data = test, columns = ['type'] )
# df2.columns


# In[ ]:


test = test.fillna(value = test.mean())


# In[ ]:


test.head()


# In[ ]:


df2 = df 
df2 = df2.drop('rating', axis = 1)
df2 = pd.concat([df2, df.rating], axis = 1)


# In[ ]:


Xtest = test 
Xtest = Xtest.drop('id', axis = 1)
# df2 = df2.drop(['A', 'C', 'G'], axis = 1)
# Xtest = Xtest.drop(['A', 'C', 'G'], axis = 1)
Xtest.columns


# In[ ]:


# test_RFC(df2, Xtest)
# Xtest.columns
# df2 = df2.drop(['B', 'G', 'type_new', 'type_old'], axis = 1)
# Xtest = Xtest.drop(['B','G', 'type_new', 'type_old'], axis = 1)


# In[ ]:


test_ans = pd.DataFrame(test_RFC(df2, Xtest))


# In[ ]:


test_ans.head()


# In[ ]:


test_ans = pd.concat([test['id'], test_ans], axis = 1)


# In[ ]:


test_ans.head()


# In[ ]:


test_ans.columns = ['id', 'rating']


# In[ ]:


test_ans['rating'] = test_ans['rating'].round()


# In[ ]:


# test_ans['rating'] = test_ans['rating'].astype('int64')


# In[ ]:


test_ans.columns = ['id', 'rating']
test_ans.head()


# In[ ]:


test_ans.to_csv('submission20.csv', index = False)


# In[ ]:


len(test)


# In[ ]:





# In[ ]:




