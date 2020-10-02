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


bgg = pd.read_csv('../input/bgg_db_2017_04.csv',encoding='latin1')
bgg.head(5)


# In[ ]:


mech_list = bgg['mechanic'].as_matrix()
cate_list = bgg['category'].as_matrix()


# In[ ]:


all_mech=list(set((', '.join(list(mech_list)).split(','))))
all_cate=list(set((', '.join(list(cate_list)).split(','))))
all_index = all_mech + all_cate


# In[ ]:


def ohe(i):
    game_mech = bgg['mechanic'][i].split(',')
    game_cate = bgg['category'][i].split(',')
    final_vec = np.zeros(53+85)
    for game_m in game_mech:
        if game_m in all_mech:
            final_vec[all_mech.index(game_m)]=1
    for game_c in game_cate:
        if game_c in all_cate:
            final_vec[all_cate.index(game_c)+53]=1
    return final_vec


# In[ ]:


ohe_list = list(map(ohe, range(4999)))
geek_rating = bgg['geek_rating'].as_matrix()
weight = bgg['weight'].as_matrix()
avg_rating = bgg['avg_rating'].as_matrix()
train_feature = np.asarray(list(ohe_list), dtype=int)
index = np.arange(4999)
np.random.shuffle(index)
cv_hold=4900


# In[ ]:


from sklearn import linear_model as lm
#reg = lm.Ridge(alpha=10)
reg = lm.BayesianRidge()


# In[ ]:


reg.fit(train_feature[index[:cv_hold]], weight[index[:cv_hold]])


# In[ ]:


reg.fit(train_feature[index[:cv_hold]], weight[index[:cv_hold]])
print(reg.score(train_feature[index[:cv_hold]], weight[index[:cv_hold]]))
print(reg.score(train_feature[index[cv_hold:]], weight[index[cv_hold:]]))


# In[ ]:


weight_imp = reg.coef_.reshape(-1,1)


# In[ ]:


reg.fit(train_feature[index[:cv_hold]], avg_rating[index[:cv_hold]])
print(reg.score(train_feature[index[:cv_hold]], avg_rating[index[:cv_hold]]))
print(reg.score(train_feature[index[cv_hold:]], avg_rating[index[cv_hold:]]))


# In[ ]:


weight_imp = np.concatenate((weight_imp,reg.coef_.reshape(-1,1)),axis=1)


# In[ ]:


reg.fit(train_feature[index[:cv_hold]], geek_rating[index[:cv_hold]])
print(reg.score(train_feature[index[:cv_hold]], geek_rating[index[:cv_hold]]))
print(reg.score(train_feature[index[cv_hold:]], geek_rating[index[cv_hold:]]))


# In[ ]:


weight_imp = np.concatenate((weight_imp,reg.coef_.reshape(-1,1)),axis=1)


# In[ ]:


df = pd.DataFrame(weight_imp,index=all_mech+all_cate,columns=['weight','avg_rating','geek_rating'])


# In[ ]:


num_owned = np.argsort(bgg['owned'].as_matrix())
name_list = []
for genre in all_index:
    for num in num_owned[::-1]:
        if genre in mech_list[num] or genre in cate_list[num]:
            name_list.append(bgg['names'][num])
            break
        if num==num_owned[0]:
            name_list.append('Not found')


# In[ ]:


df['representive']=pd.Series(name_list,index=df.index)


# # Sort by difficulty

# In[ ]:


df.sort_values('weight').tail(20)


# #Sort by average rating

# In[ ]:


df.sort_values('avg_rating').tail(20)


# # Sort by geek rating

# In[ ]:


df.sort_values('geek_rating').tail(20)


# #Normal likes but geek doesn't

# In[ ]:


df['geek_avg_dif'] = df['geek_rating']-df['avg_rating']
df[['geek_avg_dif','representive']].sort_values('geek_avg_dif').head(20)


# #Geek likes but normal doesn't

# In[ ]:


df[['geek_avg_dif','representive']].sort_values('geek_avg_dif').tail(20)


# In[ ]:


import scipy as sp
print(sp.stats.pearsonr(geek_rating, weight))
print(sp.stats.pearsonr(avg_rating, weight))


# In[ ]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler as SS
scaler = SS()


# In[ ]:


norm_geek =scaler.fit_transform(geek_rating.reshape(-1,1))
norm_avg = scaler.fit_transform(avg_rating.reshape(-1,1))
norm_weight = scaler.fit_transform(weight.reshape(-1,1))


# In[ ]:


plt.scatter(norm_weight, norm_avg)


# In[ ]:


plt.scatter(norm_weight, norm_geek)


# In[ ]:




