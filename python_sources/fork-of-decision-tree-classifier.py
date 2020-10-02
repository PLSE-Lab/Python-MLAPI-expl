#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from glob import glob 

players_file = glob("../input/*csv")
players_name = [i.strip('./train\\').replace('infor.csv','') for i in players_file]
players_list = []
for step,i in enumerate(players_file): 
    if i != '../input/test_data.csv' : 
        players_list.append(pd.read_csv(i).drop(['matchid','kills','deaths','assists','accountid','name','radiwin','roam'],axis = 1 ))
features = players_list[0].columns.drop('win')
all_data = pd.concat([i for i in players_list])

print (features)

def spearman(frame, features,players_name): 
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['win'],'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(8,0.25*len(features)))
    sns.barplot(data=spr,y='feature', x='corr', orient = 'h')
    plt.title(players_name)

spearman(all_data, features,'All')


# In[16]:


win_color='r' 
lose_color='b'

win_data = all_data[all_data['win']==1]
lose_data = all_data[all_data['win']==0]
win_kda = win_data["kda"].value_counts().sort_index()
lose_kda = -1*lose_data["kda"].value_counts().sort_index()

lose_kda.plot.bar(label='lose',color='b')
win_kda.plot.bar(label='win',color='r')
plt.legend()

def feature_hist(win, lose, feature):   
    bins = 20
    plt.figure()
    plt.hist(win[feature], bins, alpha=0.5, label='win_'+ feature)
    plt.hist(lose[feature], bins, alpha=0.5, label='lose_'+feature)
    plt.legend(loc='upper right')
    plt.show()

feature_hist(win_data,lose_data,"kda")
feature_hist(win_data,lose_data,"gpm")
feature_hist(win_data,lose_data,"xpm")


# In[22]:


train_X = all_data[['xpm','gpm','towerdamage','kda','kpm']]
train_Y = all_data['win']

from sklearn import tree

clf = tree.DecisionTreeClassifier()
test = clf.fit(train_X,train_Y)

test_data = pd.read_csv('../input/test_data.csv')
test_X = test_data[['xpm','gpm','towerdamage','kda','kpm']]
test_Y = test_data['win']

output = test.predict(test_X)
accuracy = (test_Y==output).value_counts()[1]/( (test_Y==output).value_counts()[1] + (test_Y==output).value_counts()[0] )
#print ('accuracy of test sample: %f' %accuracy)
print ( accuracy)


# In[ ]:




