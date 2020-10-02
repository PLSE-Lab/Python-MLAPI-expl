#!/usr/bin/env python
# coding: utf-8

# In[65]:


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


# In[66]:


data_dir = '../input/'
Seeds = pd.read_csv(data_dir +'WNCAATourneySeeds.csv')
CompactResults= pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')
#CompactResults= pd.read_csv(data_dir + 'WRegularSeasonCompactResults.csv')


# In[67]:


Seeds.head()


# In[68]:


CompactResults.head()


# FEATURE I

# In[69]:


def seed_int(x):
    temp = int(x[1:3])
    return temp
Seeds['tier'] = Seeds.Seed.apply(seed_int)

def seed_region(x):
    temp=x[:1]
    return temp
Seeds['region']=Seeds.Seed.apply(seed_region)
Seeds.head()


# In[70]:


tempSeeds=Seeds.loc[:,['Season','TeamID','tier','region']]
tempSeeds.columns=['Season','WTeamID','WTier','WRegion']
temp=CompactResults.merge(tempSeeds,how='left',on=['Season','WTeamID'])
temp.head()


# In[71]:


tempSeeds.columns=['Season','LTeamID','LTier','LRegion']
merge=temp.merge(tempSeeds,how='left',on=['Season','LTeamID'])
merge.head()


# In[72]:


#merge['history']=(merge['Season']-1990)
merge['tierDiff']=(merge['WTier']-merge['LTier'])
merge=merge.loc[:,['WLoc','WRegion','LRegion','tierDiff']]
merge.head()


# In[73]:


merge['VSRegion']=merge['WRegion']+merge['LRegion']
merge.head()


# In[74]:


merge=merge.loc[:,['WLoc','tierDiff','VSRegion']]
merge.head()


# In[75]:


mergeReverse=merge

def reverse(x):
    temp1 = x[:1]
    temp2 = x[1:2]
    temp=temp2+temp1
    return temp

mergeReverse['VSRegion2']=merge['VSRegion'].apply(reverse)
mergeReverse['tierDiff']=-mergeReverse['tierDiff']

mergeReverse.head()


# In[76]:


def reverseLoc(x):
    if x == 'H':
       return 'A' 
    elif x == 'A':
        return 'H'
    else:
        return x

mergeReverse['WLoc2']=merge['WLoc'].apply(reverseLoc)

mergeReverse.head()


# In[77]:


mergeReverse=mergeReverse.loc[:,['WLoc2','tierDiff','VSRegion2']]
mergeReverse.head()


# In[78]:


mergeReverse.columns=['WLoc','tierDiff','VSRegion']
mergeReverse['feature']=1
mergeReverse.head()


# In[79]:


merge=merge.loc[:,['WLoc','tierDiff','VSRegion']]
merge['feature']=0
merge.head()


# In[80]:


#test append
temp1=merge.head()
temp2=mergeReverse.head()
temp1.append(temp2,ignore_index=True)


# In[81]:


merge=merge.append(mergeReverse,ignore_index=True)
merge.head()


# In[82]:


tempdummies = pd.get_dummies(merge['WLoc'],prefix='WLoc')
tempdummies.head()


# In[83]:


merge=pd.concat([merge,tempdummies],axis=1)
merge.head()


# In[84]:


merge=merge.drop(labels=['WLoc'],axis=1)
merge.head()


# In[85]:


tempdummies = pd.get_dummies(merge['VSRegion'],prefix='VSRegion')
tempdummies.head()


# In[86]:


merge=pd.concat([merge,tempdummies],axis=1)
merge.head()


# In[87]:


merge=merge.drop(labels=['VSRegion'],axis=1)
merge.head()


# MODEL I

# In[88]:


#from sklearn.utils import shuffle
#merge=shuffle(merge)
#merge.head()
#scaler???


# In[89]:


feature=merge.feature


# In[90]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[91]:


#where to find data about WLoc in submission
merge=merge.drop(labels=['WLoc_A','WLoc_H','WLoc_N'],axis=1)
merge.head()


# In[92]:


train=merge.drop(labels=['feature'],axis=1)
train.head()


# In[93]:


hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5,7,10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [3,5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}

clf = RandomForestClassifier(random_state=3)
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)

all_X = train.values
all_y = feature

grid.fit(all_X, all_y)


# In[94]:


best_params = grid.best_params_
best_score = grid.best_score_
best_score


# In[95]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,all_X, all_y,cv=10)
scores.mean()


# PREDICT I

# In[96]:


Submission = pd.read_csv(data_dir + 'WSampleSubmissionStage1.csv')
output=Submission
output.head()


# In[97]:


def year(x):
    temp = (x[:4])
    return temp
output['year'] = output.ID.apply(year)
output.head()


# In[98]:


def team(x):
    temp = (x[5:9])
    return temp
output['team1'] = output.ID.apply(team)
output.head()


# In[99]:


def team(x):
    temp = (x[10:14])
    return temp
output['team2'] = output.ID.apply(team)
output.head()


# In[100]:


Seeds.head()


# In[101]:


tempSeeds=Seeds.loc[:,['Season','TeamID','tier','region']]
tempSeeds.head()


# In[102]:


tempSeeds.columns=['year','team1','tier1','region1']
tempSeeds.head()


# In[103]:


output.year=output.year.astype(int)
output.team1=output.team1.astype(int)
output.team2=output.team2.astype(int)


# In[104]:


output.dtypes


# In[105]:


tempSeeds.dtypes


# In[106]:


temp=pd.merge(output,tempSeeds,how='left',on=['year','team1'])
temp.head()


# In[107]:


tempSeeds.columns=['year','team2','tier2','region2']
tempSeeds.head()


# In[108]:


output=pd.merge(temp,tempSeeds,how='left',on=['year','team2'])
output.head()


# In[109]:


#output['history']=(output['year']-1990)
#output.head()


# In[110]:


output['tierDiff']=(output['tier1']-output['tier2'])
output.head()


# In[111]:


output['VSRegion']=output['region1']+output['region2']
output.head()


# In[112]:


train.columns


# In[113]:


output=output.loc[:,['ID', 'tierDiff','region1','region2','VSRegion']]
output.head()


# In[114]:


tempdummies = pd.get_dummies(output['VSRegion'],prefix='VSRegion')
tempdummies.head()


# In[115]:


output=pd.concat([output,tempdummies],axis=1)
output.head()


# In[116]:


output.columns


# In[117]:


merge.columns


# In[118]:


columns=[ 'tierDiff',
       'VSRegion_WW', 'VSRegion_WX', 'VSRegion_WY', 'VSRegion_WZ',
       'VSRegion_XW', 'VSRegion_XX', 'VSRegion_XY', 'VSRegion_XZ',
       'VSRegion_YW', 'VSRegion_YX', 'VSRegion_YY', 'VSRegion_YZ',
       'VSRegion_ZW', 'VSRegion_ZX', 'VSRegion_ZY', 'VSRegion_ZZ']


# In[119]:


best_rf = grid.best_estimator_
outcome= best_rf.predict(output[columns])
proba= best_rf.predict_proba(output[columns])


# In[120]:


proba


# In[121]:


Submission = pd.read_csv(data_dir + 'WSampleSubmissionStage1.csv')
Submission.head()


# In[122]:


Submission.Pred=proba[:,1]

Submission.head()


# In[127]:


Submission.to_csv('prediction1.csv', index=False)

