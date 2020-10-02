#!/usr/bin/env python
# coding: utf-8

# In[56]:


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


# In[57]:


#INPUT CSV
WNCAATourneySeeds=pd.read_csv('../input/WNCAATourneySeeds.csv')
WNCAATourneyCompactResults=pd.read_csv('../input/WNCAATourneyCompactResults.csv')
WRegularSeasonCompactResults=pd.read_csv('../input/WRegularSeasonCompactResults.csv')


# In[58]:


#PRE-FEATURE


# In[59]:


#byPass_SEED
#_generate feature
WNCAATourneySeeds_1=WNCAATourneySeeds
WNCAATourneySeeds_1['tier']=WNCAATourneySeeds.Seed.apply(lambda x: x[1:3])
WNCAATourneySeeds_1['region']=WNCAATourneySeeds.Seed.apply(lambda x: x[:1])
WNCAATourneySeeds_2=WNCAATourneySeeds_1.loc[:,['Season', 'TeamID', 'tier', 'region']]
#_prepare_join
WNCAATourneySeeds_2_W=WNCAATourneySeeds_2.copy()
WNCAATourneySeeds_2_W.columns=['Season', 'WTeamID', 'Wtier', 'Wregion']
WNCAATourneySeeds_2_L=WNCAATourneySeeds_2.copy()
WNCAATourneySeeds_2_L.columns=['Season', 'LTeamID', 'Ltier', 'Lregion']


# In[60]:


#mainSteam
mainSteam_1=WNCAATourneyCompactResults.merge(WNCAATourneySeeds_2_W,how='left',on=['Season','WTeamID'])
mainSteam_2=mainSteam_1.merge(WNCAATourneySeeds_2_L,how='left',on=['Season','LTeamID'])


# In[61]:


#mainSteam feature I
mainSteam_2.Wtier=mainSteam_2.Wtier.astype(int)
mainSteam_2.Ltier=mainSteam_2.Ltier.astype(int)
mainSteam_2['tierDiff']=(mainSteam_2['Wtier']-mainSteam_2['Ltier'])
mainSteam_3=mainSteam_2.loc[:,['Season',  'WTeamID', 'LTeamID', 'Wtier', 'Wregion', 'Ltier', 'Lregion', 'tierDiff']]
#_prepare merge
mainSteam_3_W=mainSteam_3.copy()
mainSteam_3_L=mainSteam_3.copy()
mainSteam_3_W.columns=['Season',  'TeamID_1', 'TeamID_2', 'tier_1', 'region_1', 'tier_2', 'region_2', 'tierDiff']
mainSteam_3_L.columns=['Season',  'TeamID_2', 'TeamID_1', 'tier_2', 'region_2', 'tier_1', 'region_1', 'tierDiff']
mainSteam_3_L.tierDiff=-mainSteam_3_L.tierDiff
#_add_label
mainSteam_3_W['label']=1
mainSteam_3_L['label']=0
#_append
mainSteam_4=mainSteam_3_W.append(mainSteam_3_L,ignore_index=True)


# In[62]:


#_bypass_lastTourCounts
WNCAATourneyCompactResults_shift=WNCAATourneyCompactResults.copy()
WNCAATourneyCompactResults_shift.Season=WNCAATourneyCompactResults_shift.Season+1
WNCAATourneyCompactResults_shift=WNCAATourneyCompactResults_shift.loc[:,['Season','WTeamID','NumOT']]
#_aggregate
lastTourWinCount=WNCAATourneyCompactResults_shift.groupby(['Season','WTeamID']).count()
#_convert to normal table
lastTourWinCount.to_csv('lastTourWinCount.csv')
lastTourWinCount_1=pd.read_csv('lastTourWinCount.csv')
lastTourWinCount_1.columns=['Season', 'WTeamID', 'counts']
lastTourWinCount_1.tail(1)
#_prepare merge
lastTourWinCount_1_W=lastTourWinCount_1.copy()
lastTourWinCount_1_L=lastTourWinCount_1.copy()
lastTourWinCount_1_W.columns=['Season', 'TeamID_1', 'lastTourWinCount1']
lastTourWinCount_1_L.columns=['Season', 'TeamID_2', 'lastTourWinCount2']
lastTourWinCount_1_W.tail()


# In[63]:


#mainSteam feature II
mainSteam_5=mainSteam_4.merge(lastTourWinCount_1_W,how='left',on=['Season','TeamID_1'])
mainSteam_6=mainSteam_5.merge(lastTourWinCount_1_L,how='left',on=['Season','TeamID_2'])


# In[64]:


#_bypass_SeasonCount
WRegularSeasonCompactResults_1=WRegularSeasonCompactResults
WRegularSeasonCompactResults_1=WRegularSeasonCompactResults_1.loc[:,['Season','WTeamID','NumOT']]
#_aggregate
seasonCount=WRegularSeasonCompactResults_1.groupby(['Season','WTeamID']).count()
#_convert to normal talbe
seasonCount.to_csv('seasonCount.csv')
seasonCount_1=pd.read_csv('seasonCount.csv')
seasonCount_1.columns=['Season', 'WTeamID', 'counts']
#_prepare merge
seasonCount_1_a=seasonCount_1.copy()
seasonCount_1_b=seasonCount_1.copy()
seasonCount_1_a.columns=['Season', 'TeamID_1', 'seasonCount1']
seasonCount_1_b.columns=['Season', 'TeamID_2', 'seasonCount2']


# In[65]:


#mainSteam feature III
mainSteam_7=mainSteam_6.merge(seasonCount_1_a,how='left',on=['Season','TeamID_1'])
mainSteam_8=mainSteam_7.merge(seasonCount_1_b,how='left',on=['Season','TeamID_2'])
mainSteam_8.sort_values(['Season','TeamID_1','TeamID_2']).tail()


# In[66]:


#mainSteam feature optmize
#_fill na
mainSteam_9=mainSteam_8.sort_values(['Season','TeamID_1','TeamID_2'])
mainSteam_9['lastTourWinCount1']=mainSteam_9['lastTourWinCount1'].fillna(0)
mainSteam_9['lastTourWinCount2']=mainSteam_9['lastTourWinCount2'].fillna(0)
#_convert to int
mainSteam_9.lastTourWinCount1=mainSteam_9.lastTourWinCount1.astype(int)
mainSteam_9.lastTourWinCount2=mainSteam_9.lastTourWinCount2.astype(int)
mainSteam_9.seasonCount1=mainSteam_9.seasonCount1.astype(int)
mainSteam_9.seasonCount2=mainSteam_9.seasonCount2.astype(int)


# In[67]:


#_numerical to bionominal
def N2B(table,columns):
    tempdummies=pd.get_dummies(table[columns],prefix=columns)
    merge=pd.concat([table,tempdummies],axis=1)
    return merge

mainTemp_1=N2B(mainSteam_9,'lastTourWinCount1')
mainTemp_2=N2B(mainTemp_1,'lastTourWinCount2')
mainTemp_3=N2B(mainTemp_2,'seasonCount1')
mainTemp_4=N2B(mainTemp_3,'seasonCount2')
mainTemp_5=N2B(mainTemp_4,'tier_1')
mainTemp_6=N2B(mainTemp_5,'tier_2')
mainTemp_7=N2B(mainTemp_6,'region_1')
mainTemp_8=N2B(mainTemp_7,'region_2')
mainSteamEnd=mainTemp_8


# In[68]:


#train set build
train=mainSteamEnd[mainSteamEnd['Season']>2014]


# In[69]:


#test set build
WSampleSubmissionStage1=pd.read_csv('../input/WSampleSubmissionStage2.csv')
#_pre feature
test=WSampleSubmissionStage1.copy()
test['Season']=test.ID.apply(lambda x: x[:4])
test['TeamID_1']=test.ID.apply(lambda x: x[5:9])
test['TeamID_2']=test.ID.apply(lambda x: x[10:14])
#_pre merge
WNCAATourneySeeds_2_1=WNCAATourneySeeds_2.copy()
WNCAATourneySeeds_2_2=WNCAATourneySeeds_2.copy()
WNCAATourneySeeds_2_1.columns=['Season', 'TeamID_1', 'tier_1', 'region_1']
WNCAATourneySeeds_2_2.columns=['Season', 'TeamID_2', 'tier_2', 'region_2']
test.Season=test.Season.astype(int)
test.TeamID_1=test.TeamID_1.astype(int)
test.TeamID_2=test.TeamID_2.astype(int)
#_merge
test1=test.merge(WNCAATourneySeeds_2_1,how='left',on=['Season','TeamID_1'])
test2=test1.merge(WNCAATourneySeeds_2_2,how='left',on=['Season','TeamID_2'])
#feature I
test2.tier_1=test2.tier_1.astype(int)
test2.tier_2=test2.tier_2.astype(int)
test2['tierDiff']=(test2['tier_1']-test2['tier_2'])
#feature II
test3=test2.merge(lastTourWinCount_1_W,how='left',on=['Season','TeamID_1'])
test4=test3.merge(lastTourWinCount_1_L,how='left',on=['Season','TeamID_2'])
#feature III
test5=test4.merge(seasonCount_1_a,how='left',on=['Season','TeamID_1'])
test6=test5.merge(seasonCount_1_b,how='left',on=['Season','TeamID_2'])
#feature opt
test6['lastTourWinCount1']=test6['lastTourWinCount1'].fillna(0)
test6['lastTourWinCount2']=test6['lastTourWinCount2'].fillna(0)
test6.lastTourWinCount1=test6.lastTourWinCount1.astype(int)
test6.lastTourWinCount2=test6.lastTourWinCount2.astype(int)
test6.seasonCount1=test6.seasonCount1.astype(int)
test6.seasonCount2=test6.seasonCount2.astype(int)
#_numerical to bionominal
testTemp_1=N2B(test6,'lastTourWinCount1')
testTemp_2=N2B(testTemp_1,'lastTourWinCount2')
testTemp_3=N2B(testTemp_2,'seasonCount1')
testTemp_4=N2B(testTemp_3,'seasonCount2')
testTemp_5=N2B(testTemp_4,'tier_1')
testTemp_6=N2B(testTemp_5,'tier_2')
testTemp_7=N2B(testTemp_6,'region_1')
testTemp_8=N2B(testTemp_7,'region_2')
testEnd=testTemp_8


# In[70]:


#fix test set select attribute
temp=['tier_1',
 'tier_2',
 'tierDiff',
 'lastTourWinCount1',
 'lastTourWinCount2',
 'seasonCount1',
 'seasonCount2',
 'lastTourWinCount1_0',
 'lastTourWinCount1_1',
 'lastTourWinCount1_2',
 'lastTourWinCount1_3',
 'lastTourWinCount1_4',
 'lastTourWinCount1_5',
 'lastTourWinCount1_6',
 'lastTourWinCount2_0',
 'lastTourWinCount2_1',
 'lastTourWinCount2_2',
 'lastTourWinCount2_3',
 'lastTourWinCount2_4',
 'lastTourWinCount2_5',
 'lastTourWinCount2_6',
 'seasonCount1_20',
 'seasonCount1_21',
 'seasonCount1_22',
 'seasonCount1_23',
 'seasonCount1_24',
 'seasonCount1_25',
 'seasonCount1_26',
 'seasonCount1_27',
 'seasonCount1_28',
 'seasonCount1_29',
 'seasonCount1_30',
 'seasonCount1_31',
 'seasonCount1_32',
 'seasonCount1_33',
 'seasonCount1_34',

 'seasonCount2_20',
 'seasonCount2_21',
 'seasonCount2_22',
 'seasonCount2_23',
 'seasonCount2_24',
 'seasonCount2_25',
 'seasonCount2_26',
 'seasonCount2_27',
 'seasonCount2_28',
 'seasonCount2_29',
 'seasonCount2_30',
 'seasonCount2_31',
 'seasonCount2_32',
 'seasonCount2_33',
 'seasonCount2_34',
 'tier_1_1',
 'tier_1_2',
 'tier_1_3',
 'tier_1_4',
 'tier_1_5',
 'tier_1_6',
 'tier_1_7',
 'tier_1_8',
 'tier_1_9',
 'tier_1_10',
 'tier_1_11',
 'tier_1_12',
 'tier_1_13',
 'tier_1_14',
 'tier_1_15',
 'tier_1_16',
 'tier_2_1',
 'tier_2_2',
 'tier_2_3',
 'tier_2_4',
 'tier_2_5',
 'tier_2_6',
 'tier_2_7',
 'tier_2_8',
 'tier_2_9',
 'tier_2_10',
 'tier_2_11',
 'tier_2_12',
 'tier_2_13',
 'tier_2_14',
 'tier_2_15',
 'tier_2_16',
 'region_1_W',
 'region_1_X',
 'region_1_Y',
 'region_1_Z',
 'region_2_W',
 'region_2_X',
 'region_2_Y',
 'region_2_Z']
columns=temp

#_testfix
testFixed=testEnd


# In[71]:


#output for test
savetocheck=train[columns]
savetocheck['label']=train['label']
savetocheck.to_csv('savetocheck.csv')
testFixed.to_csv('testFixed.csv')


# In[72]:


from sklearn import svm
clfsvc=svm.SVC( gamma=0.001,probability=True)

clfsvc.fit(train[columns],train['label'])


# In[73]:


clfsvc.score(train[columns],train['label'])


# In[74]:


#testSub
testSub=testFixed

testSub['seasonCount1_28']=0
testSub['seasonCount1_33']=0
testSub['seasonCount1_34']=0

testSub['seasonCount2_28']=0
testSub['seasonCount2_33']=0
testSub['seasonCount2_34']=0


# In[75]:


#prob
prob=clfsvc.predict_proba(testSub[columns])
prob


# In[76]:


upload=testFixed.loc[:,['ID','Pred']]
upload.Pred=prob[:,1]
upload.to_csv("Submission_selectInput.csv",index=False)
upload.head()


# In[ ]:




