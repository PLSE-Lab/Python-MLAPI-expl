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

import os
print(os.listdir("../input"))
##
print(os.listdir("../input/neuralmoredecisive"))
print(os.listdir("../input/adjustlgbm1003"))
print(os.listdir("../input/mergesvm"))

##
print(os.listdir("../input/ourpathtowherewearenewparams"))
print(os.listdir("../input/adjustpredictiondataframe"))

###
print(os.listdir("../input/closethegapbetweencvandlb"))
print(os.listdir("../input/betternormalizedwithxgb"))

# Any results you write to the current directory are saved as output.


# In[ ]:


##
#df1010 = pd.read_csv("../input/ourpathtowherewearenewparams/subm_0.610175_2018-12-07-23-29.csv")
#df1010=df1010.sort_values('object_id').reset_index(drop=True)
#df992 = pd.read_csv("../input/adjustpredictiondataframe/justSetZeroProbas.csv")
#df992=df992.sort_values('object_id').reset_index(drop=True)

##
#dftestize= pd.read_csv("../input/closethegapbetweencvandlb/subm_0.668470_2018-12-06-19-58.csv")
#dftestize=dftestize.sort_values('object_id').reset_index(drop=True)
#dfxgb= pd.read_csv("../input/betternormalizedwithxgb/subm_0.644318_2018-12-11-18-25.csv")
#dfxgb=dfxgb.sort_values('object_id').reset_index(drop=True)

#print(df1010.shape)
#print(df992.shape)
#print(dftestize.shape)
#print(dfxgb.shape)


# In[ ]:


## Fundamentally different models:
#dfsvm = pd.read_csv("../input/mergesvm/mergeOriginalSvm.csv") #I have never submitted this on its own
#dfsvm=dfsvm.sort_values('object_id').reset_index(drop=True)
dflgbm = pd.read_csv("../input/adjustlgbm1003/justSetZeroProbas.csv") #haven't scored this alone, similar 1.024
dflgbm=dflgbm.sort_values('object_id').reset_index(drop=True)
dfnn=pd.read_csv("../input/neuralmoredecisive/moreDecisiveNeural.csv")
dfnn=dfnn.sort_values('object_id').reset_index(drop=True)

#print(dfsvm.shape)
print(dflgbm.shape)
print(dfnn.shape)


# In[ ]:


dflgbm.head()


# In[ ]:


dfBlend=.5*dflgbm + .5*dfnn #+ .1*(df992 + df1010 + dfxgb)
dfBlend.loc[:,'object_id']=dflgbm.loc[:,'object_id']


# In[ ]:


dfBlend.head()


# In[ ]:


dfBlend.tail()


# In[ ]:


dfBlend.describe()


# In[ ]:


import random
#dfBlend=.35*dflgbm + .25*dfnn + .1*(df992 + df1010 + dftestize + dfxgb)
dfs=[dflgbm, dfnn]
wts=[.5, .5]
#num per class
npc=1000


def validateBlend(bdf, dfs, wts, npc=1000):
    
    recs=bdf.shape[0]
    cols=len(bdf.columns)
    failures=0
    for i in range(npc):
        record=random.randint(0,recs-1)
        feat=bdf.columns[random.randint(0, cols-1)]
        
        
        shouldBe=0
        isVal=bdf.loc[record, feat]
        for j in range(len(wts)):
            
            shouldBe += dfs[j].loc[record, feat] * wts[j]
        shouldBe=round(shouldBe,6)
        isVal=round(isVal,6)
        if shouldBe==isVal:
            #print('match')
            dosomething=False
        else:
            print(record)
            print(feat)
            print('should be: ' + str(shouldBe))
            print('is: ' + str(isVal))
            failures+=1
            
    print('total failures: ' + str(failures) + ' out of ' + str(npc) +' sample points')
    return failures
            
    
failures=validateBlend(dfBlend, dfs, wts)    


# In[ ]:


dfBlend.describe()


# In[ ]:


dfBlend.to_csv('lgbm50DecisiveNeural50.csv', index=False)

