#!/usr/bin/env python
# coding: utf-8

# Read raw data 

# In[1]:


import numpy as np 
import math
import pandas as pd 
import matplotlib,pylab as plt
from tqdm import tqdm
rdir='../input/'
wish = pd.read_csv(rdir+'santa-gift-matching/child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
gift = pd.read_csv(rdir+'santa-gift-matching/gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
n_children = wish.shape[0] # n children/gift to give
n_gift_type = gift.shape[0] # n types of gifts available


# Read two submissions. One from my base scoring, one from ZFTurbo's infinite probablistic improver (Great work! Learned a lot from his code! many thanks!).

# In[2]:


BaseScore = rdir+'how-to-get-a-score-of-0-894-in-two-minutes/ZetaSantaGift_Score_None.csv'
BetterScore = rdir+'infinite-probabilistic-improver-0-931/subm_0.934960657680.csv'
pred_base  = pd.read_csv(BaseScore);
pred_better  = pd.read_csv(BetterScore);


# Helper functions

# In[3]:


chigif = np.full((n_children, n_gift_type), -1,dtype=np.int16)
VAL = (np.arange(200,0,-2)+1)
for c in tqdm(range(n_children)):
    chigif[c, wish[c]] += VAL    
gifchi = np.full((n_gift_type, n_children), -1,dtype=np.int16)
VAL = (np.arange(2000,0,-2)+1)
for g in tqdm(range(n_gift_type)):
    gifchi[g, gift[g]] += VAL    
def getGCvalMatrx(pred):
    pred['gval']=pd.Series(chigif[range(n_children),pred['GiftId']])
    pred['cval']=pd.Series(gifchi[pred['GiftId'],range(n_children)])
    return pred


# Scores for the two submissions:

# In[4]:


def avgNH(pred): ## faster scoring function
    """ Adapted from TeraFlops's code: 
    https://www.kaggle.com/sekrier/50ms-scoring-just-with-numpy"""      
    TCH = np.sum(chigif[range(n_children),pred])
    TSH = np.sum(gifchi[pred,range(n_children)])     
    ANH = float(math.pow(TCH*10.0,3) + math.pow(np.sum(TSH),3.0))    / float(math.pow(2000000000,3))
    return ANH
def onoderaIndex(pred): ## onodera's solution
    """https://github.com/KazukiOnodera/Santa2017/tree/master/solution"""      
    TCH = np.sum(chigif[range(n_children),pred])
    TSH = np.sum(gifchi[pred,range(n_children)])   
    CH = TCH*6 
    SH = TSH*6
    return CH, SH

CH_max = 1173959622
SH_max = 1703388
print("Base score: %12.10f" % avgNH(pred_base['GiftId'].values))
print("TFTurobo's improver score: %12.10f" % avgNH(pred_better['GiftId'].values))


# Calcuate gift and children happiness for each prediction

# In[5]:


pred_base=getGCvalMatrx(pred_base)
pred_better=getGCvalMatrx(pred_better)
def plotGCVal(minVal,maxVal,minY=0,maxY=1000000):
    pred_base.groupby(by='gval').size().plot(        label='gift value Base',xlim=(minVal,maxVal),ylim=(minY,maxY))
    pred_base.groupby(by='cval').size().plot(        label='children value Base',xlim=(minVal,maxVal),ylim=(minY,maxY))
    pred_better.groupby(by='gval').size().plot(        label='gift value Better',xlim=(minVal,maxVal),ylim=(minY,maxY))
    pred_better.groupby(by='cval').size().plot(        label='children value Better',xlim=(minVal,maxVal),ylim=(minY,maxY))
    plt.legend()
    plt.xlabel('Children (gift) happiness ')
    plt.ylabel('Number of Children (gift)')


# Overall happiness for Santa and children

# In[6]:


plotGCVal(0,1000)


# Zoom in to check 7- maximal happiness

# In[7]:


plotGCVal(7,201,0,1500)


# Zoom in to 190-200

# In[8]:


plotGCVal(190,201,0,700000)


# zoom in to 0-20. close to 1000,000 gift has -1 happines

# In[9]:


plotGCVal(-3,20)
print('Gift unhappy: %d'%pred_better.groupby(by='cval').size()[-1])


# In[10]:


plotGCVal(-3,20,0,20)


# **Conclusion:**
#     just make Santa as unhappy as possible and maximise children's happiness

# Correction: above conclusion seems wrong based on ONODERA and hoxosh's solution (Many thanks for providing the solution and congratulations!
# https://github.com/KazukiOnodera/Santa2017/tree/master/solution )

# In[11]:


CH_base, SH_base = onoderaIndex(pred_base['GiftId'].values)
CH_better, SH_better = onoderaIndex(pred_better['GiftId'].values)
print("onodera score: CH:%d,SH:%d"%(CH_max,SH_max))
print("Base score: CH:%d,d_CH: %d; SH:%d, d_SH: %d"%(CH_base,CH_base-CH_max,SH_base,SH_base-SH_max))
print("TFT score: CH:%d,d_CH: %d; SH:%d, d_SH: %d"%(CH_better,CH_better-CH_max, SH_better, SH_better-SH_max))


# Child 34267 and child 34268 is hard to pleased according to ONODERA and hoxosh's solution and will only make an optimal solution if are give gift number 207. Let's check our solutions!

# In[12]:


pred_base.loc[34267]


# In[13]:


pred_base.loc[34268]


# In[14]:


pred_better.loc[34267]


# In[15]:


pred_better.loc[34268]


# ZFTurbo's solution got the magic gift number (207) for this twin.
