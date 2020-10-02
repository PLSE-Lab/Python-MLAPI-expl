#!/usr/bin/env python
# coding: utf-8

# ### This Kernel shows how to train HMM without any priors about groups. 
# #### Basic Steps
# 1. Estimated distribution using regions of high confidence intervals
# 2. Estimated Transition Matrix using preliminary predictions
# 3. Estimated Mean and Standard Deviations
# 4. Trained the Posterior Decoder models
# This Kernel beats all the public lb scores in the private lb. The findings from this kernel helped to build powerful random forest models. As this approach don't hold any prior, It's highly powerful with unseen data. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from IPython.display import display, HTML
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

train_df = pd.read_csv("/kaggle/input/scaling-3/new_train.csv")
test_df = pd.read_csv("/kaggle/input/scaling-3/new_test.csv")
train_df['batch']=((train_df.time-0.0001)//50).astype(int)
test_df['batch']=((test_df.time-0.0001)//50).astype(int)
train_df['mini_batch']=((train_df.time-0.0001)//10).astype(int)
test_df['mini_batch']=((test_df.time-0.0001)//10).astype(int)
train_df['mini_mini_batch']=((train_df.time-0.0001)//0.5).astype(int)
test_df['mini_mini_batch']=((test_df.time-0.0001)//0.5).astype(int)

shifted = [4,9]
train_df.loc[train_df.batch.isin(shifted),'signal'] = train_df.loc[train_df.batch.isin(shifted),'signal'] #+ 2.72
shifted = [55,57]
test_df.loc[test_df.mini_batch.isin(shifted),'signal'] = test_df.loc[test_df.mini_batch.isin(shifted),'signal'] #+ 2.72

# # Dirty Batches
# dirty = [x for x in range(728,765)]
# train_df = train_df[~train_df.mini_mini_batch.isin(dirty)]

# for batch in [10,9,8]: train_df.loc[(train_df.batch==batch) ,'batch'] = batch+1
    
# train_df.loc[(train_df.batch==7) & (train_df.mini_mini_batch>=765),'batch'] = 8

train_df = train_df.reset_index(drop=True)

train_df.batch.value_counts()


# In[ ]:


train_df.groupby(['batch','open_channels']).signal.agg(['mean','std'])


# In[ ]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

from scipy.stats import norm


# Estimating Distribution & Transition Matrix

# In[ ]:


from sklearn.metrics import f1_score,accuracy_score
signal_stats = train_df[['open_channels','signal']].groupby('open_channels').agg(['mean','std']).reset_index()
window = 0.1
signal_lb = signal_stats[('signal', 'mean')] - window
signal_ub = signal_stats[('signal', 'mean')] + window

for oc in range(11): train_df.loc[(train_df.signal>signal_lb.loc[oc])&(train_df.signal<signal_ub.loc[oc]),'prediction']=oc
    
#Estimating Percentiles  0.9275004465932014
distribution = train_df[['signal','mini_batch','prediction']].groupby(['mini_batch','prediction']).agg('count').reset_index()
distribution = distribution[distribution.signal>=5]
distribution['signal'] = distribution.groupby(['mini_batch'])['signal'].apply(lambda x: x.cumsum())
distribution = distribution.pivot_table(values='signal',index=distribution.mini_batch,columns='prediction')
distribution = distribution.div(distribution.max(axis=1),axis=0).fillna(-1)
train_distribution = distribution
Actual_Transitions = {}
Transition_matrices = {}

for mini_batch in tqdm(range(50)):
#     print('mini_batch',mini_batch)
    temp_df = train_df[train_df.mini_batch==mini_batch]
    if temp_df.shape[0]>1000:
        oc=0
        percentile = distribution.loc[mini_batch,oc]
        temp_df['prediction'] = -1
        while(percentile==-1):
            oc+=1
            percentile = distribution.loc[mini_batch,oc]
        while(percentile!=-1 and oc<11):
            threshold = temp_df.signal.quantile(percentile)
            temp_df.loc[(temp_df.prediction==-1) & (temp_df.signal<=threshold),'prediction'] =oc
            oc+=1
            if oc<11: percentile = distribution.loc[mini_batch,oc]
                
            
            temp_df['next_oc'] = temp_df.open_channels.shift(-1)
            trans_mat = temp_df[['open_channels','next_oc','signal']].groupby(['open_channels','next_oc']).agg('count').reset_index()
            trans_mat = trans_mat.pivot_table(values='signal',index=trans_mat.open_channels,columns='next_oc')
            transition = pd.DataFrame(columns = np.unique(temp_df.open_channels),index = np.unique(temp_df.open_channels))
            transition.loc[:,:] = 1 #Smoothing Factor
            transition += trans_mat
            transition = transition.fillna(1)
            transition = transition.div(transition.sum(axis=1),axis=0)
            Actual_Transitions[mini_batch] = transition
            
            temp_df['next_oc'] = temp_df.prediction.shift(-1)
            trans_mat = temp_df[['prediction','next_oc','signal']].groupby(['prediction','next_oc']).agg('count').reset_index()
            trans_mat = trans_mat.pivot_table(values='signal',index=trans_mat.prediction,columns='next_oc')
            transition = pd.DataFrame(columns = np.unique(temp_df.prediction),index = np.unique(temp_df.prediction))
            transition.loc[:,:] = 0.5 #Smoothing Factor
            transition += trans_mat
            transition = transition.fillna(0.5)
            transition = transition.div(transition.sum(axis=1),axis=0)
            Transition_matrices[mini_batch] = transition


# In[ ]:


window = 0.1
signal_lb = signal_stats[('signal', 'mean')] - window
signal_ub = signal_stats[('signal', 'mean')] + window

for oc in range(11): test_df.loc[(test_df.signal>signal_lb.loc[oc])&(test_df.signal<signal_ub.loc[oc]),'prediction']=oc
    
#Estimating Percentiles
distribution = test_df[['signal','mini_batch','prediction']].groupby(['mini_batch','prediction']).agg('count').reset_index()
distribution = distribution[distribution.signal>=10]
distribution['signal'] = distribution.groupby(['mini_batch'])['signal'].apply(lambda x: x.cumsum())
distribution = distribution.pivot_table(values='signal',index=distribution.mini_batch,columns='prediction')
distribution = distribution.div(distribution.max(axis=1),axis=0).fillna(-1)
test_distribution = distribution

for mini_batch in tqdm(range(50,70)):
    temp_df = test_df[test_df.mini_batch==mini_batch]
    oc=0
    percentile = distribution.loc[mini_batch,oc]
    temp_df['prediction'] = -1
    while(percentile==-1):
        oc+=1
        percentile = distribution.loc[mini_batch,oc]
    while(percentile!=-1 and oc<11):
        threshold = temp_df.signal.quantile(percentile)
        temp_df.loc[(temp_df.prediction==-1) & (temp_df.signal<=threshold),'prediction'] =oc
        oc+=1
        if oc<11: percentile = distribution.loc[mini_batch,oc]
    temp_df['next_oc'] = temp_df.prediction.shift(-1)
    trans_mat = temp_df[['prediction','next_oc','signal']].groupby(['prediction','next_oc']).agg('count').reset_index()
    trans_mat = trans_mat.pivot_table(values='signal',index=trans_mat.prediction,columns='next_oc')
    transition = pd.DataFrame(columns = np.unique(temp_df.prediction),index = np.unique(temp_df.prediction))
    transition.loc[:,:] = 0.5 #Smoothing Factor
    transition += trans_mat
    transition = transition.fillna(0.5)
    transition = transition.div(transition.sum(axis=1),axis=0)
    Transition_matrices[mini_batch] = transition


# In[ ]:


test_distribution


# Standard Deviation Estimated from: https://www.kaggle.com/ks2019/estimating-standard-deviations-on-new-data?scriptVersionId=34750556

# In[ ]:


signal_means = train_df[['open_channels','signal']].groupby('open_channels').agg('mean')
standard_dev = {
 0: 0.24069495895789328,
 1: 0.24661370964721863,
 2: 0.2462656210881039,
 3: 0.24593499463670376,
 4: 0.24285132812035926,
 5: 0.26328499147888296,
 6: 0.24325060224204165,
 7: 0.24219770330946155,
 8: 0.24284409806346507,
 9: 0.24213353276403837,
 10: 0.24582776879974166,
 11: 0.24496232384251448,
 12: 0.24419957371390066,
 13: 0.24390344537922137,
 14: 0.24401429614362458,
 15: 0.2650094712461518,
 16: 0.26515878374200286,
 17: 0.26653346413097956,
 18: 0.2666851765378241,
 19: 0.2661300555645225,
 20: 0.4069574034485397,
 21: 0.4057626703610925,
 22: 0.40661851101785595,
 23: 0.4014537024692114,
 24: 0.40342408294931176,
 25: 0.2864659504038526,
 26: 0.28792615084155104,
 27: 0.2861063013595865,
 28: 0.28894890652916927,
 29: 0.28746591555904344,
 30: 0.24588901544435604,
 31: 0.24562918245064735,
 32: 0.24454967653259863,
 33: 0.24712026725094094,
 34: 0.24671466539168418,
 35: 0.2944529092246321,
 36: 0.2897700004860426,
 38: 0.29459674149215787,
 39: 0.27525129540509075,
 40: 0.2855328333419974,
 41: 0.28533031164314704,
 42: 0.28859497286113045,
 43: 0.28256191424770405,
 44: 0.2841569645597404,
 45: 0.4066384374459793,
 46: 0.4061604936865585,
 47: 0.40514687628102375,
 48: 0.4067380853754845,
 49: 0.40637947598259294,
 50: 0.24073643515935061,
 51: 0.2754600438388737,
 52: 0.29036115205832036,
 53: 0.2410069541187777,
 54: 0.2448870602255716,
 55: 0.40433399257591063,
 56: 0.2861781553817768,
 57: 0.40588197314984553,
 58: 0.24015921663211293,
 59: 0.2724222128619417,
 60: 0.2388213978815562,
 61: 0.2409860881280551,
 62: 0.2404194193333159,
 63: 0.2454693052384903,
 64: 0.24194863011797696,
 65: 0.2413491063034735,
 66: 0.24715158528432513,
 67: 0.24293092821730067,
 68: 0.24327976456590195,
 69: 0.24585838370130864
}


# ### Posterior Decoder from: https://www.kaggle.com/group16/lb-0-936-1-feature-forward-backward-vs-viterbi

# In[ ]:


class ViterbiClassifier:
    def __init__(self):
        self._p_trans = None
        self._p_signal = None
    
    def fit(self, mini_batch):
        self._n_states = 11
        self._states = list(range(self._n_states))
        
        self._p_trans = self.markov_p_trans(mini_batch)
        
        self._dists = []
        for s in np.arange(0, 11):
            self._dists.append((signal_means.loc[s,'signal'], standard_dev[mini_batch]))
        
        return self
        
    def predict(self, x, p_signal=None, proba=False):
        if p_signal is None:
            p_signal = self.markov_p_signal(x)

        preds, probs = self.viterbi(self._p_trans, p_signal[self._states], x)
        
        
        return preds, probs
    
    def markov_p_signal(self, signal):
#         print(self._n_states, len(signal))
        p_signal = np.zeros((self._n_states, len(signal)))
        for k, dist in enumerate(self._dists):
            p_signal[k, :] = norm.pdf(signal, *dist)
            
        return p_signal
    
    def markov_p_trans(self, mini_batch):
        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution
        transition = Transition_matrices[mini_batch]
        TM = pd.DataFrame(index=range(11),columns=range(11))
        TM.loc[transition.index,transition.columns] = transition
        TM = TM.fillna(0)
        return TM.values
    
    def viterbi(self, p_trans, p_signal, signal):
        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution
        offset = 10**(-20) # added to values to avoid problems with log2(0)

        p_trans_tlog  = np.transpose(np.log2(p_trans  + offset)) # p_trans, logarithm + transposed
        p_signal_tlog = np.transpose(np.log2(p_signal + offset)) # p_signal, logarithm + transposed
        
        T1 = np.zeros(p_signal.shape)
        T2 = np.zeros(p_signal.shape)

        T1[:, 0] = p_signal_tlog[0, :]
        T2[:, 0] = 0

        for j in range(1, p_signal.shape[1]):
            for i in range(len(p_trans)):
                T1[i, j] = np.max(T1[:, j - 1] + p_trans_tlog[:, i] + p_signal_tlog[j, i])
                T2[i, j] = np.argmax(T1[:, j - 1] + p_trans_tlog[:, i] + p_signal_tlog[j, i])
        
        x = np.empty(p_signal.shape[1], 'B')
        x[-1] = np.argmax(T1[:, p_signal.shape[1] - 1])
        for i in reversed(range(1, p_signal.shape[1])):
            x[i - 1] = T2[x[i], i]
    
        return x, T1
    
class PosteriorDecoder:
    def __init__(self):
        self._p_trans = None
        self._p_signal = None
    
    def fit(self, mini_batch):
        self._n_states = 11
        self._states = list(range(self._n_states))
        
        self._p_trans = self.markov_p_trans(mini_batch)
        
        self._dists = []
        for s in np.arange(0, 11):
            self._dists.append((signal_means.loc[s,'signal'], standard_dev[mini_batch]))
        
        return self
        
    def predict(self, x, p_signal=None, proba=False):
        if p_signal is None:
            p_signal = self.markov_p_signal(x)
        preds,probs = self.posterior_decoding(self._p_trans, p_signal[self._states])
        
        return preds, probs
    
    def markov_p_signal(self, signal):
        p_signal = np.zeros((self._n_states, len(signal)))
        for k, dist in enumerate(self._dists):
            p_signal[k, :] = norm.pdf(signal, *dist)
            
        return p_signal
    
    def markov_p_trans(self, mini_batch):
        # https://www.kaggle.com/friedchips/the-viterbi-algorithm-a-complete-solution
        transition = Transition_matrices[mini_batch]
        TM = pd.DataFrame(index=range(11),columns=range(11))
        TM.loc[transition.index,transition.columns] = transition
        TM = TM.fillna(0)
        return TM.values
    
    def forward(self, p_trans, p_signal):
        """Calculate the probability of being in state `k` at time `t`, 
           given all previous observations `x_1 ... x_t`"""
        T1 = np.zeros(p_signal.shape)
        T1[:, 0] = p_signal[:, 0]
        T1[:, 0] /= np.sum(T1[:, 0])

        for j in range(1, p_signal.shape[1]):
            for i in range(len(p_trans)):
                T1[i, j] = p_signal[i, j] * np.sum(T1[:, j - 1] * p_trans[i, :])
            T1[:, j] /= np.sum(T1[:, j])

        return T1

    def backward(self, p_trans, p_signal):
        """Calculate the probability of observing `x_{t + 1} ... x_n` if we 
           start in state `k` at time `t`."""
        T1 = np.zeros(p_signal.shape)
        T1[:, -1] = p_signal[:, -1]
        T1[:, -1] /= np.sum(T1[:, -1])

        for j in range(p_signal.shape[1] - 2, -1, -1):
            for i in range(len(p_trans)):
                T1[i, j] = np.sum(T1[:, j + 1] * p_trans[:, i] * p_signal[:, j + 1])
            T1[:, j] /= np.sum(T1[:, j])

        return T1
    
    def posterior_decoding(self, p_trans, p_signal):
        fwd = self.forward(p_trans, p_signal)
        bwd = self.backward(p_trans, p_signal)

        preds = np.empty(p_signal.shape[1], 'B')
        probs = np.zeros((11,fwd.shape[1]))
        for i in range(p_signal.shape[1]):
            preds[i] = np.argmax(fwd[:, i] * bwd[:, i])
            probs[:,i] = fwd[:, i] * bwd[:, i]

        return preds,probs


# Idea to try: Use both Forward and backward prob as feature in any supervised model

# In[ ]:


oof_probs = []
oof_predictions = []
for mini_batch in range(50):
    if mini_batch in Transition_matrices:
        signal = train_df[train_df.mini_batch==mini_batch].signal.values
        open_channels = train_df[train_df.mini_batch==mini_batch].open_channels.values
        viterbi = PosteriorDecoder().fit(mini_batch)
        viterbi_predictions,viterbi_probabilities = viterbi.predict(signal)
        print(mini_batch,accuracy_score(viterbi_predictions,open_channels))
        train_df.loc[train_df.mini_batch==mini_batch,'prediction'] = viterbi_predictions
        oof_probs.append(viterbi_probabilities)
        oof_predictions.append(viterbi_predictions)
oof_probs = np.transpose(np.concatenate(oof_probs,axis=1))
oof_predictions = np.concatenate(oof_predictions)
np.save('oof_probs.npy', oof_probs)  
np.save('oof_predictions.npy', oof_predictions)  


# In[ ]:


print(f1_score(train_df.open_channels,train_df.prediction,average='macro'))
pd.DataFrame(confusion_matrix(train_df.open_channels,train_df.prediction))


# In[ ]:


test_probs = []
test_predictions = []
for mini_batch in tqdm(range(50,70)):
    if mini_batch in Transition_matrices:
        signal = test_df[test_df.mini_batch==mini_batch].signal.values
        viterbi = PosteriorDecoder().fit(mini_batch)
        viterbi_predictions,viterbi_probabilities = viterbi.predict(signal)
        test_probs.append(viterbi_probabilities)
        test_predictions.append(viterbi_predictions)
test_probs = np.transpose(np.concatenate(test_probs,axis=1))
np.save('test_probs.npy', test_probs)  
test_predictions = np.concatenate(test_predictions)
np.save('test_predictions.npy', test_predictions)  


# In[ ]:


submission_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
submission_df['open_channels'] = test_predictions
submission_df.to_csv("submission.csv", float_format='%.4f', index=False)
submission_df.open_channels.value_counts()


# In[ ]:


for oc in range(11):
    print("Open Channels",oc)
    print(f1_score((train_df.open_channels==oc).astype(int),(train_df.prediction==oc).astype(int)))


# In[ ]:


test_df['prediction'] = test_predictions
distribution = test_df[['signal','mini_batch','prediction']].groupby(['mini_batch','prediction']).agg('count').reset_index()
distribution = distribution.pivot_table(values='signal',index=distribution.mini_batch,columns='prediction')
distribution = distribution.div(distribution.sum(axis=1),axis=0).fillna(-1) 
distribution

