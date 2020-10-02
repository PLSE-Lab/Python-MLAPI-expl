#!/usr/bin/env python
# coding: utf-8

# 
# 
# In https://www.kaggle.com/miklgr500/ghost-drift-and-outliers author cleans up outliers and removes "ghost drift". After that he's able to get .927 LB with only boundary classifier.
# 
# I want to make a further step with this data. First I get class probabilities for each sample with GaussianNB classifier - it gives me .929 macro f1 on training data.
# 
# Then I create a markov chain transition probability matrix and use it as a weight matrix for RNN. This way the class probability for each sample is the product of it's classification probability based on signal value via GaussianNB and the probaility to get this class via transition from previos samples via Markov transition matrix. 
# 
# As a result we essintially get an RNN that we don't need to train as all the weights obtained via classic models.
# 
# I also use this RNN in a bidirectional way to get information from future transitions. 
# 
# Using this RNN boosts the score up to .935
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm.notebook import tqdm
import os


# In[ ]:


#from https://www.kaggle.com/miklgr500/ghost-drift-and-outliers
TRAIN_DIR = "/kaggle/input/ion-ghost-drift-removed/"


# In[ ]:


dftrain = pd.read_csv(os.path.join(TRAIN_DIR, "train_clean_kalman.csv"))
dftrain.head()


# In[ ]:


gnb = GaussianNB()
gnb.fit(dftrain[["signal"]], dftrain.open_channels)
proba = gnb.predict_proba(dftrain[["signal"]])
pred = np.argmax(proba, axis=1)
f1_score(dftrain.open_channels, pred, average="macro")


# In[ ]:


MC = np.zeros((11,11))
prev_c = None
for c in tqdm(dftrain.open_channels.values):
    if prev_c is not None:
        MC[c, prev_c] += 1
    prev_c = c
MC_normed = (MC / MC.sum(axis=0))


# In[ ]:


class RNN():
    def __init__(self, weights):
        self.weights = weights
        self.hidden = None
    def forward(self, x):
        ret = x.copy()
        if self.hidden is not None:
            ret = self.hidden.dot(self.weights) * ret
        self.hidden = ret / ret.sum()
        return ret
    def reset(self):
        self.hidden = None


# In[ ]:


class BiDirectionalPredict:
    def __init__(self, weights, silent=False):
        self.forwardDirectionRNN = RNN(weights)
        self.backwardDirectionRNN = RNN(weights.T)
        self.silent = silent
    
    def predict_proba(self, samples):
        self.forwardDirectionRNN.reset()
        self.backwardDirectionRNN.reset()
        pred_fwd = [self.forwardDirectionRNN.forward(x) for x in tqdm(samples, disable=self.silent)]
        bred_bkwd = [self.backwardDirectionRNN.forward(x) for x in tqdm(samples[::-1], disable=self.silent)]
        
        bidirect_pred = np.array(pred_fwd) * np.array(bred_bkwd[::-1])
        return bidirect_pred
    
    def predict(self, samples):
        proba = self.predict_proba(samples)
        return proba.argmax(axis=-1)


# In[ ]:


clf = BiDirectionalPredict(MC_normed, silent=True)


# In[ ]:


pred_mc = []
for grp in tqdm(np.array_split(proba, 10)):
    pred_grp = clf.predict(grp)
    pred_mc.extend(pred_grp)


# In[ ]:


print(classification_report(dftrain.open_channels, pred_mc))


# In[ ]:


f1_score(dftrain.open_channels, pred_mc, average="macro")


# In[ ]:


dftest = pd.read_csv(os.path.join(TRAIN_DIR, "test_clean_kalman.csv"))


# In[ ]:


proba_test = gnb.predict_proba(dftest[["signal"]])


# In[ ]:


pred_test = clf.predict(proba_test)


# In[ ]:


sub = pd.read_csv(os.path.join("/kaggle/input/liverpool-ion-switching/", "sample_submission.csv"), dtype={"time":object})


# In[ ]:


sub.open_channels = pred_test


# In[ ]:


sub.open_channels.value_counts()


# In[ ]:


sub.to_csv("submission.csv", index=False)

