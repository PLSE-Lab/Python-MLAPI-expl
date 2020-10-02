#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train 990 leaves
# test 594 samples

import numpy as np
import pandas as pd
from numpy.linalg import inv

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train.species
labelid = train.id
testid =test.id
train = train.drop(['species', 'id'], axis=1)
test = test.drop(['id'], axis=1)
#A=train
A=train.transpose()

# singular value decomposition
U,s,V=np.linalg.svd(A,full_matrices=False)
# reconstruct
S=np.diag(s)
Q=test.transpose()
iS=inv(S)
US=np.dot(U,iS)
Qtemp=np.dot(Q.transpose(),US)
simila=np.dot(Qtemp,V)/np.dot(np.abs(Qtemp),np.abs(V))
for xya in range(0,594):
  for xyz in range (len(V)):
    simila=np.dot(Qtemp[xya,:],V[:,xyz])/np.dot(np.abs(Qtemp[xya,:]),np.abs(V[:,xyz]))*100
    if simila>50:
      print(testid[xya],labels[xyz],labelid[xyz], round(simila,1),"%" ) 


# 
# 
# Lowering the similarity treshold , you have them all.
# 
# 

# In[ ]:


# train 990 leaves
# test 594 samples

import numpy as np
import pandas as pd
from numpy.linalg import inv

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train.species
labelid = train.id
testid =test.id
train = train.drop(['species', 'id'], axis=1)
test = test.drop(['id'], axis=1)
#A=train
A=train.transpose()

# singular value decomposition
U,s,V=np.linalg.svd(A,full_matrices=False)
# reconstruct
S=np.diag(s)
Q=test.transpose()
iS=inv(S)
US=np.dot(U,iS)
Qtemp=np.dot(Q.transpose(),US)
simila=np.dot(Qtemp,V)/np.dot(np.abs(Qtemp),np.abs(V))
for xya in range(0,594):
  for xyz in range (len(V)):
    simila=np.dot(Qtemp[xya,:],V[:,xyz])/np.dot(np.abs(Qtemp[xya,:]),np.abs(V[:,xyz]))*100
    if simila>80:
      print(testid[xya],labels[xyz],labelid[xyz], round(simila,1),"%" ) 

