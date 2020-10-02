#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test = pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


#SIMPLIFICATION NORMALIZE =  cossimilarity becomes DOT ( Xtrain, Qn.T)
#Qt=np.linalg.inv(np.diag(Sigma)) #.dot(U.T) #.dot(Q)
#pd.DataFrame( (U/Sigma).dot(Q) ).sort_values(0)
from sklearn import preprocessing
Xtrain = preprocessing.normalize(train.drop('label',axis=1), norm='l2')
Xtest=preprocessing.normalize(test.drop('id',axis=1), norm='l2')


# In[ ]:


Q=Xtest[0]
top10=pd.DataFrame(np.dot(Xtrain,Q)).sort_values(0)[-10:]
[str(x)[-1:] for x in top10.index]
top10[0]


# In[ ]:


results=[] #speedup with list ipo pandas
for xi in range(len(test)):
    Q=Xtest[xi]
    top10=pd.DataFrame(np.dot(Xtrain,Q)).sort_values(0)[-10:]
    results.append([train.iat[x,0] for x in top10.index]) #+[top10[-1:].values[0]]  #correct error with  train.iloc[x].label  str(x)[-1:] 


# In[ ]:


results=pd.DataFrame(results)
results


# In[ ]:


def most_frequent(List): 
    return max(set(List), key = List.count) 

results['pred']=-1

for xi in range( len(test) ):
    results.iat[xi,10]=most_frequent(list( results.iloc[xi] ) )
results


# In[ ]:


results['id']=test['id']
results['label']=results['pred']
results[['id','label']].to_csv(path_or_buf ='cossim.csv', index=False)

