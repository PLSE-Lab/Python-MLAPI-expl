#!/usr/bin/env python
# coding: utf-8

# # Reducing Commercial Aviation Fatalities

# ### EDA for the competition Aviation Fatalities on Kaggle

# In[ ]:


# Imports for data manipulation and data vizualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
scaler = MinMaxScaler()
import os
print(os.listdir("../input"))

print ("Hello World")

train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head(5)


# In[ ]:


#Scaling the data
import gc 
trainN = train.loc[:, train.dtypes == np.float64]
trainN['seat'] = train.seat
trainN['crew'] = train.crew
trainN[:] = scaler.fit_transform(trainN[:])
trainN['experiment'] = train['experiment'].map({'CA': -1, 'DA': 0,'SS':1})


# #### Let's see how features are distributed by class

# In[ ]:


trainA = trainN[train.event=='A']
trainB = trainN[train.event=='B']
trainC = trainN[train.event=='C']
trainD = trainN[train.event=='D']

fig = plt.figure(figsize=(65,65))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.grid()

for row,i in zip(trainN,range(0,len(trainN.columns))):
    
    plt.subplot(len(trainN.columns)/3, 4, i+1)
    plt.hist(trainA[row],label='A',alpha=0.4)
    plt.hist(trainB[row],label='B',alpha=0.4)
    plt.hist(trainC[row],label='C',alpha=0.4)
    plt.hist(trainD[row],label='D',alpha=0.4)
    plt.xlabel(row,size=26)
    plt.legend(fontsize=26)


# #### In this plot we can see that some features represents more embracing values,in this way, we also can notice that a lot of represents values with low variance. As this plot was divided by classes, in this group of histograms we can observe that some features could be useless 
# 
# #### Now lets see how are the correlation of the features 

# In[ ]:


fig = plt.figure(figsize=(55,25))
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.subplot(2, 2, 1)
corr = trainA.corr()

a = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.tick_params(axis='y', which='major', labelsize=26)
plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)

plt.subplot(2, 2, 2)
corr = trainB.corr()
b = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.tick_params(axis='y', which='major', labelsize=26)
plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)
plt.subplot(2, 2, 3)
corr = trainC.corr()
c = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.tick_params(axis='y', which='major', labelsize=26)
plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)
plt.subplot(2, 2, 4)
corr = trainD.corr()
d = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.tick_params(axis='y', which='major', labelsize=26)
plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)
 


# #### lets see now how every crew features are distributed 

# In[ ]:


crews = np.unique(train.crew)
grCrews = []
for c in crews:
    grCrews.append(trainN[train.crew==c])
fig = plt.figure(figsize=(65,65))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.grid()
for row,i in zip(trainN,range(0,len(trainN.columns))):
    
    plt.subplot(len(trainN.columns)/3, 4, i+1)
    for gr,l in zip(grCrews,np.unique(train.crew)):
        plt.hist(gr[row].values,label=str(l),alpha=0.4)
    plt.xlabel(row,size=26)
    plt.legend(fontsize=26)


# #### it seems like every crew have your on characteristics, special in the features more relevant, by variance, showed in the first histogram
# 
# #### lets see now the most important features by variance found by PCA technique

# In[ ]:


cov_mat = np.cov(trainN.values.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,len(trainN.columns)+1), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,len(trainN.columns)+1), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.grid()
plt.show()


# #### only with 5\4 features with explained more than 90% of the data!!!
# #### lets see how data looks like in 2D dimension

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
space2D = pca.fit_transform(trainN.values)
labels = train.event.map({"A":0,"B":1,"C":2,"D":3}).values
plt.scatter(space2D[:,0],space2D[:,1],c=labels)
plt.grid()


# #### Now lets see if the data in test set seems like that train data , thereby, we can see if  the distribution in train data sounds equal to test data.

# In[ ]:



'''
test = pd.read_csv("../input/test.csv")

i = 0
eeg_features  = trainN.columns
for eeg in eeg_features:
    i += 1
    plt.subplot(len(test.columns)/4+1, 4, i)
    sns.distplot(train[eeg], label='Test set', hist=False)
    sns.distplot(test[eeg], label='Train set', hist=False)
    #plt.xlim((-500, 500))
    plt.legend()
    plt.xlabel(eeg, fontsize=12)

plt.show()
'''
## No space left in disk


# #### Looks like ok! In this way, we can conclude that cluster analysis could be a good solution for this problem
