#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')
test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')


# In[ ]:


magicNum = 131073
default_cols = [c for c in train.columns if c not in ['id', 'target','target_pred', 'wheezy-copper-turtle-magic']]
cols = [c for c in default_cols]
sub = pd.read_csv('../input/sample_submission.csv')
sub.to_csv('submission.csv',index=False)
train.shape,test.shape


# There are many posts explaining in detail why the n_clusters_per_class param should be 3. Hence ideally using gaussian cluster the dataset will be divided to 6 clusters, EQUALLY! But when you do this on train+public_test, you'll not get same number clusters because the data is not full. Hence I tried it online and get perfect result there.
# 
# Once eace sub-group is divided into 6 clusters, the remaining thing is easy - for each cluster, the majoriy label is the correct label. Till now we could get ideally perfect result. And this gives you public LB 0.97443 and private LB 0.97579 - the later is pretty close to a golden medal.
# 
# But the evaluation metric is AUC, not precision score, so I am wondering whether there is any way to improve the score based on the above perfect classification. 
# 
# 
# Consider there are 20 labels, the perfect classification is ten zeros follows ten ones. But there are 4 numbers fllipped. If we don't know which numbers are fllipped, the idea score we could get is 0.8:

# In[ ]:


y_perfect = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
y_flliped = [1,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,1]
roc_auc_score(y_perfect,y_flliped)


# Now, consider we have info that some fllipes may happen at "some" places, for example, we know in the first three numbers there is a flip, in 4 to 5 numbers there is another flip, in 13-14 numbers there is a flip, and in last 3 numbers there is a flip. Then, based on this info, we set each number the converted score which reflects the flipping probability, and the AUC score improves!

# In[ ]:


y_preds = [0.33,0.33,0.33,0.5,0.5,0,0,0,0,0,1,1,0.5,0.5,1,1,1,0.66,0.66,0.66]
roc_auc_score(y_flliped,y_preds)


# Return to this topic, although we don't know the exact labels which are flliped, but if we could infer the labels with higher probablity to get flliped, we could imporve the AUC  -  for those '1's with higher probability to be flliped to 0, they should have lower score, and for those '0's with higher probability to be flliped to 1, they should have higher score. 
# 
# So now it turns into a mathmatical problem: give you 1024 green balls and you will choose 51 balls with 0.5 probability to chagne the green ball to red ball, after the operation, the 1024 balls are seperated into three parts, you know the exact red ball in first part(which is our training set), what's the expected red balls on the second part and third part?
# 
# Sadlly this problem is not easy to solve, also the variance seems high, which means even you calculated the mathematical expectation, it is quite unstable.
# 
# Hence I just do it with luck: 
# 1. for each sub group, if in train set there are more flliped values, I guess the fllipped values will be less in test set. vice versa. 
# 1. for each test sub group, I guess if the group is bigger, it should have more fllipped values. 
# 
# Based on the above rules and with some parameters trying, I quickly reached the public top1. But sadly to say, above rules are quite un-stable, hence in private set, they fails to 60+ position. Actually some other parameters works bad in public LB works quite well in privates....
# 
# Anyway, interesting puzzels and fun time really. And many many thanks to those who shared quite a lot thoughts, like Vlad, Chris... Thank you!

# In[ ]:


if sub.shape[0] == magicNum:
    [].shape   

preds=np.zeros(len(test))
train_err=np.zeros(512)
test_err=np.zeros(512)

for i in range(512):  
    
    X = train[train['wheezy-copper-turtle-magic']==i].copy()
    Y = X.pop('target').values
    X_test = test[test['wheezy-copper-turtle-magic']==i].copy()

    idx_train = X.index 
    idx_test = X_test.index
    
    X.reset_index(drop=True,inplace=True)
    
    X = X[cols].values             
    X_test = X_test[cols].values

    vt = VarianceThreshold(threshold=2).fit(X)
    
    X = vt.transform(X)         
    X_test = vt.transform(X_test)
    X_all = np.concatenate([X,X_test])
    train_size = len(X)
    test1_size = test[:131073][test[:131073]['wheezy-copper-turtle-magic']==i].shape[0]
    compo_cnt = 6
    for ii in range(30):
        gmm = GaussianMixture(n_components=compo_cnt,init_params='random',covariance_type='full',max_iter=100,tol=1e-10,reg_covar=0.0001).fit(X_all)
        labels = gmm.predict(X_all)
        
        cntStd = np.std([len(labels[labels==j]) for j in range(compo_cnt)])
        #there are chances that the clustering doesn't converge, so we only choose the case that it clustered equally
        #in which case, the sizes are 171,170,171,170,...
        if round(cntStd,4) == 0.4714:
            check_labels = labels[:train_size]
            cvt_labels=np.zeros(len(labels))

            #first get the perfect classification label
            for iii in range(compo_cnt):
                mean_val = Y[check_labels==iii].mean()
                mean_val = 1 if mean_val > 0.5 else 0
                cvt_labels[labels==iii] = mean_val
            
            #then try to predict the expected err for the test set
            train_err[i] = len(Y[Y != cvt_labels[:train_size]])
            if (train_err[i] >= 10) and (train_err[i] <= 15):
                train_err[i] = 12.5
            exp_err = max(0,(25 - train_err[i])/(train_size + test1_size))

            for iii in range(compo_cnt):
                mean_val = Y[check_labels==iii].mean()
                mean_val = (1-exp_err) if mean_val > 0.5 else exp_err
                cvt_labels[labels==iii] = mean_val

            preds[idx_test] = cvt_labels[train_size:]
            break

sub['target'] = preds
sub.to_csv('submission.csv',index=False)

