#!/usr/bin/env python
# coding: utf-8

# # Can we Trust CV and LB?
# Kaggle's "Don't Overfit II" is a unique competition where the training dataset only has 250 observations and the public test dataset has 1975 observations. Can we trust the training dataset CV and public dataset LB? Do higher AUC scores on these small datasets indicate that a more accurate model was found? In this kernel we explore this question with four experiments. We find that we can trust LB but must be careful trusting CV. 
# 
# Afterward in the appendix, using insights from these experiments, we estimate how many useful variables exist in the real dataset.
# # Experiment 1 : CV with 300 useless variables
# Let's create a synthetic sample of size 250. We will create 300 **useless** variables and a completely random (meaningless) target and see what CV says.

# In[ ]:


import numpy as np, pandas as pd, os
np.random.seed(300)

# GENERATE USELESS VARIABLES AND RANDOM TARGETS
train = pd.DataFrame(np.zeros((250,300)))
for i in range(300): train.iloc[:,i] = np.random.normal(0,1,250)
train['target'] = np.random.uniform(0,1,250)
train.loc[ train['target']>0.34, 'target'] = 1.0
train.loc[ train['target']<=0.34, 'target'] = 0.0


# Now we will perform repeated stratified k fold with logistic regression and L1-penalty. The regularization constant `C=0.1` was determined using grid search on our synthetic data.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

oof = np.zeros(len(train))
rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)
for train_index, test_index in rskf.split(train.iloc[:,:-1], train['target']):
    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1,class_weight='balanced')
    clf.fit(train.loc[train_index].iloc[:,:-1],train.loc[train_index]['target'])
    oof[test_index] += clf.predict_proba(train.loc[test_index].iloc[:,:-1])[:,1]
aucTR = round(roc_auc_score(train['target'],oof),5)
print('CV =',aucTR)


# Wow, CV is 0.690 which implies we have a good model but we don't because no model exists. It appears that logistic regression with L1-penalty found patterns within our small synthetic training dataset even though the targets are completely random. How is this possible?? 
#   
# # Experiment 2 : Distribution of CV of useless variables
# The reason that we found patterns within our synthetic dataset is because the sample size is so small. In only 250 samples, by complete chance, there will be correlation between **useless** variables and a randomly generated target. Let's see how well each **useless** variable predicts target by itself.

# In[ ]:


dfTR = pd.DataFrame({'var':np.arange(300),'CV':np.zeros(300)})
for i in range(300):
    logr = LogisticRegression(solver='liblinear').fit(train[[i]],train['target'])
    dfTR.loc[i,'CV'] = roc_auc_score(train['target'],logr.predict_proba(train[[i]])[:,1])
dfTR.sort_values('CV',inplace=True,ascending=False)
dfTR.head()


# In[ ]:


plt.hist(dfTR['CV'],bins=25)
plt.title('Histogram of CV of 300 useless variables')
plt.show()


# Wow, many **useless** variables can predict our random synthetic target with greater than 0.600 AUC. That's crazy. Let's plot synthetic variables 133 and 162 together.

# In[ ]:


plt.figure(figsize=(5,5))
plt.scatter(train[133],train[162],c=train['target'])
plt.plot([-2,2],[-2,2],':k')
plt.title('Among 300 simulated useless variables, we find these two!')
plt.xlabel('synthetic variable 133')
plt.ylabel('synthetic variable 162')
plt.show()


# Whoa, it appears that sythetic variables 133 and 162 can predict the target but this isn't true because all variables are **useless** and the target was generated completely randomly. This is a consequence of pure chance occuring in a small dataset. Weird.

# # Experiment 3 : LB with 300 useless variables
# The training dataset has only 250 observations so we observe many things by chance. The public test dataset has 1975 observations. Let's see how the LB performs on these **useless** variables. 

# In[ ]:


# GENERATE USELESS VARIABLES AND RANDOM TARGETS
public = pd.DataFrame(np.zeros((1975,300)))
for i in range(300): public.iloc[:,i] = np.random.normal(0,1,1975)
public['target'] = np.random.uniform(0,1,1975)
public.loc[ public['target']>0.34, 'target'] = 1.0
public.loc[ public['target']<=0.34, 'target'] = 0.0


# In[ ]:


oof = np.zeros(len(public))
rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)
for train_index, test_index in rskf.split(public.iloc[:,:-1], public['target']):
    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1,class_weight='balanced')
    clf.fit(public.loc[train_index].iloc[:,:-1],public.loc[train_index]['target'])
    oof[test_index] += clf.predict_proba(public.loc[test_index].iloc[:,:-1])[:,1]
aucPU = round(roc_auc_score(public['target'],oof),5)
print('LB =',aucPU)


# Interesting, LB is not fooled! The public dataset is large enough that logistic regression can't find any patterns in 1975 observations when the target is randomly generated.  
#   
# # Experiment 4 : Distribution of LB of useless variables 

# In[ ]:


dfPU = pd.DataFrame({'var':np.arange(300),'LB':np.zeros(300)})
for i in range(300):
    logr = LogisticRegression(solver='liblinear').fit(public[[i]],public['target'])
    dfPU.loc[i,'LB'] = roc_auc_score(public['target'],logr.predict_proba(public[[i]])[:,1])
dfPU.sort_values('LB',inplace=True,ascending=False)
dfPU.head()


# Again, LB is not fooled. No variable can predict target better than 0.54 LB.

# In[ ]:


plt.hist(dfPU['LB'],bins=25)
plt.title('Histogram of LB of 300 useless variables')
plt.show()


# # Conclusion
# The above 4 experiments are very enlightening. We created synthetic data where all 300 variables are **useless** and target is randomly generated. In a sample size of 250 (like training dataset), by complete random chance many variables could predict target with CV as high as 0.640 by themselves. When we applied logistic regression with L1-penaltiy on all 300 **useless** varialbles, it obtained CV 0.690. Thus CV indicated things that did not exist. 
#   
# On the otherhand, in a sample size of 1975 (like the public test dataset), no **useless** variable could predict target with LB greater than 0.540 by itself. And when logistic regression with L1-penalty was performed on all 300 **useless** variables, it could not find any patterns and only reported LB 0.510.  
#   
# This demonstrates that we can trust LB but we must be careful trusting CV. The real data contains variables that are **useful**. Useful variables will be able to predict target by themselves with AUC between 0.54 and 0.64 and above. When **useful** variables are present in the training data, we **cannot** detect them because some **useless** variables also show CV between 0.54 and 0.64 (like **useful** variables). However when these variables are present in the public data, we **can** detect them because **useless** variables have LB below 0.54 while **useful** variables have LB above 0.54

# # Appendix : How many real variables are useful?
# In this appendix, we will attempt to calculate how many real "Don't Overfit II" variables are useful. First let's find the standard deviation of AUC for **useless** variables in a training dataset with 250 observations. We will find a 80% confidence interval.  
#   
# We will calculate AUC with `roc_auc_score(target,variable)`. Therefore if a variable is postively correlated with target, it will obtain a high `AUC > 0.5`. And if a variable is negatively correlated with target, it will obtain a low `AUC < 0.5`. AUC's near `AUC = 0.5` indicate that a variable is not correlated with target and has no individual predictive power.

# In[ ]:


auc = []
target = np.ones(250); target[:90] = 0.0
for i in range(10000):
    useless = np.random.normal(0,1,250)
    auc.append( roc_auc_score(target,useless) )
    #if i%1000==0: print(i)
z = 1.28 # 80% CE, 1.645 is 90% CE
low = round( 0.500 - z * np.std(auc),3)
high = round( 0.500 + z * np.std(auc),3)
print('80% of useless AUC are between',low,'and',high)
plt.hist(auc,bins=100); plt.show()


# Next we will calculate what is the expected number of **useless** variables that fall outside the 80% confidence interval when you have 300 useless variables.

# In[ ]:


outliers = []
target = np.ones(250); target[:90] = 0.0
for i in range(1000):
    ct = 0
    for j in range(300):
        useless = np.random.normal(0,1,250)
        auc = roc_auc_score(target,useless)
        if (auc<low)|(auc>high): ct += 1
    outliers.append(ct)
    #if i%100==0: print(i)
plt.hist(outliers,bins=100); plt.show()
mn = np.mean(outliers); st = np.std(outliers)
lw = round(mn-z*st,1); hg = round(mn+z*st,1)
print('We are 80% confident that between',lw,'and',hg,
      'useless variables have AUC less than',low,'or greater than',high)


# We will now calculate how many real "Don't Overfit II" variables have AUC that falls outside the 80% confidence interval for useless variables.

# In[ ]:


train = pd.read_csv('../input/train.csv')
ct = 0
for i in range(300):
    auc = roc_auc_score(train['target'],train[str(i)])
    if (auc<low)|(auc>high): ct += 1   
print('There are',ct,'real variables with AUC less than',low,'or greater than',high)
a = round(ct-hg,1); b = round(ct-lw,1)
print('Therefore we are 80% confident that between',a,'and',b,
      'real variables are useful with AUC less than',low,'or greater than',high)
print('Additionally there are possible useful real variables with weak AUC between',low,'and',high)


# Note if a variable has low AUC like 0.3, then that variable is negatively correlated with target. Therefore we can reverse the predictions and obtain AUC = 0.7 = 1 - 0.3. Thus low or high AUCs are advantageous. Only AUC's near 0.500 are not advantageous.
