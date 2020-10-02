#!/usr/bin/env python
# coding: utf-8

# Hey all. I'm really new at this but I'm trying to get my hands dirty. Please jump in and correct me if you see anything that doesn't make sense, or if you have any pointers I'm all ears! 
# 
# I'm taking a lot of ideas from Tak's kernel (https://www.kaggle.com/takaishikawa/experiment02-corr-select-logistic-reg) and trying to expand on that a bit.

# In[ ]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X = np.array(train.drop(['id','target'], axis=1))
y = np.array(train['target'])
X_test = np.array(test.iloc[:,1:])


# Take a look at correlation between each feature and the target

# In[ ]:


# get correlation of each feature with target
corr = train.corr()['target'][2:]
sns.boxplot(corr)


# Lots of very low correlation features. Seems like ditching some might help in avoiding overfitting.
# 
# First I'll make a quick function for repeating cross validation. I realized after I wrote this that there it's built in to sklearn, but this way seems to work too. 

# In[ ]:


from sklearn.model_selection import StratifiedKFold

def repeat_cross_val(model, X, y, n_iters=10, n_folds=5):
    
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True)
    scores = np.zeros([n_iters, n_folds])
    
    for i in range(n_iters):
        for j, (cv_train, cv_test) in enumerate(folds.split(X,y)):
            model.fit(X[cv_train], y[cv_train])
            scores[i,j] = model.score(X[cv_test],y[cv_test])    
    return scores.mean()


# I'm going to define a correlation threshold, this will be a value that used to decide whether a feature is going to be removed from the training data. I'll get the correation with each feature and the target, and then set a value. If the absolute value of correlation is above this, I'll leave that feature in.

# In[ ]:


#set correltaion threshold and filter training data
corr_thresh = 0.1
high_corr = abs(corr)>corr_thresh
X_corr = X[:,high_corr]

X_corr.shape[1]/X.shape[1]


# In[ ]:


# give it a quick test
from sklearn.linear_model import LogisticRegression

lrc = LogisticRegression(penalty='l1', solver='liblinear')
print(repeat_cross_val(model=lrc, X=X_corr, y=y))


# Seems alright.
# 
# Let's see if we can get some idea of what a good correlation threshold will be.

# In[ ]:


# Testing values from 0-0.3
corr_test = np.arange(0, 0.3, 0.01)
cv_score = np.zeros(corr_test.shape[0])

lrc = LogisticRegression(penalty='l1', solver='liblinear')

for i, c in enumerate(corr_test):
    high_corr = abs(corr)>c
    X_corr = X[:,high_corr]
    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=25)

plt.scatter(x=corr_test, y=cv_score)
plt.xlabel('correlation threshold')
plt.ylabel('cv score')
plt.title('Testing correlation threshold')


# It looks like 0.1 - 0.13 is the sweet spot. I'll check how many features that leaves.

# In[ ]:


corr_thresh = 0.11
high_corr = abs(corr)>corr_thresh
X_corr = X[:,high_corr]
X_corr.shape


# Down form 300 to 45.

# In[ ]:


lrc = LogisticRegression(penalty='l1', solver='liblinear')
print('l1: {0:.3f}'.format(repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=250)))
lrc = LogisticRegression(penalty='l2', solver='liblinear')
print('l2: {0:.3f}'.format(repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=250)))


# l2 seems to be just a touch better. Next up is the C value. Smaller C means more regularization (i.e., the algorithm is more willing to let a training point be on the wrong side of the decision boundry), intuitively it seems like this will help with the overfitting problem. Let's take a look...

# In[ ]:


# Testing values from 0-0.3
C_test = np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]) 
cv_score = np.zeros(C_test.shape[0])
for i, C in enumerate(C_test):
    lrc = LogisticRegression(penalty='l2', solver='liblinear', C=C)
    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=50)
plt.scatter(x=C_test, y=cv_score)
plt.xlabel('C-value')
plt.ylabel('cv score')
plt.title('Testing C-value')
plt.xscale('log')
plt.xlim((0.00000001,10000))


# It looks like smaller C values are the way to go! (default is 1.0) I'll zome in on the peak.

# In[ ]:


get_ipython().run_cell_magic('time', '', "C_test = np.logspace(-4.0, 0, 60)\ncv_score = np.zeros(C_test.shape[0])\nfor i, C in enumerate(C_test):\n    lrc = LogisticRegression(penalty='l2', solver='liblinear', C=C)\n    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=50)\nplt.scatter(x=C_test, y=cv_score)\nplt.xlabel('C-value')\nplt.ylabel('cv score')\nplt.title('Testing C-value')\nplt.xscale('log')\nplt.xlim((10**-4.2,1))")


# Looks like something around 0.05 will be a good option. I'm going to check the corr_threshold again with this C value. 

# In[ ]:


# Testing values from 0-0.3
corr_test = np.arange(0, 0.3, 0.01)
cv_score = np.zeros(corr_test.shape[0])
lrc = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
for i, c in enumerate(corr_test):
    high_corr = abs(corr)>c
    X_corr = X[:,high_corr]
    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=25)
plt.scatter(x=corr_test, y=cv_score)
plt.xlabel('correlation threshold')
plt.ylabel('cv score')
plt.title('Testing correlation threshold')


# Looks about the same. Lets see how it does.

# In[ ]:


# reset X_corr
corr_thresh = 0.11
high_corr = abs(corr)>corr_thresh
X_corr = X[:,high_corr]

lrc = LogisticRegression(penalty='l2', solver='liblinear', C=0.05)
lrc.fit(X_corr, y)


# In[ ]:


predict = lrc.predict(X_test[:,high_corr]) # this got a .704
predict_prob = lrc.predict_proba(X_test[:,high_corr]) # wow! this got a 0.786
print(predict[0], predict_prob[:,1])


# Using the probabilities instead of binary predictions got a much better score (0.786 vs 0.704). This surprised me, time to learn more about AUCROC.

# In[ ]:


sub = pd.DataFrame({
    'id': test['id'],
    'target': predict_prob[:,1]
})
print(sub.head())
print(pd.read_csv('../input/sample_submission.csv').head())


# In[ ]:


sub.to_csv('submission.csv', index=False)

