#!/usr/bin/env python
# coding: utf-8

# ## Approach
# 
# 1. The data is strongly imbalanced, so the first step will be to undersample the data and obtain both classes 1:1. 
# 2. Based on this, you will see a Logistic Regression with some Regularization, a Random Forrest and a SVM with some parameter tuning.
# 3. Finally, I plan to compute and compare different evaluation metrics to see if our models are performing good on the test set. 
# 4. Open for suggestions, post comments what you would like to see in the future ...
# 
# **If you learn anything from this, feel free to upvote and/or leave a comment. ;)**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the dataset

# In[ ]:


data = pd.read_csv("../input/biddings.csv")
print(data.shape)
data.head()


# We have 1 million records with 88 principle components that can be used to predict our target variable *convert*.

# ## Checking the target classes

# We have 1 million records with 88 principle components that can be used to predict our target variable *convert*.

# In[ ]:


count_classes = pd.value_counts(data['convert'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("bidding conversion histogram")
plt.xlabel("Conversion")
plt.ylabel("Count")


# The data is heavily unbalanced. Less than 1% belongs to converted users.

# ### Split 80:20
# The data is already shuffled (so we can easily select train and test set with indexes). 

# In[ ]:


#advantage of being the creator of the dataset I already shuffled the
train = data[:800000]
test = data[800000:]


# # 1. Undersampe the data

# In[ ]:


def undersample(data, ratio=1):
    conv = data[data.convert == 1]
    oth = data[data.convert == 0].sample(n=ratio*len(conv))
    return pd.concat([conv, oth]).sample(frac=1) #shuffle data

ustrain = undersample(train)

y = ustrain.convert
X = ustrain.drop('convert', axis=1)

print("Remaining rows", len(ustrain))


# Just give this a second thought. We reduced our dataset from 0.8 million records to roughly 3000, so we don't use 99 percent of the original dataset anymore. This seems like we would introduce a lot of bias, so to generalise we need to do this step multiple times and obtain many different undersampled datasets to train our model on. 

# # 2.  Fit the Models
# A big thumbs up to the sklean developers, thanks to good implementation and documentation it's pretty straightforward to apply all sort of models, cross validate them and whatnot. 

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import auc,roc_curve


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
C_s = np.logspace(-10, 1, 11)
scores = list()
scores_std = list()
lr = LogisticRegression(penalty = 'l1')

for C in C_s:
    lr.C = C
    this_scores = cross_val_score(lr, X, y, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    
lr_results = pd.DataFrame({'score':scores, 'C':C_s}) 
lr_results


# #### Random Forrest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
msl_s = [1,2,4,8,16,32,64,128,256]
scores = list()
scores_std = list()
rf = RandomForestClassifier(n_estimators = 15)

for msl in msl_s:
    rf.min_samples_leaf = msl
    this_scores = cross_val_score(rf, X, y, cv=4,scoring='roc_auc')
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    
rf_results = pd.DataFrame({'score':scores, 'Minimum samples leaf': msl_s}) 
rf_results


# ## SVM

# In[ ]:


from sklearn import svm
C_s = np.logspace(-10, 1, 11)
scores = list()
scores_std = list()
svc = svm.SVC(kernel='linear', probability=True)

for C in C_s:
    svc.C = C
    this_scores = cross_val_score(svc, X, y, cv=4,scoring='roc_auc', n_jobs=-1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    
svm_results = pd.DataFrame({'score':scores, 'C':C_s})    
svm_results


# # 3. Evaluate the Fit 
# Now we are going to test how good are these models performing with regard to the imbalanced test set, we reserved in the beginning. We are going to use the models with the best parameters (selected with sklearn cross-validation)

# In[ ]:


#not really elegant, but fits the pupose
y_preds = []

lr.C = lr_results.loc[lr_results['score'].idxmax()]['C']
y_preds.append(lr.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])

rf.min_samples_leaf = int(rf_results.loc[rf_results['score'].idxmax()]['Minimum samples leaf'])
y_preds.append(rf.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])

svc.C = svm_results.loc[svm_results['score'].idxmax()]['C']
y_preds.append(svc.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])


# In[ ]:


model = ['LogR','RanF','SVM']
colors = ['b','r','g']

for i in range(0,3):
    fpr, tpr, thresholds = roc_curve(test.convert,y_preds[i])
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, 'b',label='%s AUC = %0.2f'% (model[i] ,roc_auc),  color=colors[i], linestyle='--')
    plt.legend(loc='lower right')
    
plt.title('Receiver Operating Characteristic')
plt.plot([-0.1,1.1],[-0.1,1.1],color='gray', linestyle=':')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# As we can see Logistic Regression (with regularization) performed best in this case. However, I bet with some better parameter tuning Random Forrest and SVM could probably catch up.
