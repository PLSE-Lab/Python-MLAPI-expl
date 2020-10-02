#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# import packages
print(__doc__)
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc


# In[ ]:


# load the data
dataframe = pd.read_csv("../input/creditcard.csv")


# In[ ]:


# get column names
colNames = dataframe.columns.values
colNames


# In[ ]:


# get dataframe dimensions
print ("Dimension of dataset:", dataframe.shape)


# In[ ]:


# get attribute summaries
print(dataframe.describe())


# In[ ]:


# get class distribution
print ("Normal transaction:", dataframe['Class'][dataframe['Class'] == 0].count()) #class = 0
print ("Fraudulent transaction:", dataframe['Class'][dataframe['Class'] == 1].count()) #class = 1


# In[ ]:


# separate classes into different datasets
class0 = dataframe.query('Class == 0')
class1 = dataframe.query('Class == 1')

# randomize the datasets
class0 = class0.sample(frac=1)
class1 = class1.sample(frac=1)


# ## Undersampling to deal with class imbalance
# The examples of the majority class, in this case the normal transactions drastically outnumber the incidences of fraud in our dataset. One of the strategies employed in the data science community is to delete instances from the over-represented class to improve the learning function. Here, we selected 6000 instances of the normal class from the original 284315 records.

# In[ ]:


# undersample majority class due to class imbalance before training - train
class0train = class0.iloc[0:6000]
class1train = class1

# combine subset of different classes into one balaced dataframe
train = class0train.append(class1train, ignore_index=True).values


# In[ ]:


# split data into X and y
X = train[:,0:30].astype(float)
Y = train[:,30]


# ## The Learning Algorithm: XGBoost
# Extreme Gradient Boosting is also known as XGBoost. This model is preferred due to its execution speed and learning performance.

# In[ ]:


# XGBoost CV model
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)

# use area under the precision-recall curve to show classification accuracy
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring = scoring)
print( "AUC: %.3f (%.3f)" % (results.mean(), results.std()) )


# ## Plot the Result

# In[ ]:


# change size of Matplotlib plot
fig_size = plt.rcParams["figure.figsize"] # Get current size

old_fig_params = fig_size
# new figure parameters
fig_size[0] = 12
fig_size[1] = 9
   
plt.rcParams["figure.figsize"] = fig_size # set new size


# In[ ]:


# plot roc-curve
# code adapted from http://scikit-learn.org
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X, Y), colors):
    probas_ = model.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ## Remarks
# From the above results our algorithm achieved auc-roc (i.e. area under the precision-recall curve) score of 0.979. The auc-roc curve in insensitive to class imbalanace and hence is the preferred evaluation metric for estimating the performance of our learning function.
# 
# ### Further Remarks
# Several other techniques that can be explored/ benchmarked:  
# - Visualization to understand the transaction trends over time, more ideas can be gotten to understand more about fraudulent transactions  
# - Research deep learning techniques to this problem such as Reccurent Neural Networks using the time component for sequence-to-sequence learning  
# - Consider other learning options such as anomaly detection or change detection
