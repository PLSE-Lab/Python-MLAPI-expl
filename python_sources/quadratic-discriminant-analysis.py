#!/usr/bin/env python
# coding: utf-8

# First of all, we are going to use the statistical model called quadratic discriminant analysis  which finds linear combination of features that separates two or more classes. This method is more general than linear discriminant analysis. The following web page is a good reference in order to understand how quadratic discriminant analysis works internally. 
# 
# https://web.stanford.edu/class/stats202/content/lec9.pdf

# In[ ]:


# Loading the packages
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
sns.set(style="ticks", color_codes=True)


# In[ ]:


# Loading the training dataset
df_train = pd.read_csv("../input/train.csv")


# In[ ]:


y = df_train["target"]
# We exclude the target and id columns from the training dataset
df_train.pop("target");
df_train.pop("id")
colnames1 = df_train.columns


# Quadratic discriminant analysis is sensitive to variables that are not standardized, that is why we are going to standardize the explanatory variables by removing the mean and scaling to unit variance. The standard score for the variable X is calculated as follows: 
# 
# $$ z = (X-\mu) / s $$
# 
# Where $\mu$ is the mean and s is the standard deviation. Moreover, one of the hypothesis of linear discriminant analysis is that the predictors must have the same variance.
# 

# In[ ]:


scaler = StandardScaler()
scaler.fit(df_train)
X = scaler.transform(df_train)
df_train = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 


# In this kernel: 
# 
# https://www.kaggle.com/ricardorios/random-forests-don-t-overfit
# 
# We have found the following variables that are related with the target variable: 33, 279, 272, 83, 237, 241, 91, 199, 216, 19, 65, 141, 70, 243, 137, 26, 90. We are going to use these variables to fit the model.

# In[ ]:


random_forest_predictors = ["33", "279", "272", 
                           "83", "237", "241", 
                           "91", "199", "216", 
                           "19", "65", "141", "70", "243", "137", "26", "90"]

predictors = random_forest_predictors
print(predictors)


def five_num(X):
    
    quartiles = np.percentile(X, [25, 50, 75])
    data_min, data_max = X.min(), X.max()
    print("Minimum: {}".format(data_min))
    print("Q1: {}".format(quartiles[0]))
    print("Median: {}".format(quartiles[1]))
    print("Q3: {}".format(quartiles[2]))
    print("Maximum: {}".format(data_max))    


def fit_discriminant(predictors, X):
    
    X = X[predictors]
    X = X.values 
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    
    
    train_auc = []
    valid_auc = []
    
    for train_index, test_index in skf.split(X, y):
        
        model = QuadraticDiscriminantAnalysis()
        model.fit(X[train_index], y[train_index])    
        
        y_train = y[train_index]
        y_test = y[test_index]
    
        y_train_predict = model.predict_proba(X[train_index])
        y_train_predict = y_train_predict[:,1]
        y_test_predict = model.predict_proba(X[test_index], )
        y_test_predict = y_test_predict[:,1]           
        
        train_auc.append(roc_auc_score(y_train, y_train_predict))
        valid_auc.append(roc_auc_score(y_test, y_test_predict))
        
    n_bins = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, tight_layout=True);
    ax1.hist(train_auc, bins=n_bins);
    ax1.set_title("Histogram of AUC training")
    ax2.hist(valid_auc, bins=n_bins);
    ax2.set_title("Histogram of AUC validation")  
    
    print("Five numbers Training AUC\n")
    five_num(np.array(train_auc))
    print("\nFive numbers Valid AUC\n")
    five_num(np.array(valid_auc))

    


# In[ ]:


fit_discriminant(predictors, df_train)


# It seems that this model exhibits overfitting. In order to solve the overfitting problem we are going to start with the two most important variables and then we will continue to increase progressively until we find a good model.

# In[ ]:


predictors = ['33', '279']
fit_discriminant(predictors, df_train)


# In[ ]:


predictors = ['33', '279',  '272']
fit_discriminant(predictors, df_train)


# In[ ]:


predictors = ['33', '279',  '272', '83']
fit_discriminant(predictors, df_train)


# In[ ]:


predictors = ['33', '279',  '272', '83', '237']
fit_discriminant(predictors, df_train)


# In[ ]:


predictors = ['33', '279',  '272', '83', '237', '241']
fit_discriminant(predictors, df_train)


# The best model has the following explanatory variables: 33, 279, and 272.

# In[ ]:


predictors = ['33', '279',  '272']

# We fit the model with the whole training dataset
model = QuadraticDiscriminantAnalysis()
model.fit(df_train[predictors], y)


# Finally, we will send the submission.

# In[ ]:


df_test = pd.read_csv("../input/test.csv")
df_test.pop("id");
X = df_test 
X = scaler.transform(X)
df_test = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 
X = df_test[predictors]
del df_test
y_pred = model.predict_proba(X)
y_pred = y_pred[:,1]


# In[ ]:


# submit prediction
smpsb_df = pd.read_csv("../input/sample_submission.csv")
smpsb_df["target"] = y_pred
smpsb_df.to_csv("quadratic_discrimant_analysis.csv", index=None)


# ## References 
# 
# [1] https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

# In[ ]:




