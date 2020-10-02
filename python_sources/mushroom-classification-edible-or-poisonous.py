#!/usr/bin/env python
# coding: utf-8

# **Problem Statement:**
# 
# We have been given a data file on mushroom species and we need to classify them as **e**dible or **p**oisonous. 
# 
# Data file description:[](http://)
# 
# **Target variable**: 
# * ****class: e=edible, p=poisonous 
# 
# **Features:**
# * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# * bruises: bruises=t,no=f
# * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# * gill-attachment: attached=a,descending=d,free=f,notched=n
# * gill-spacing: close=c,crowded=w,distant=d
# * gill-size: broad=b,narrow=n
# * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# * stalk-shape: enlarging=e,tapering=t
# * stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * veil-type: partial=p,universal=u
# * veil-color: brown=n,orange=o,white=w,yellow=y
# * ring-number: none=n,one=o,two=t
# * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# * population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# * habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# In[ ]:


# Python 3 environment

# imports and tweaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# tweaks for Numpy & Pandas
pd.set_option('display.notebook_repr_html',True)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1024)
# force all numoy & pandas floating point output to 3 decimal places
float_formatter = lambda x: '%.3f' % x
np.set_printoptions(formatter={'float_kind':float_formatter})
pd.set_option('display.float_format', float_formatter)
# force Numpy to display very small floats using floating point notation
np.set_printoptions(threshold=np.inf)
# force GLOBAL floating point output to 3 decimal places
get_ipython().run_line_magic('precision', '3')

# tweaks for plotting libraries (Matplotlib & Seaborn) [recommended]
plt.style.use('seaborn-muted')
sns.set_context(context='notebook',font_scale=1.0)
sns.set_style('whitegrid')

seed = 42 #sum(map(ord, 'Kaggle - Pima Indians Diabetes Analysis'))
np.random.seed(seed)


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Loading Data & Preliminary Analysis

# In[ ]:


data_file = '../input/mushrooms.csv'
data = pd.read_csv(data_file)
data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


# how many unique output classes?
np.unique(data['class'])


# In[ ]:


# save names of features & outcome columns of dataset
outcome = data.columns.values[0]    # name of target (= 'class')
features = data.columns.values[1:]  # names of 21 features
outcome, features


# In[ ]:


# lets check how many unique values each feature has
for feature in features:
    print('\'%s\' feature has %d unique values' % (feature, len(np.unique(data[feature]))))


# In[ ]:


# distribution of target 
f, ax = plt.subplots(figsize=(6,4))
_ = sns.countplot(x='class', data=data, ax=ax)


# In[ ]:


# let us view countplot of each feature (we have 22 features)
num_features = len(features)
plots_per_row = 3
num_rows = int(num_features/plots_per_row) + 1
col_id = 0

with sns.axes_style('ticks'):
    for row in range(num_rows):
        f, ax = plt.subplots(nrows=1, ncols=plots_per_row, sharey=False, figsize=(12,2))
        for col in range(plots_per_row):
            sns.countplot(x=data[features[col_id]], data=data, ax=ax[col])
            col_id += 1
            if col_id >= num_features:
                break
plt.show()
plt.close()


# **Observations:**
# * We have dataset with 8124 samples.
# * There are 22 features and the 'class' column is the target value
#     > _Action:_ We have adequate number of records, so we won't have to resort to PCA techniques for feature compression.
# * All features as well as the target value are categorical variables!
# * The target value (class) has 2 possible outcomes 'e' and 'p' - this is a binary classification problem.
# * Each feature has a differnt number of unique values - from 1 (veil-type) to 12 (gill-color). 
#     > _Action:_ We will have to use `LabelEncoding` on all the columns to convert from categorical values to numeric values. We **won't** use `One-Hot-Encoding` because all features will then be on 0-1 scale and most classifiers will give accuracy_score = 1. We will use a classifier like `DecisionTreeClassifier()` which can work well with LabelEncoded data.
# * The _target class is not unbalanced_ - we have approximatly equal proportions of each class 'p' and 'e'
#     > _Action:_ We don't have to use starification when splitting data into train/test sets.

# # Data Pre-processing

# In[ ]:


from sklearn.preprocessing import LabelEncoder
# apply label encoding to all the columns 
data2 = data.copy()
for col in data2.columns.values:
    data2[col] = LabelEncoder().fit_transform(data2[col])

data2.head() # we can now use data2


# We have not been provided a _separate_ testing file to do an _independent_ check of the classifier. To _create_ such a file, I will keep aside 10% of the data records as an _independent test set_. This is different from `X_test` and `y_test`. The advantage of this approach is I can 'simulate' me feeding an independent file to the classifier, with the advantage that I have the expected result also available for cross-checking. We will call this 10% of records the *testing_set*

# In[ ]:


# select 10% of number of data records
set_aside_count = int(0.10 * len(data2)) 
set_aside_ids = np.random.choice(data2.index, set_aside_count, replace=False)
testing_set = data2.loc[set_aside_ids]   # independent testing data
data_set = data2[~data2.index.isin(set_aside_ids)]
len(data2), set_aside_count, len(set_aside_ids), len(testing_set), len(data_set)


# In[ ]:


# split the data_set into train & test sets
from sklearn.model_selection import train_test_split

# we will use 80:20 split
train_set, test_set =   train_test_split(data_set, test_size=0.20, random_state=0)

X_train = train_set[features] 
X_test = test_set[features]  
y_train = train_set[outcome] 
y_test = test_set[outcome] 
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## Utility functions

# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix


# In[ ]:


# some utility functions
def do_kfold_cv(classifier, X_train, y_train, n_splits=10, scoring='roc_auc'):
    """ do a k-fold cross validation run on classifier & training data
      and return cross-val scores """   
    kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(classifier, X_train, y_train, scoring=scoring, cv=kfolds)
    return cv_scores

def test_classifier(clf_tuple, X_train, y_train, X_test, y_test, scoring='roc_auc', verbose=2):
    """ run a k-fold test, fit model to training data & report
        scores for training & test data """
    # extract classifier instance & name
    classifier, classifier_name = clf_tuple
   
    if verbose > 0:
        print('Testing classifier %s...' % classifier_name)
    
    classifier.fit(X_train, y_train)
    
    # accuracy scores, against test data
    acc_score = classifier.score(X_test, y_test)
    
    # k-fold cross-validation scores
    cv_scores = do_kfold_cv(classifier, X_train, y_train, scoring=scoring)

    # roc-auc score
    y_pred_proba_train = classifier.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, y_pred_proba_train)

    if verbose > 1:   
        print('   - Cross-val score       : Mean - %.3f Std - %.3f Min - %.3f Max - %.3f' %                   (np.mean(cv_scores), np.std(cv_scores), np.min(cv_scores), np.max(cv_scores)))
        print('   - Accuracy score (test) : %.3f' % (acc_score))
        print('   - AUC score (test)      : %.3f' % (auc_score))
              
    return cv_scores, acc_score, auc_score


# ## Create _Decision Tree Classifier_ & run on X_train and check on X_test

# In[ ]:


# create & run the DecisionTreeClassifier on the data
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=seed, max_depth=6)
# following line will print the metrics
_ = test_classifier((clf_dt,'Decision Tree'), X_train, y_train, X_test, y_test, verbose=2)


# **NOTE**:
# * We got a _test accuracy_ of 99.5%
# * We got an AUC score on test (X_test, y_test) data of 99.9% (nearly perfect classifer?)

# In[ ]:


# lets check the confusion matrix - I am using pandas crosstab function to create a better view
pd.crosstab(y_test.ravel(), clf_dt.predict(X_test), rownames=['Actual'], 
            colnames=['Predicted->'], margins=False)


# ** Results:**
# * We have just 8 incorrect classifications on X_test, y_test - all *false negatives*
# 
# Next, let us run our classifier against the *set-aside* data file to check how the classifier behaves.

# In[ ]:


X_testing = testing_set[features]
y_testing = testing_set[outcome]


# In[ ]:


pd.crosstab(y_testing.ravel(), clf_dt.predict(X_testing), 
            rownames=['Actual'], colnames=['Predicted->'], margins=False)


# We have just 2 mis-classifications here.

# In[ ]:


# accuracy from this is calculates as
(433. + 377.) / (433. + 0. + 2. + 377.)


# In[ ]:




