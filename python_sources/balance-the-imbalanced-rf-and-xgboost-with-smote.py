#!/usr/bin/env python
# coding: utf-8

# # Fraud analysis: 
# ### Random Forest, XGBoost, OneClassSVM, Multivariate GMM and SMOTE, all in one cage against an imbalanced dataset.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## 1. Supervised learning tests

# I will now test a series of different machine learning models (no Neural Networks!) to see which one performs better, with some optimization here and there. 

# ### 1.1 Data import, quick view and functions

# In[ ]:


df = pd.read_csv('../input/creditcard.csv')


# In[ ]:


df.head()


# I wonder if it should be treated as a data series rather than a table... 

# In[ ]:


df.describe()


# Check for NaNs

# In[ ]:


df.isnull().sum()


# WOW! Seriously, no NaNs? 
# 
# Ok, let's check for the classes distributions

# In[ ]:


print('Fraud \n',df.Time[df.Class==1].describe(),'\n',
      '\n Non-Fraud \n',df.Time[df.Class==0].describe())


# Imbalanced dataset. Might be worth to work on upsampling/downsampling of the data, but I will try without it for the moment and hope I get good results. Now let's check which variable is more correlated with the fraudulent activities. 

# In[ ]:


plt.figure(figsize=(12,30*4))
import matplotlib.gridspec as gridspec
features = df.iloc[:,0:30].columns
gs = gridspec.GridSpec(30, 1)
for i, feature in enumerate(df[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[feature][df.Class == 1], bins=50)
    sns.distplot(df[feature][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('Feature: ' + str(feature))
plt.show()


# Remove the features that do not have significantly different distributions between the two classes (i.e. will not contribute to our model).

# In[ ]:


df2 = df.drop(['V15','V20','V22','V23','V25','V28', 'Time', 'Amount'], axis=1)


# Data preparation and general functions for plots

# In[ ]:


from sklearn.metrics import confusion_matrix
def plot_cm(classifier, predictions):
    cm = confusion_matrix(y_test, predictions)
    
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap='RdBu')
    classNames = ['Normal','Fraud']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                     horizontalalignment='center', color='White')
    
    plt.show()
        
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2*recall*precision/(recall+precision)

    print('Recall={0:0.3f}'.format(recall),'\nPrecision={0:0.3f}'.format(precision))
    print('F1={0:0.3f}'.format(F1))


# In[ ]:


from sklearn.metrics import average_precision_score, precision_recall_curve
def plot_aucprc(classifier, scores):
    precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=0)
    average_precision = average_precision_score(y_test, scores)

    print('Average precision-recall score: {0:0.3f}'.format(
          average_precision))

    plt.plot(recall, precision, label='area = %0.3f' % average_precision, color="green")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="best")
    plt.show()


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# ## 1.2. Test a Random Forest model

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


pre = RandomForestClassifier(n_jobs=-1, random_state = 42,
                             max_features= 'sqrt', 
                             criterion = 'entropy')
pre.fit(X_train, y_train)

#Make predictions
y_pred = pre.predict(X_test)
try:
    scores = pre.decision_function(X_test)
except:
    scores = pre.predict_proba(X_test)[:,1]

#Make plots
plot_cm(pre, y_pred)
plot_aucprc(pre, scores)


# The F-1 score is not that bad! Let's try to fine tune some parameters and see if we can improve that.
# 
# *Note: since it takes too long for the Kaggle kernel, I executed it on my computer and here I am just showing the results of the GridSearchCV*

# In[ ]:


#from sklearn.model_selection import GridSearchCV
#param_grid = { 
#    'n_estimators': [10, 500],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'min_samples_leaf' : [len(X)//10000, len(X)//28000, 
#                          len(X)//50000, len(X)//100000]
#}

#CV_rfc = GridSearchCV(estimator=pre, 
#                      param_grid=param_grid, 
#                      scoring = 'f1',
#                      cv=10, 
#                      n_jobs=10,
#                      verbose=2,
#                      pre_dispatch='2*n_jobs',
#                      refit=False)
#CV_rfc.fit(X_train, y_train)

#CV_rfc.best_params_


# In[ ]:


#rfc = RandomForestClassifier(n_jobs=-1, random_state = 42,
#                             n_estimators=CV_rfc.best_params_['n_estimators'], 
#                             min_samples_leaf=CV_rfc.best_params_['min_samples_leaf'], 
#                             max_features= CV_rfc.best_params_['max_features'])

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=2,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            n_estimators=500, n_jobs=-1, oob_score=False, random_state=42,
#            verbose=0, warm_start=False)


rfc = RandomForestClassifier(n_jobs=-1, random_state = 42,
                             n_estimators=500, 
                             max_features='auto',
                             min_samples_leaf=2,
                             criterion = 'entropy')

rfc.fit(X_train, y_train)


# In[ ]:


#Make predictions
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)


# Yass! Nice increase! Now let's see if I can get any better with the most popular boosting algorithm...

# ## 1.3. Test a XGBoost model

# In[ ]:


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 42, n_jobs = -1)
xgb.fit(X_train, y_train)


# In[ ]:


#Make predictions
y_pred = xgb.predict(X_test)
try:
    scores = xgb.decision_function(X_test)
except:
    scores = xgb.predict_proba(X_test)[:,1]
#Make plots
y_pred = xgb.predict(X_test)
plot_cm(xgb, y_pred)
plot_aucprc(xgb, scores)


# Ok, now we're talking. Any chances of getting better with optimization?

# In[ ]:


fraud_ratio=y_train.value_counts()[1]/y_train.value_counts()[0]
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [1,3,5], 
             'min_child_weight': [1,3,5], 
             'n_estimators': [100,200,500,1000], 
             'scale_pos_weight': [1, 0.1, 0.01, fraud_ratio]}


# In[ ]:


#CV_GBM = GridSearchCV(estimator = xgb, 
#                      param_grid = param_grid,
#                      scoring = 'f1', 
#                      cv = 10, 
#                      n_jobs = -1,
#                      refit = True)

#CV_GBM.fit(X_train, y_train)

#CV_GBM.best_params_


# In[ ]:


#optimized_GBM = XGBClassifier(n_jobs=-1, random_state = 42,
#                             n_estimators=CV_GBM.best_params_['n_estimators'], 
#                             max_depth=CV_GBM.best_params_['max_depth'],
#                             min_child_weight=CV_GBM.best_params_['min_child_weight'],
#                             criterion = 'entropy')
optimized_GBM = XGBClassifier(n_jobs=-1, random_state = 42,
                             n_estimators=100, 
                             max_depth=1,
                             min_child_weight=1,
                             criterion = 'entropy',
                             scale_pos_weight=fraud_ratio)
optimized_GBM.fit(X_train, y_train)


# In[ ]:


#Make predictions
y_pred = optimized_GBM.predict(X_test)
try:
    scores = optimized_GBM.decision_function(X_test)
except:
    scores = optimized_GBM.predict_proba(X_test)[:,1]
    
#Make plots
plot_cm(optimized_GBM, y_pred)
plot_aucprc(optimized_GBM, scores)


# ## 1.4. OneClassSVM?

# This should be, according to Scikit-learn tutorials, the best algorithm to infer anomalies in an imbalanced dataset. Let's give it a try.

# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import OneClassSVM
classifier = OneClassSVM(kernel="rbf", random_state = 42)
classifier.fit(X_train, y_train)


# In[ ]:


#Make predictions
y_pred = classifier.predict(X_test)
y_pred = np.array([y==-1 for y in y_pred])

try:
    scores = classifier.decision_function(X_test)
except:
    scores = classifier.predict_proba(X_test)[:,1]

#Make plots
plot_cm(classifier, y_pred)
plot_aucprc(classifier, scores)


# I don't really like these results, honestly... 

# ## 1.5. Test a (Multivariate) GMM module

# Per Ng's lessons, we should divide the dataset into a training set with ONLY normal transactions, and validation and test set containing all the fraudulent transations. I will try at first without crossvalidation.

# In[ ]:


df3 = df2#.sample(frac = 0.1, random_state=42)
train = df3[df3.Class==0].sample(frac=0.75, random_state = 42)

X_train = train.iloc[:,:-1]
y_train = train.iloc[:,-1]

X_test = df3.loc[~df3.index.isin(X_train.index)].iloc[:,:-1]#.sample(frac=.50, random_state = 42)
y_test = df3.loc[~df3.index.isin(y_train.index)].iloc[:,-1]#.sample(frac=.50, random_state = 42)

#X_cval = df3.loc[~df3.index.isin(X_test.index)& ~df3.index.isin(X_train.index)].iloc[:,:-1]
#y_cval = df3.loc[~df3.index.isin(y_test.index)& ~df3.index.isin(X_train.index)].iloc[:,-1]


# In[ ]:


print('df3', df3.shape,'\n',
      'train',train.shape,'\n',
      'X_train',X_train.shape,'\n',
      'y_train',y_train.shape,'\n',
      'X_test',X_test.shape,'\n',
      'y_test',y_test.shape,'\n', 
      #'X_val',X_cval.shape,'\n',
      #'y_val',y_cval.shape,'\n'
     )


# In[ ]:


df3.shape[0] == train.shape[0] + X_test.shape[0]


# In[ ]:


def covariance_matrix(X):
    X = X.values
    m, n = X.shape 
    tmp_mat = np.zeros((n, n))
    mu = X.mean(axis=0)
    for i in range(m):
        tmp_mat += np.outer(X[i] - mu, X[i] - mu)
    return tmp_mat / m


# In[ ]:


cov_mat = covariance_matrix(X_train)


# In[ ]:


cov_mat_inv = np.linalg.pinv(cov_mat)
cov_mat_det = np.linalg.det(cov_mat)
def multi_gauss(x):
    n = len(cov_mat)
    return (np.exp(-0.5 * np.dot(x, np.dot(cov_mat_inv, x.T))) 
            / (2. * np.pi)**(n/2.) 
            / np.sqrt(cov_mat_det))


# In[ ]:


eps = min([multi_gauss(x) for x in X_train.values])
predictions = np.array([multi_gauss(x) <= eps for x in X_test.values])
y_test = np.array(y_test, dtype=bool)


# In[ ]:


cm = confusion_matrix(y_test, predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap='RdBu')
classNames = ['Normal','Fraud']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                 horizontalalignment='center', color='White')

plt.show()

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)
print("F1=",F1)


# Adapting NG's code from MatLab

# In[ ]:


from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
    
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return p.pdf(dataset)

def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000;
    epsilons = np.arange(min(probs),max(probs),stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(gt, predictions, average = "binary")
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon
    return best_f1, best_epsilon


# In[ ]:


#fit the model
mu, sigma = estimateGaussian(X_train)
p = multivariateGaussian(X_train,mu,sigma)

p_cv = multivariateGaussian(X_test,mu,sigma)
fscore, ep = selectThresholdByCV(p_cv,y_test)
outliers = np.asarray(np.where(p < ep))


# In[ ]:


predictions = np.array([p_cv <= ep]).transpose()
y_test = np.array(y_test, dtype=bool)

cm = confusion_matrix(y_test, predictions)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap='RdBu')
classNames = ['Normal','Fraud']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                 horizontalalignment='center', color='White')

plt.show()

tn, fp, fn, tp = cm.ravel()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)
print("F1=",F1)


# ## 2.  Working on imbalanced dataset: upsampling of the underrepresented class

# ## 2.1 BalancedBaggingClassifier

# The package imbalanced-learn (not yet part of the official scikitlearn) contains an imbalanced classifier which should be able, using a bagging method, to increase our prediction capabilities without resampling the dataset.

# First, let's reset our original dataset, without the unwanted features.

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = df2.iloc[:,:-1]
y = df2.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

from imblearn.metrics import classification_report_imbalanced

bagging = BaggingClassifier(random_state=0)
balanced_bagging = BalancedBaggingClassifier(random_state=0)

bagging.fit(X_train, y_train)
balanced_bagging.fit(X_train, y_train)

#Make predictions
print('Classification of original dataset with Bagging (scikit-learn)')
y_pred = bagging.predict(X_test)
try:
    scores = bagging.decision_function(X_test)
except:
    scores = bagging.predict_proba(X_test)[:,1]

#Make plots
plot_cm(bagging, y_pred)
plot_aucprc(bagging, scores)

#Make predictions
print('Classification of original dataset with BalancedBagging (imbalanced-learn)')
y_pred = balanced_bagging.predict(X_test)
try:
    scores = balanced_bagging.decision_function(X_test)
except:
    scores = balanced_bagging.predict_proba(X_test)[:,1]

#Make plots
plot_cm(balanced_bagging, y_pred)
plot_aucprc(balanced_bagging, scores)


# So, the imbalanced-learn packages without resampling of the data allows us to have higher true positive numbers, but lowers the true negative, meaning that it's just mistakenly saying there are more frauds than in reality. Not good.  
# Let's try again, with SMOTE, which will produce synthetical samples of the under-represented class

# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25, random_state = 42)

#fit the best models so far
xgb.fit(X_train, y_train)
rfc.fit(X_train, y_train)

#Make predictions
print('Classification of SMOTE-resampled dataset with XGboost')
y_pred = xgb.predict(X_test)
try:
    scores = xgb.decision_function(X_test)
except:
    scores = xgb.predict_proba(X_test)[:,1]
#Make plots
y_pred = xgb.predict(X_test)
plot_cm(xgb, y_pred)
plot_aucprc(xgb, scores)

#Make predictions
print('Classification of SMOTE-resampled dataset with optimized RF')
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)


# WOW! So, if we now use this new RF classifier (which parameters were optimized on the resampled dataset) on the ORIGINAL dataset, we'll get our perfect fraud analysis prediction.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Make predictions
y_pred = rfc.predict(X_test)
try:
    scores = rfc.decision_function(X_test)
except:
    scores = rfc.predict_proba(X_test)[:,1]

#Make plots
plot_cm(rfc, y_pred)
plot_aucprc(rfc, scores)


# "Pretty cool huh?"

# In[ ]:




