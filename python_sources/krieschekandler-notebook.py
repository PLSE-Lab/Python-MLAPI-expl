#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project

# ### Contents
# 1. [Preprocessing](#preprocessing)
# 1. [Classifiers](#classifiers)
# 1. [Cross-Validation (parameter tuning)](#cross-validation)
# 4. [Classifier Test](#test)
# 5. [Submission](#submission)

# In[ ]:


import pandas as pd
import os
from sklearn import preprocessing, svm
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_regression
import sklearn
import seaborn as sb
import matplotlib.pyplot as plt


# ### Import Data

# In[ ]:


# Read CSV file
filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)

# Split dataset into x and y
fea_col = data_train.columns[2:]
data_Y = data_train['target']
data_X = data_train[fea_col]


# ### Preprocessing <a name="preprocessing"></a>

# #### Split Data

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.2)


# #### Standard Scaling

# In[ ]:


scaler = StandardScaler()
scaler = scaler.fit(x_train,y_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)


# #### Feature Selection

# In[ ]:


selector = SelectKBest(f_classif, k=30)
selector = selector.fit(x_train, y_train)
x_train = selector.transform(x_train)
x_val = selector.transform(x_val)


# #### RFE Feature Selection

# In[ ]:


rfe = RFE(estimator=clf, n_features_to_select=30)
rfe = rfe.fit(x_train,y_train)
x_train = rfe.transform(x_train)
x_val = rfe.transform(x_val)


# #### Replace missing data

# In[ ]:


imp = SimpleImputer(missing_values=-1, strategy='median')
x_train = imp.fit_transform(x_train,y_train)


# #### Undersample

# In[ ]:


rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_resample(x_train,y_train)


# #### Oversample

# In[ ]:


ros = RandomOverSampler(random_state=0)
x_train, y_train = ros.fit_resample(x_train,y_train)


# In[ ]:


smote = SMOTE(sampling_strategy='auto', random_state=0)
x_train, y_train = smote.fit_resample(x_train,y_train)


# In[ ]:


adasyn = ADASYN(random_state=0)
x_train, y_train = adasyn.fit_resample(x_train,y_train)


# #### MinMax Scale

# In[ ]:


scaler = preprocessing.MinMaxScaler()
data_X = scaler.fit_transform(data_X, data_Y)


# #### Normalize

# In[ ]:


x_train = preprocessing.normalize(x_train, norm='l2')


# ### Classifiers <a name="classifiers"></a>

# #### Logistic Regression

# In[ ]:


clf = LogisticRegression(C=1,  max_iter=5000,  class_weight=None)


# #### Stochastic Gradient Descent

# In[ ]:


clf = SGDClassifier(loss="modified_huber",penalty="l2",max_iter=5000, class_weight='balanced')


# #### Multi-Layer Perceptron

# In[ ]:


clf = MLPClassifier(random_state=1, max_iter=5000, hidden_layer_sizes=(100,100,100))


# #### Support Vector Machine

# In[ ]:


clf = svm.SVR(kernel='linear')
# too slow


# #### Support Vector Classifier

# In[ ]:


clf = sklearn.svm.SVC(class_weight='balanced')


# #### Random Forest Classifier
# This classifier was used to generate the final results.

# In[ ]:


clf = RandomForestClassifier(class_weight={0:1,1:25}, max_depth=15, min_samples_leaf=40, max_features=10, n_jobs=-1, n_estimators=500)


# ### Cross-Validation (parameter tuning) <a name="cross-validation"></a>

# #### SGD Tuning

# In[ ]:


clf = SGDClassifier(loss="log",penalty="l1",max_iter=5000, class_weight='balanced')
parameters = {'loss':('log','modified_huber', 'hinge'), 'penalty':('l1', 'l2', 'elasticnet')}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)


# #### LR Tuning

# In[ ]:


clf = LogisticRegression(max_iter=5000)

parameters = {'C':(1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6, 1e9), 'class_weight':('balanced',None)}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)


# #### SVC Tuning

# In[ ]:


def validate_svc(kernel):
    clf = sklearn.svm.SVC(kernel=kernel, class_weight='balanced')
    print(np.mean(cross_val_score(clf,data_X,data_Y, scoring='f1'))*100)
    
for kernel in ['poly', 'rbf', 'sigmoid', 'precomputed']:
    print("Kernel: ", kernel)
    validate_svc(kernel)
    print("*************")


# #### Forest Tuning

# In[ ]:


clf = RandomForestClassifier(class_weight={0:1,1:25}, max_depth=15, min_samples_leaf=40, max_features=10, n_jobs=-1, n_estimators=500)


# ##### Max Features

# In[ ]:


parameters = {'max_features':(5,10,13,15,17,20,25)}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)
plot_results('max_features')


# ##### N Estimators

# In[ ]:


parameters = {'n_estimators':(10,100,250,500,1000)}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)
plot_results('n_estimators')


# ##### Max Depth

# In[ ]:


parameters = {'max_depth':(1,5,10,11,13,15,17,19,20)}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)
plot_results('max_depth')


# ##### Class Weight

# In[ ]:


parameters = {'class_weight':('balanced',{0:1,1:35},{0:1,1:30},{0:1,1:25},{0:1,1:20})}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)
plot_results('class_weight')


# ##### Min Samples Leaf

# In[ ]:


parameters = {'min_samples_leaf':(5,10,20,30,40,50)}
grid = GridSearchCV(clf, parameters, scoring='f1_macro', return_train_score=True)
grid.fit(data_X, data_Y)
plot_results('min_samples_leaf')


# #### Show Validation results

# In[ ]:


def plot_results(attr, print_results=False):
    results = np.asarray(list(zip(grid.cv_results_['params'],grid.cv_results_['mean_test_score'])))
    xs = [None] * results.shape[0]
    ys = np.empty(results.shape[0])
    
    for i,r in enumerate(results):
        xs[i] = str(r[0][attr])
        ys[i] = r[1]
        
    if print_results:
        results = np.flip(results[results[:,1].argsort()], axis=0)
        for r in results:
            print("| ", r[0][attr], " | ", r[1], " |")
      
    plt.figure(figsize=(20,10))
    plt.plot(xs,ys)
    plt.savefig(attr, bbox_inches='tight')


# ### Test Classifier <a name="test"></a>

# In[ ]:


clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)


# In[ ]:


score = f1_score(y_val, y_pred, average='macro')
print("f1:", score)
print("recall", recall_score(y_val, y_pred, average='macro'))
print("precision:", precision_score(y_val, y_pred, average='macro'))


# #### Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(2,2))
sb.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Score: {:.4}'.format(score)
plt.title(all_sample_title, size = 15);


# #### Confusion Matrix Test Data

# In[ ]:


train_pred = clf.predict(x_train)
cm = confusion_matrix(y_train, train_pred)
train_score = f1_score(y_train, train_pred, average='macro')
plt.figure(figsize=(2,2))
sb.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Score: {:.4}'.format(train_score)
plt.title(all_sample_title, size = 15);


#  ## Submission <a name="submission"></a>

# ### Preprocessing

# In[ ]:


data_test_X = data_test.drop(columns=['id'])


# #### Scaling

# In[ ]:


scaler = StandardScaler()
data_X = scaler.fit_transform(data_X,data_Y)
data_test_X = scaler.fit_transform(data_test_X)


# #### Feature Selection

# In[ ]:


rfe = RFE(estimator=clf, n_features_to_select=30)
rfe = rfe.fit(data_X,data_Y)
data_X = rfe.transform(data_X)
data_test_X = rfe.transform(data_test_X)


# #### Missing Data

# In[ ]:


imp = SimpleImputer(missing_values=-1, strategy='median')
imp = imp.fit(data_X,data_Y)
data_X = imp.transform(data_X)
data_test_X = imp.transform(data_test_X)


# ### Export Results

# In[ ]:


clf = clf.fit(data_X, data_Y)

y_target = clf.predict(data_test_X)
data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)

