#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will see how Grid Search works in a simplified manner. We will use `GridSearchCV` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV). 

# In[72]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.


# Scikit-learn module comes with some popular reference datasets including the methods to load and fetch them easily. We will use the [breast cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) wisconsin dataset for binary classification. The breast cancer dataset is a classic and very easy binary classification dataset.
# 
# `load_breast_cancer` method loads and returns the breast cancer wisconsin dataset. If `return_X_y` is made true then it returns `(data, target)`.

# In[73]:


X, y = load_breast_cancer(return_X_y=True)
print(X.shape)


# Lets use `train_test_split` to split the dataset into train and test sets.

# In[75]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)


# Use Standard Scalar for preprocessing the data. 

# In[77]:


ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)


# ## Random Forest without Grid Search
# First we will use `RandomForestClassifier` without Grid Search and with default values of hyperparameters.

# In[84]:


clf = RandomForestClassifier()
clf.fit(X_train_ss, y_train)
y_pred = clf.predict(X_test_ss)


# `plot_conf_matrix` is a function to plot a heatmap of confusion matrix.

# In[82]:


def plot_conf_matrix (conf_matrix, dtype):
    class_names = [0,1]
    fontsize=14
    df_conf_matrix = pd.DataFrame(
            conf_matrix, index=class_names, columns=class_names, 
        )
    fig = plt.figure()
    heatmap = sns.heatmap(df_conf_matrix, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix for {0}'.format(dtype))


# In[85]:


acc_rf = accuracy_score(y_test, y_pred)
print(acc_rf)


# In[86]:


plot_conf_matrix(confusion_matrix(y_test, y_pred), "Test data")


# There are 52 True Negatives, 82 Ttrue Positives, 2 False Positives and 7 False Negatives.

# ## Random Forest with Grid Search
# 
# We are tuning two hyperparameters of Random Forest classifier here - `n_estimators` and `max_depth`. We will use a list of values for `n_estimators` and a list of values for `max_depth`. Grid Search will search through all possible combinations (in this case it is 5 x 5 = 25 combinations) of hyperparameters to find out the best combination. Grid Search function will use `roc_auc` score here to evaluate validation set. There can be other [scoring functions](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) also that can be used based on the use cases. List of values of `n_estimators` and `max_depth` are given as key-value pairs (dictionary). We are also using 3 fold cross validation scheme (`cv = 3`).
# 
# Once the training data is fit into the model, best parameters from the Grid Search can be extracted using their names as keys.

# In[87]:



estimators = [10, 50, 100, 200, 500] 
max_depths = [3, 6, 10, 15, 20] 

grid_values = {'n_estimators': estimators, 'max_depth':max_depths}

clf = GridSearchCV(RandomForestClassifier(), grid_values, scoring='roc_auc', n_jobs=-1, verbose=10, cv=3)
clf.fit(X_train_ss, y_train)
best_n_estimators_value = clf.best_params_['n_estimators']
best_max_depth_value = clf.best_params_['max_depth']
best_score = clf.best_score_


# We will now plot the heatmap of AUC values for all possible combinations of `n_estimators` and `max_depth` values. There will be two heatmaps - one for train data and another for test data.

# In[88]:


max_depth_list = list(clf.cv_results_['param_max_depth'].data)
estimators_list = list(clf.cv_results_['param_n_estimators'].data)

sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_train_score']})
data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Training data')
plt.subplot(1,2,2)
data = pd.DataFrame(data={'Estimators':estimators_list, 'Max Depth':max_depth_list, 'AUC':clf.cv_results_['mean_test_score']})
data = data.pivot(index='Estimators', columns='Max Depth', values='AUC')
sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Test data')
plt.show()


# `plot_roc_curve` is a function to plot Receiver Operating Characteristic curve from train data and test data. 

# In[89]:


def plot_roc_curve(roc_auc_train, roc_auc_test):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr_tr, tpr_tr, 'g', label = 'Training AUC = %0.2f' % roc_auc_train)
    plt.plot(fpr_ts, tpr_ts, 'b', label = 'Testing AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# Now we will make a Random Forest Classifier with the best values of the hyperparameters. Then we will plot the ROC curve.

# In[90]:


#Best hyper parameter 
clf = RandomForestClassifier(n_estimators=best_n_estimators_value, max_depth=best_max_depth_value)
clf.fit(X_train_ss, y_train)

y_pred_train = clf.predict_proba(X_train_ss)[:,1]
y_pred_test = clf.predict_proba(X_test_ss)[:,1]

   
#train data ROC
fpr_tr, tpr_tr, threshold = roc_curve(y_train, y_pred_train)
roc_auc_train = auc(fpr_tr, tpr_tr)

#test data ROC
fpr_ts, tpr_ts, threshold = roc_curve(y_test, y_pred_test)
roc_auc_test = auc(fpr_ts, tpr_ts)

#Plot ROC curve
plot_roc_curve(roc_auc_train, roc_auc_test)


# In[91]:


acc_rf_grid = accuracy_score(y_test, clf.predict(X_test_ss))

print(acc_rf_grid)


# In[92]:


plot_conf_matrix(confusion_matrix(y_train, clf.predict(X_train_ss)), "Training data")


# In[93]:


plot_conf_matrix(confusion_matrix(y_test, clf.predict(X_test_ss)), "Test data")


# ## Conclusion

# In[94]:


# Compare both the models using Prettytable library    
x = PrettyTable()

x.field_names = ["Model", "n_estimators", "max_depth","Accuracy"]

x.add_row(["Random Forest w/o GridSearch", "default 10", "None", acc_rf])
x.add_row(["Random Forest with GridSearch", best_n_estimators_value, best_max_depth_value, acc_rf_grid])

print(x)


# In[ ]:




