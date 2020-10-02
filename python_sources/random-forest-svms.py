#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Importing the required libabries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data = pd.read_csv(r'/kaggle/input/hr-attrition/HR_Employee_Attrition_Data.csv')


# In[ ]:


# Description of DataFrame

print("The data has {} rows and {} columns".format(data.shape[0], data.shape[1]))
print('#'*75)
print(data.dtypes)
data_cols = list(data.columns)
print(data_cols)


# In[ ]:


# Description of the dataset

print(data.describe(include='all').T)


# In[ ]:


data.head()


# In[ ]:


# Resetting the index
data.set_index('EmployeeNumber', inplace = True)


# In[ ]:


data.head()


# In[ ]:


# Mapping the catagorical variable 'Attrition' to 'Numerical' values using map

data['Attrition'] = data['Attrition'].map({'Yes':1, 'No':0})

# 1 Indicates employee resigning and 0 indicates employee staying with the Org


# In[ ]:


data.head(10)


# In[ ]:


cols_object = [var for var in data.columns if data[var].dtype == 'O']
print(cols_object)


# In[ ]:


data.drop('Over18', axis = 1, inplace = True)


# In[ ]:


from sklearn import preprocessing

def preprocessor(df):
    res_df = df.copy()
    le = preprocessing.LabelEncoder()
    
    res_df['BusinessTravel'] = le.fit_transform(res_df['BusinessTravel'])
    res_df['Department'] = le.fit_transform(res_df['Department'])
    res_df['EducationField'] = le.fit_transform(res_df['EducationField'])
    res_df['Gender'] = le.fit_transform(res_df['Gender'])
    res_df['JobRole'] = le.fit_transform(res_df['JobRole'])
    res_df['MaritalStatus'] = le.fit_transform(res_df['MaritalStatus'])
    res_df['OverTime'] = le.fit_transform(res_df['OverTime'])
    
    return res_df


# In[ ]:


encoded_df = preprocessor(data)


# In[ ]:


encoded_df.head()
print(encoded_df.dtypes)


# In[ ]:


feature_space = encoded_df.iloc[:, encoded_df.columns != 'Attrition']
feature_class = encoded_df.iloc[:, encoded_df.columns == 'Attrition']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(feature_space, feature_class, test_size = 0.2, random_state =42 )


# In[ ]:


X_train.values


# In[ ]:


rf = RandomForestClassifier(random_state = 42)


# In[ ]:


y_test = y_test.values.ravel() 
y_train = y_train.values.ravel() 


# # Hyperparameter tuning using GridSearchCV

# In[ ]:


import time
np.random.seed(42)

start = time.time()

param_dist = {'max_depth':[2,3,4,5,6,7,8],
             'bootstrap':[True, False],
             'max_features':['auto', 'sqrt', 'log2', None],
             'criterion':['gini', 'entropy']}

cv_rf = GridSearchCV(rf, cv = 10, param_grid = param_dist, n_jobs = 3)
cv_rf.fit(X_train, y_train)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[ ]:


rf.set_params(criterion = 'entropy',
                  max_features = None, 
                  max_depth = 8, bootstrap = True)


# In[ ]:


rf.set_params(warm_start = True, oob_score = True)


# In[ ]:


# Estimation of the error rate for each n_estimators

# For estimating n_estimators, warm_start has to be set as True and oob_score as True
# rf.set_params(***) - sets the parameters for the model defined earlier. 
# In thi scase, rf is the model name for RandomForestClassifier

min_estimators = 100
max_estimators = 1000





error_rate = {}

for i in range(min_estimators, max_estimators+1):
    rf.set_params(n_estimators = i)
    rf.fit(X_train, y_train)
    oob_error = 1 - rf.oob_score_
    error_rate[i] = oob_error


# # OOB Rate
# 
# - OOB stands for Out of Bag
# - Bootstrapping is a technique wherein the samples are selected in random with replacement from the original dataset
# - It may also happen that an observation might be selected more than once in a bootstrap
# - The proportion of samples that are left out after bootstrapping is equal to (1-(1/N))^N. This approximates to 36.8 % 
# - All the out of bag samples are used to cross validate the classification against each Decision Tree that form the Ensemble(Random Forest)
# - The aggreagate voting from each validation is used to classify the out of bag data from the dataset
# - The OOB error rate is the number of misclassifications that occur on the out of bag samples from the train data
# - This is equal to (1-oob_score_). Choose the n_estimators that gives the less oob error rate

# In[ ]:


oob_series = pd.Series(error_rate)


# In[ ]:


plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 12))

ax.set_facecolor('#e6e6ff')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.074, color='#33FFE5',linestyle='--')
plt.axhline(0.071, color='#33FFE5',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 100 to 1000 trees)')
plt.show()


# In[ ]:


for i in range(100, 1000, 100):
    print('OOB Error rate for {} trees is {}'.format(i, oob_series[i]))


# In[ ]:


# Refine the tree via OOB Output
rf.set_params(n_estimators=800,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


cols = [var for var in X_train.columns]
cols_df = pd.DataFrame(cols, columns = ['Feature_Name'])

importance = list(rf.feature_importances_)
print(importance, len(importance))


# In[ ]:


imp = pd.DataFrame(importance, columns = ['Importance'])
feature_imp = pd.concat([cols_df, imp], axis = 1)


# In[ ]:


feature_imp


# In[ ]:


# Plotting a barplot to identify & visualize feature importance
plt.figure(figsize=(16,12))
x = sns.barplot(feature_imp['Feature_Name'], feature_imp['Importance'])
x.set_xticklabels(labels=feature_imp.Feature_Name.values, rotation=90)
plt.show()


# # Making predictions on the Test Data

# In[ ]:


predictions = rf.predict(X_test)

probability = rf.predict_proba(X_test)

fpr, tpr, threshold = roc_curve(y_test, probability[:,1])
print(fpr)
print(tpr)
print(threshold)


# # Calculating Sensitivity (Recall) & Specificity

# In[ ]:


# Printing of the Confusion Matrix

cm = confusion_matrix(y_test, predictions)

print(cm)
print(type(cm))


# In[ ]:


# Code to plot Confusion matrix in graphical way

sns.set(font_scale=1.8) # scaling the font sizes
plt.figure(figsize=(10,10))

sns.heatmap(cm, annot=True, cbar=False, fmt = '', xticklabels = ['TRUE', 'FALSE'], yticklabels = ['TRUE', 'FALSE'])
plt.xlabel('Predicted', color = 'blue', fontsize = 'xx-large' )
plt.ylabel('Actual', color = 'blue', fontsize = 'xx-large')
plt.show()


# In[ ]:


TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

print(TP, FP, FN, TN)


# In[ ]:


Recall = (TP / (TP+FN))
Specificity = (TN / (FP+TN))
Accuracy = ((TP+TN) / (TP+TN+FP+FN))


print("The Recall score is: {} ".format(np.round(Recall,2)))
print("The Specificity score is : {}".format(np.round(Specificity,2)))
print("The Accuracy score is: {}".format(np.round(Accuracy,2)))


# In[ ]:


# The Accuracy score obtained using rf.score and manual calculation yield the same result
print(rf.score(X_test, y_test))


# # Plotting ROC AUC Curve

# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,10))

plt.plot(fpr, tpr, color = 'red', lw =2, label = 'Decision Tree (AUC = {})'.format(np.round(roc_auc, 2)))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()


# # HR Attrition using SVMs

# In[ ]:


data_svm = encoded_df.copy()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[ ]:


cont_cols = [var for var in data_svm.columns if data_svm[var].dtype != 'O']
print(cont_cols)


# In[ ]:


corr_mat = data_svm.corr()
sns.set(font_scale=1.5)
f, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(corr_mat, vmax =1, annot = True , square = False, annot_kws={"size":15},  cbar = False, fmt=".2f", cmap='coolwarm')


# Arguments used for heatmap
# cmap = colormap (coolwarm )


# In[ ]:


cols_drop = ['JobLevel', 'YearsWithCurrManager', 'StandardHours', 'EmployeeCount', 'YearsInCurrentRole']


# In[ ]:


data_wo_corr = data_svm.drop(cols_drop, axis = 1)
data_wo_corr.head()


# In[ ]:


# feature separation

X = data_wo_corr.drop(['Attrition'], axis = 1)
y = data_wo_corr['Attrition']


scaler.fit(X)
X = scaler.transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # SVM with default HyperParameters

# In[ ]:


from sklearn.svm import SVC
from sklearn import metrics
svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


# Using Linear Kernel

svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


# Using Polynomial kernel

svc=SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


kernel_choice = ['linear', 'poly', 'rbf']

for val in kernel_choice:
    svc = SVC(kernel = val)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print("Accuracy score using {} kernel is: {}".format(val, metrics.accuracy_score(y_test,y_pred)))


# # Optimizing the HyperParameter C

# In[ ]:


#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score

C_range=list(range(1,26))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


C_values=list(range(1,26))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')


# In[ ]:


C_range=list(np.arange(0.1,6,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    
    acc_score.append(scores.mean())
print(acc_score)    


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

C_values=list(np.arange(0.1,6,0.1))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.figure(figsize=(16,8))
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0.0,6,0.3))
plt.xlabel('Value of C for SVC ')
plt.ylabel('Cross-Validated Accuracy')


# # Optimizing the HyperParameter Gamma

# In[ ]:


gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)  


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.figure(figsize = (16,12))
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.xticks(np.arange(0.0001,100,5))
plt.ylabel('Cross-Validated Accuracy')


# # Using GridSearchCV to find Hyperparameters (C, gamma, kernel)

# In[ ]:


from sklearn.svm import SVC
svm_model= SVC()

tuned_parameters = { 'kernel': ['linear', 'rbf', 'poly'],
 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'degree': [2,3,4] }


# In[ ]:


#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')

model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[ ]:


print(model_svm.best_params_)


# In[ ]:


svm_model.set_params(C = 0.9, degree = 3, gamma = 0.05, kernel = 'poly', probability = True)


# In[ ]:


svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)


# In[ ]:


proba = svm_model.predict_proba(X_test)

fpr, tpr, threshold = roc_curve(y_test, proba[:,1])
print(fpr)
print(tpr)
print(threshold)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

print(cm)
print(type(cm))


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(svm_model, X_test, y_test)


# In[ ]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(cm,display_labels = [1,0]).plot()


# In[ ]:


# Code to plot Confusion matrix in graphical way

sns.set(font_scale=1.8) # scaling the font sizes
plt.figure(figsize=(10,10))

sns.heatmap(cm, annot=True, cbar=False, fmt = '', xticklabels = ['TRUE', 'FALSE'], yticklabels = ['TRUE', 'FALSE'])
plt.xlabel('Predicted', color = 'blue', fontsize = 'xx-large' )
plt.ylabel('Actual', color = 'blue', fontsize = 'xx-large')
plt.show()


# In[ ]:


TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]

print(TP, FP, FN, TN)

Recall = (TP / (TP+FN))
Specificity = (TN / (FP+TN))
Accuracy = ((TP+TN) / (TP+TN+FP+FN))


print("The Recall score is: {} ".format(np.round(Recall,2)))
print("The Specificity score is : {}".format(np.round(Specificity,2)))
print("The Accuracy score is: {}".format(np.round(Accuracy,2)))


# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,10))

plt.plot(fpr, tpr, color = 'red', lw =2, label = 'Decision Tree (AUC = {})'.format(np.round(roc_auc, 2)))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc="lower right")
plt.show()

