#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import Libraries 
import numpy as np 
import xgboost as xgb
import numpy as np
from collections import OrderedDict
import gc
from glob import glob
import os
import pandas as pd
from copy import copy
from time import time
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


### Display all the columns of dataframe
pd.set_option('display.max_columns', None)


# In[ ]:


train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
validation_sample = pd.read_csv('../input/gender_submission.csv')
#print(train_set.head())
print(len(train_set))
print(len(test_set))
print(len(validation_sample))


# In[ ]:


print (train_set.columns)
print(validation_sample.columns)


# ###  Exploratory Data Anlysis:

# In[ ]:


### Check the rows for each class
pd.DataFrame(train_set['Survived'].value_counts())


# In[ ]:


train_set.describe()


# In[ ]:


train_set.columns


# In[ ]:


train_set.dtypes


# In[ ]:


train_set.head()


# In[ ]:


pd.DataFrame(train_set['Parch'].value_counts())


# In[ ]:


pd.DataFrame(train_set['SibSp'].value_counts())


# In[ ]:


train_set["Age"]=train_set["Age"].fillna(train_set["Age"].median())
train_set["Fare"]=train_set["Fare"].fillna(train_set["Fare"].median())
train_set["Embarked"]=train_set["Embarked"].fillna(train_set["Embarked"].mode()[0])


# In[ ]:


#train_set["Child"]=train_set["Age"].apply(lambda x : 1 if x<15 else 0 )
#train_set["Teenager"]=train_set["Age"].apply(lambda x : 1 if (x>=15) and (x<25) else 0 )
#train_set["Adult"]=train_set["Age"].apply(lambda x : 1 if (x>=25) & (x<65) else 0 )
#train_set["Old"]=train_set["Age"].apply(lambda x : 1 if x>=65 else 0 )


# In[ ]:


train_set['Age_new'] = np.log(1+train_set.Age) 
#train_set['Fare_new'] = np.log(1+train_set.Fare) 


# In[ ]:


train_set['Age_new'].value_counts()


# In[ ]:


pd.DataFrame(train_set.isnull().sum())


# In[ ]:


### One hot encoding for the categorical data
cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# find unique labels for each category
cat_data = pd.get_dummies(train_set[cat_vars])

# check that the categorical variables were created correctly
cat_data.head()


# In[ ]:


numeric_vars = list(set(train_set.columns.values.tolist())- set(cat_vars))
numeric_vars


# In[ ]:


numeric_vars = list(set(train_set.columns.values.tolist()) - set(cat_vars))
numeric_vars.remove('Survived')
numeric_vars.remove('PassengerId')
numeric_vars.remove('Name')
numeric_vars.remove('Ticket')
numeric_vars.remove('Cabin')
numeric_vars.remove('Age')
numeric_data = train_set[numeric_vars].copy()
# check that the numeric data has been captured accurately
numeric_data.head()


# In[ ]:





# In[ ]:


# concat numeric and the encoded categorical variables
numeric_cat_data = pd.concat([numeric_data, cat_data], axis=1)

# check that the data has been concatenated correctly by checking the dimension of the vectors
print(cat_data.shape)
print(numeric_data.shape)
print(numeric_cat_data.shape)


# In[ ]:


numeric_cat_data.head()


# In[ ]:


# capture the labels
labels = train_set['Survived'].copy()
# split data into test and train
x_train, x_test, y_train, y_test = train_test_split(numeric_cat_data,
                                                    labels,
                                                    test_size=.30, 
                                                    random_state=2046)


# In[ ]:


# check that the dimensions of our train and test sets are okay
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


x_train.head()


# In[ ]:


pd.DataFrame(y_train[0:10])


# In[ ]:


### Hyper Parameter Tuning: 
params = {
    'num_rounds':       10,
    'max_depth':         4,
    'max_leaves':        2**4,
    'alpha':             0.9,
    'eta':               0.1,
    'gamma':             0.1,
    'learning_rate':     0.11115,
    'subsample':         1,
    'reg_lambda':        1,
    'scale_pos_weight':  2,
    'objective':         'binary:logistic',
    'verbose':           True
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "dtrain = xgb.DMatrix(x_train, label=y_train)\ndtest = xgb.DMatrix(x_test, label= y_test)\nevals = [(dtest, 'test',), (dtrain, 'train')]\n\nnum_rounds = params['num_rounds']\nmodel = xgb.train(params, dtrain, num_rounds, evals=evals)")


# In[ ]:


threshold = .5
true_labels = y_test.astype(int)
true_labels.sum()

# make predictions on the test set using our trained model
preds = model.predict(dtest)
print(preds)


# In[ ]:


pred_labels = (preds > threshold).astype(int)
print(pred_labels)


# In[ ]:


pred_labels.sum()


# In[ ]:


# compute the auc
auc = roc_auc_score(true_labels, pred_labels)
print(auc)


# In[ ]:


print ('Accuracy:', accuracy_score(true_labels, pred_labels))


# In[ ]:


results = confusion_matrix(true_labels, pred_labels) 

print ('Confusion Matrix :')

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


plot_confusion_matrix(results, ['0','1'])


# In[ ]:


### AUC
fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, pred_labels)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Plot')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


### Model Inspection 
xgb.plot_importance(model)


# ### Random Forest:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


rfc = RandomForestClassifier(n_estimators=1000, random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)


# In[ ]:


Accuracy_rfc = accuracy_score(y_test, y_pred)
print("The Random Forest Classifier Model has an accuracy of : %.5f%%" % (Accuracy_rfc * 100.0))


# In[ ]:


# grid_param dictionary
grid_param = {  
    'n_estimators': [10,20,30,60,100],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


# In[ ]:


# instance of the GridSearchCV class
gds_rfc = GridSearchCV(estimator=rfc,     
                     param_grid=grid_param,    
                     scoring='accuracy',       
                     cv=5,                     
                     n_jobs=-1) 


# In[ ]:


gds_rfc.fit(x_train, y_train)


# In[ ]:


# Optimal hyperparameters: best_params_
gds_rfc.best_params_


# In[ ]:


# Best score found (mean score on all folds used as validation set): best_score_
Acc_gds_rfc=gds_rfc.best_score_


# In[ ]:


print("Random Forest Classifier-Hyperparameters Model has an accuracy of : %.2f%%" % (Acc_gds_rfc * 100.0))


# > ### Support Vector Classifier:

# In[ ]:


### Support Vector Machine:
from sklearn.svm import SVC


# In[ ]:


svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)


# In[ ]:


Acc_svc = accuracy_score(y_test, y_pred)
print("The Naive Bayes Model has an accuracy of : %.2f%%" % (Acc_svc * 100.0))


# ### Apply in Test Data Set:

# In[ ]:


test_set["Age"]=test_set["Age"].fillna(test_set["Age"].median())
test_set["Fare"]=test_set["Fare"].fillna(test_set["Fare"].median())
test_set["Embarked"]=test_set["Embarked"].fillna(test_set["Embarked"].mode()[0])


# In[ ]:


pd.DataFrame(test_set.isnull().sum())


# In[ ]:


test_set['Age_new'] = np.log(1+test_set.Age) 
#test_set['Fare_new'] = np.log(1+test_set.Fare) 


# In[ ]:



### One hot encoding for the categorical data
cat_vars_test = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# find unique labels for each category
cat_data_test = pd.get_dummies(test_set[cat_vars_test])

# check that the categorical variables were created correctly
cat_data_test.head()

numeric_vars_test = list(set(test_set.columns.values.tolist())- set(cat_vars_test))
numeric_vars_test

numeric_vars_test = list(set(test_set.columns.values.tolist()) - set(cat_vars_test))
numeric_vars_test.remove('PassengerId')
numeric_vars_test.remove('Name')
numeric_vars_test.remove('Ticket')
numeric_vars_test.remove('Cabin')
numeric_vars_test.remove('Age')
#numeric_vars_test.remove('Fare')
numeric_data_test = test_set[numeric_vars_test].copy()
# check that the numeric data has been captured accurately
numeric_data_test.head()

# concat numeric and the encoded categorical variables
numeric_cat_data_test = pd.concat([numeric_data_test, cat_data_test], axis=1)

# check that the data has been concatenated correctly by checking the dimension of the vectors
print(cat_data_test.shape)
print(numeric_data_test.shape)
print(numeric_cat_data_test.shape)


# In[ ]:


dtest = xgb.DMatrix(numeric_cat_data_test)
# make predictions on the test set using our trained model
preds = model.predict(dtest)
print(preds)


# In[ ]:


pred_labels = (preds > threshold).astype(int)
print(pred_labels)
print (len(pred_labels))


# In[ ]:


test_set['Survived'] = pred_labels
submission = test_set[['PassengerId', 'Survived']]
submission.to_csv('submission_xgboost3.csv', index=False)


# In[ ]:


submission.head()


# ### RFC Prediction:

# In[ ]:


preds_rfc = rfc.predict(numeric_cat_data_test)
print(preds_rfc)


# In[ ]:


test_set['Survived'] = preds_rfc
submission = test_set[['PassengerId', 'Survived']]
submission.to_csv('submission_rfc.csv', index=False)


# In[ ]:


len(test_set)


# In[ ]:




