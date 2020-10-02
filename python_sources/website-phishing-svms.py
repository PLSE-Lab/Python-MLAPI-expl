#!/usr/bin/env python
# coding: utf-8

#   # Introduction
# Multiclass Website Phishing dataset problem. Fitting SVMs algorithm and creating accuracy score and confusion matrix of predicted values and real values. I was able to get 88 % accuracy.
# 

# # STEP #1: Import Dataset

# In[ ]:


import numpy as np
import pandas as pd

phishing_data = pd.read_csv('../input/Website Phishing.csv')
print(phishing_data.columns)
phishing_data.head()


# # STEP #2: Explore /Visualze Data set

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

a=len(phishing_data[phishing_data.Result==0])
b=len(phishing_data[phishing_data.Result==-1])
c=len(phishing_data[phishing_data.Result==1])
print(a,"times suspecious(0) repeated in Result")
print(b,"times phishy(-1) repeated in Result")
print(c,"times legitimate(1) repeated in Result")
sns.countplot(phishing_data['Result'])


# In[ ]:


sns.heatmap(phishing_data.corr(),annot=True)


# In[ ]:


phishing_data.info()
phishing_data.describe()


# # STEP #3: Loading data

# In[ ]:


x = phishing_data.drop('Result',axis=1).values 
y = phishing_data['Result'].values


# In[ ]:


#splitting data holdout method

from sklearn.model_selection import train_test_split
#splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)


# # STEP #4: Model one-vs-one Training

# In[ ]:


#multiple class clasification one-vs-one
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict

svm_model_oneVSone = OneVsOneClassifier(SVC(kernel='linear', C=1, gamma=0.1))
svm_model_oneVSone.fit(x_train, y_train)
y_pred = cross_val_predict(svm_model_oneVSone,x_train,y_train,cv=10)
y_pred = svm_model_oneVSone.predict(x_test)

# accuracy and confusion matric
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix ,f1_score

cm = confusion_matrix(y_test, y_pred) 
print("confusion_matrix: ")
print(cm)
score = f1_score(y_test , y_pred,average=None)
print("f1_score: ",score)
cross_val_score1 = cross_val_score(svm_model_oneVSone,x_train,y_train,cv=10)
print("cross validation mean : ",cross_val_score1.mean())


# # STEP #5: Model one-vs-rest Training

# In[ ]:


#multiple class clasification one-vs-Rest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

svm_model_oneVSall = OneVsRestClassifier(LinearSVC(random_state=300))
#svm_model_oneVSall = LinearSVC(random_state=300)
svm_model_oneVSall.fit(x_train, y_train)
y_pred_multiclass = svm_model_oneVSall.predict(x_test)

# accuracy and confusion matric
from sklearn.metrics import confusion_matrix ,f1_score

cm = confusion_matrix(y_test, y_pred_multiclass) 
print("confusion_matrix: ")
print(cm)
score = f1_score(y_test , y_pred_multiclass,average=None)
print("f1_score: ",score)
cross_val_score2 = cross_val_score(svm_model_oneVSall,x_train,y_train,cv=10)
print("cross validation mean : ",cross_val_score2.mean())


# 1. # STEP #6: Model hyperparameter tuning using GridSearch and Cross Validation

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
params_grid = [{'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1,0.7],'C': [0.1, 1, 10,100]},
               {'kernel': ['linear'], 'gamma': [0.001, 0.01, 0.1, 0.7,1,10],'C': [0.1, 1, 10,100]},
               {'kernel': ['poly'],'gamma': [0.001, 0.01, 0.1,0.7, 1,10],'C': [0.1, 1, 10,100]}]


# Performing CV to tune parameters for best SVM fit 
svm_model = GridSearchCV(SVC(), params_grid, cv= 5)
svm_model.fit(x_train, y_train)

# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n") 

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n") 
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

final_model = svm_model.best_estimator_
y_pred_best = final_model.predict(x_test)
y_pred_label = list(y_pred)


# # STEP #7: Model Testing and Evaluation

# In[ ]:


# accuracy and confusion matric
from sklearn.metrics import confusion_matrix ,f1_score

cm = confusion_matrix(y_test, y_pred_best) 
print("confusion_matrix: ")
print(cm)
score = f1_score(y_test , y_pred_best,average=None)
print("f1_score: ",score)
cross_val_score3 = cross_val_score(final_model,x_train,y_train,cv=10)
print("cross validation mean : ",cross_val_score3.mean())

