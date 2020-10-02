#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[ ]:


# data viz and dataframe handling packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

#file handling
import os
#from google.colab import files

# data preprocessing
from sklearn.preprocessing import StandardScaler

# train test split
from sklearn.model_selection import train_test_split

# machine learning model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# model selection
from sklearn.model_selection import GridSearchCV

# model evaluation
from sklearn.metrics import (confusion_matrix,plot_confusion_matrix,plot_roc_curve,classification_report,accuracy_score,confusion_matrix)


# # Import Data & EDA

# In[ ]:


# import diabetes dataset from kaggle
os.environ['KAGGLE_USERNAME'] = "minkewang" # username
os.environ['KAGGLE_KEY'] = "4ac64942fb1cdf679a628708e3cae405" # key
get_ipython().system(' kaggle datasets download -d uciml/pima-indians-diabetes-database # api copied from kaggle')


# In[ ]:


dbdata = pd.read_csv('pima-indians-diabetes-database.zip', compression='zip', header=0, sep=',', quotechar='"',thousands=r',',encoding= 'unicode_escape')


# In[ ]:


dbdata.sample(5)


# In[ ]:


#check if there is nulls in the dataset
dbdata.isnull().sum()


# In[ ]:


# summary statistics of all columns
dbdata.describe()


# In[ ]:


# correlation plot
dbdata.corr().style.background_gradient(cmap=plt.cm.Blues)


# From the above correlation plot we can see that there is no sever mutlicoliearity problem with the dataset.

# In[ ]:


## patients with diabetes seem to have a normally distributed pregnancies times
sns.distplot(dbdata[dbdata['Outcome']==1]['Pregnancies'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['Pregnancies'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('Pregnancies for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['Pregnancies']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['Pregnancies']),color='blue', linestyle='--')


# In[ ]:


## Glucose level
sns.distplot(dbdata[dbdata['Outcome']==1]['Glucose'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['Glucose'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('Glucose for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['Glucose']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['Glucose']),color='blue', linestyle='--')


# In[ ]:


## BloodPressure
sns.distplot(dbdata[dbdata['Outcome']==1]['BloodPressure'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['BloodPressure'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('BloodPressure for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['BloodPressure']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['BloodPressure']),color='blue', linestyle='--')


# In[ ]:


## SkinThickness
sns.distplot(dbdata[dbdata['Outcome']==1]['SkinThickness'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['SkinThickness'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('SkinThickness for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['SkinThickness']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['SkinThickness']),color='blue', linestyle='--')


# In[ ]:


## Insulin
sns.distplot(dbdata[dbdata['Outcome']==1]['Insulin'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['Insulin'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('Insulin for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['Insulin']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['Insulin']),color='blue', linestyle='--')


# In[ ]:


## BMI
sns.distplot(dbdata[dbdata['Outcome']==1]['BMI'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['BMI'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('BMI for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['BMI']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['BMI']),color='blue', linestyle='--')


# In[ ]:


## DiabetesPedigreeFunction 
sns.distplot(dbdata[dbdata['Outcome']==1]['DiabetesPedigreeFunction'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['DiabetesPedigreeFunction'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('DiabetesPedigreeFunction for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['DiabetesPedigreeFunction']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['DiabetesPedigreeFunction']),color='blue', linestyle='--')


# In[ ]:


## Age 
sns.distplot(dbdata[dbdata['Outcome']==1]['Age'],bins=10,kde_kws={'label':'Diebetes'})
sns.distplot(dbdata[dbdata['Outcome']==0]['Age'],bins=10,kde_kws={'label':'No Diebetes'})
plt.title('Age for patients/non-patients')
plt.axvline(np.median(dbdata[dbdata['Outcome']==0]['Age']),color='red', linestyle='--')
plt.axvline(np.median(dbdata[dbdata['Outcome']==1]['Age']),color='blue', linestyle='--')


# In[ ]:


dbdata['Outcome'].value_counts()/len(dbdata)


# # Prediction: Diabetes Diagnosis 

# ### Standardize and split Tran Test Dataset 

# In[ ]:


#Define X, y variable Standardization 
y=dbdata.iloc[:,-1]
std = StandardScaler()                 # scale numeric columns
X = pd.DataFrame(std.fit_transform(dbdata.iloc[:,:-1]),columns=dbdata.iloc[:,:-1].columns)


# In[ ]:


#Split train test dataset
X_train,X_test, y_train,y_test= train_test_split(X, y, test_size=0.25, random_state=0)


# ### Define Functions for model and evaluation
# 
# **(confusion matrix, precision, recall, ROC AUC)**
# 
# 
# 

# In[ ]:


def model_prediction(algorithm, X_train, X_test, y_train):
  algorithm_fit = algorithm.fit(X_train, y_train)
  predictions  = algorithm.predict(X_test)
  probabilities = algorithm.predict_proba(X_test)
  return algorithm_fit, predictions, probabilities


# In[ ]:


def prediction(algorithm, X_train, X_test, y_train, y_test) :
    
    # model prediction
    algorithm_fit, predictions, probabilities = model_prediction(algorithm, X_train, X_test, y_train)

    # print summary
    print ("\n Classification report : \n", classification_report(y_test, predictions))
    print ("Accuracy   Score : ", accuracy_score(y_test, predictions))

    # plot confusion matrix 
    plot_confusion_matrix(algorithm_fit, X_test, y_test, cmap=plt.cm.Blues,display_labels=['No Diabetes','Diabetes'])
    plt.title('Confusion Matrix')
    
    # plot roc curve
    plot_roc_curve(algorithm_fit, X_test, y_test)
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
    plt.title('Receiver Operating Characteristic')
    


# ### Logistic Regression

# In[ ]:


##logistic regression classifier with hyperparameter tuning using GridSearchCV
parameters = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 20)}
logit = GridSearchCV(LogisticRegression(random_state=0),parameters,cv = 5, verbose=True, n_jobs=-1)
prediction(logit,X_train, X_test, y_train, y_test)


# Model Evaluation and Intepretation: 
# 

# ### Support Vector Machine -SVC

# In[ ]:


##svm classifer with hyperparameter tuning using GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = GridSearchCV(SVC(random_state=0,probability=True), parameters, cv = 5, verbose=True, n_jobs=-1)
prediction(svc,X_train, X_test, y_train, y_test)


# ### Random Forest Classification

# In[ ]:


## random forest classifer with hyperparameter tuning using GridSearchCV
parameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = GridSearchCV(RandomForestClassifier(), parameters, cv = 3, n_jobs = -1, verbose = 2)
prediction(rf,X_train, X_test, y_train, y_test)


# # Model Evaluation and Intepretation 
# 
# **Classification Report** <br>
# From the model classification report, we can see the accuracy rate of the model shows the overall rate of correctly predicted results or both true positive and true negatives out of all the predictions made. But it is not a good measure in this case because it gives equal importance to the false positives and false negatives. The data is imbalanced where patients data with only 35% with diabetes. Thus, correctly predicting no diabetes is of less use. 
# 
# **Confusion Matrix** <br>
# The confusion matrix shows the predicted result on the test dataset using our trained model. 
# 
# **Precision vs Recall**<BR>
# In this classification task where we would like to correctly predict the result of diabete diagnostic, the result of false negative is more severe than false positives because informing patients of no disease can result in delayed medical treatment and damage their health. Thus, we should give more emphasis to the metrics on the false negative rate, or the recall which is 
# the proportion of correctly identified positive out of all actual positives. Precision here indicates the proportion of correctly predicted positive observations out of all predicted psitive indentifications which is of less importance than the recall because higher recall lead to more severe outcome on the patients. Based on the tuned model of Logistic Regression, Random Forest and Support Vector Machine, we can clearly see that the SVC model performace is better. Thus, we can use this trained model to predict the diabetes diagnostic in the future. 
# 
# **ROC-AUC**<BR>
# The ROC-AUC curve shows the True positive rate versus the False positive rate curve for all the threshold values ranging from 0 to 1. In an ROC curve, each point in the ROC space is associated with a different confusion matrix. A diagnoal line from the bottom left to the top right represent we have at least 50% chance to correctly predict diabetes even if we are guess randomly. AUC shows how much the model is capable of distingushing between different classes. From the tuned model above, we can see that the three models have similar ROC curve meaning that the three models perform similar in terms of identify postive and negative diabetes diagnostics. 

# In[ ]:




