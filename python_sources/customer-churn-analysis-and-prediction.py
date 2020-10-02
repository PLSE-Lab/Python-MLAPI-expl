#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Analysis and Prediction

# ## Introduction and Data
# 
# Customer Churn is one of the most important and common business problems. Even though it may not be fun to look at, it most definetly will provide insights on what improvements business can make in order to retain their customers. I have done couple of these analysis up to this point and my intend is to add couple more steps to the evaluation and model tunning process. The main business problem is the classic "Can we predict which type of customers most likely to leave our services?" The intend of this analysis is not to go into detail around broken down business objectives and problems but rather outline and practice the process of customer churn analysis and possible machine learning methodologies that I can use.
# 
# Data set is another common, publicly available and not real data set. The customer attributes are straight forward and self explanatory.

# In[ ]:


# Import initial libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data Collection and Cleaning

# In[ ]:


df = pd.read_csv("https://raw.githubusercontent.com/anilak1978/customer-churn/master/bigml_59c28831336c6604c800002a.csv")


# In[ ]:


df.head()


# In[ ]:


# checking for missing values
df.isnull().sum().values.sum()


# In[ ]:


# for loop to see unique values
for column in df.columns.values.tolist():
    print(column)
    print(df[column].unique())
    print("")


# In[ ]:


# check data types
df.dtypes


# In[ ]:


# update churn data type and boolen values to 0 and 1
df["churn"]=df["churn"].astype("str")


# In[ ]:


df["churn"]=df["churn"].replace({"False":0, "True":1})


# The dataset doesnt require further cleaning. It doesnt have any missing values, data types are correct and it is further ready for exploration.

# ## Data Exploration

# In[ ]:


# look at the brief overview of the data
df.info()


# In[ ]:


# look at statistical information
df.describe()


# In[ ]:


# group the data to see churn rate by state
df_state = df.groupby("state")["churn"].mean().reset_index()


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(x="state", y="churn", data=df_state)


# In[ ]:


# look at churn rate for all categorical variables
categorical_variables = ["area code", "international plan", "voice mail plan", "state"]
for i in categorical_variables:
    data=df.groupby(i)["churn"].mean().reset_index()
    plt.figure(figsize=(20,5))
    sns.barplot(x=data[i], y="churn", data=data)


# The churn rate for area code is around the same, while churn rate for internal plan is much higher and voice mail plan much lower.

# In[ ]:


# look at the distribution of categorical variables
for i in categorical_variables:
    plt.figure(figsize=(20,5))
    sns.countplot(x=df[i], data=df)


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x=df["churn"], data=df)


# Looking at the categorical variables and churn, the data is imbalanced. 

# In[ ]:


# Analysing numerical variables
numerical_variables=["account length", "number vmail messages", "total day minutes", "total day charge", "total day calls", 
                     "total day charge", "total eve minutes", "total eve charge", "total night minutes",
                    "total intl minutes", "total intl calls",
                    "total intl charge", "customer service calls"]


# In[ ]:


# looking at relationship for each numerical variable and churn
for i in numerical_variables:
    plt.figure(figsize=(20,5))
    sns.regplot(x=df[i], y="churn", data=df)


# In[ ]:


# looking at correlation within numerical variables
corr=df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True)


# ## Model Development - Random Forest
# 
# As it is obvious, we have labeled target (Churn) so I will use a classification to create a prediction model. I can create a model using Random Forest, Decision Tree, Support Vector Machine or Logistic Regression. I will use RandomForest and Support Vector Machine and compare the scores to see which model would be best. 
# 
# Based on our data exploration, I decided to select account length, international plan, total day charge, total night charge, total int charge, customer service calls and state as the feature set and of course the target (response) variable is Churn.

# In[ ]:


# feature selection
X = df[["account length", "international plan", "total day charge", "total night charge", "total intl charge", "customer service calls", "state"]]


# In[ ]:


# target selection
y =df["churn"]


# In[ ]:


# review feature set
X[0:5]


# In[ ]:


# update state with one hot coding
X=pd.get_dummies(X, columns=["state"])


# In[ ]:


# make sure i am using feature set values 
X=X.values


# In[ ]:


# preprocess to update str variables to numerical variables
from sklearn import preprocessing
international_plan=preprocessing.LabelEncoder()
international_plan.fit(["no", "yes"])
X[:,1] = international_plan.transform(X[:,1])


# In[ ]:


# create training and testing set
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)


# In[ ]:


#create model using random forest classifier and fit the training set
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_trainset, y_trainset)


# In[ ]:


#create prediction using the model
rf_pred = rf_model.predict(X_testset)
rf_pred[0:5]


# ## Model Evaluation Random Forest
# 
# In order to evaluate the model, I will first look at the accuracy score, however keep in mind, in the data analysis phase we observed that the data is imbalanced so the score for the model that we will get may not be as accurate as we think. Hence, i will also look at the precision and recall values and f1_score. Finally i will evaluate the selected feature set to see if we can make any improvements to the model.

# In[ ]:


# Looking at the accuracy score (using two methods)
from sklearn import metrics
rf_model.score(X_testset, y_testset)
metrics.accuracy_score(y_testset, rf_pred)


# In[ ]:


# confusion matrics to find precision and recall
from sklearn.metrics import confusion_matrix
confusion_matrix(y_testset, rf_pred)


# The model predicts 560 True Negatives, 13 False Positives, 54 False Negatives, 40 True Positives. 

# In[ ]:


# Looking at the precision score
from sklearn.metrics import precision_score
precision_score(y_testset, rf_pred)


# In[ ]:


# Looking at the recall score
from sklearn.metrics import recall_score
recall_score(y_testset, rf_pred)


# In[ ]:


# find probability for each prediction
prob=rf_model.predict_proba(X_testset)[:,1]


# In[ ]:


# look at ROC curve, which gives us the false and true positive predictions
from sklearn.metrics import roc_curve
fpr, tpr, thresholds=roc_curve(y_testset, prob)
plt.plot(fpr, tpr)


# In[ ]:


# Looking at the area under the curve
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_testset, prob)
auc


# In[ ]:


#looking at the f1_score
from sklearn.metrics import f1_score
f1_score(y_testset, rf_pred)


# In[ ]:


#Looking at the best possible estimator
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
param_grid={'n_estimators': np.arange(10,51)}
rf_cv=GridSearchCV(RandomForestClassifier(), param_grid)
rf_cv.fit(X,y)
rf_cv.best_params_


# In[ ]:


# looking at the best feature score
rf_cv.best_score_


# Couple things to note up to this point in the model evaluation phase; it is clear that I can certainly improve the precision , specially true positive count and n_estimator in the RandomForestClassifier parameter.

# In[ ]:


# looking at the importance of each feature
importances=rf_model.feature_importances_


# In[ ]:


# visualize to see the feature importance
indices=np.argsort(importances)[::-1]
plt.figure(figsize=(20,10))
plt.bar(range(X.shape[1]), importances[indices])
plt.show()


# Based on the bar chart, we can see that the first 5 features in the created X feature set is important and the rest are not. When we look at the X array, we will see that we can remove the variable state from the feature set.

# ## Model Development Support Vector Machine

# In[ ]:


# creating the svm model and fitting training set
# make sure to update probability to True for proabbility evaluation
from sklearn.svm import SVC
svc_model=SVC(probability=True)
svc_model.fit(X_trainset, y_trainset)


# In[ ]:


# creating the svm prediction
svc_pred=svc_model.predict(X_testset)
svc_pred[0:5]


# ## Model Evaluation Support Vector Machine

# In[ ]:


# look at the accuracy score
svc_model.score(X_testset, y_testset)


# SVM Accuracy score is lower than RandomForestClassifier accuracy score. 

# In[ ]:


# Look at the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_testset, svc_pred)


# The model predicts 567 True Negatives, 6 False Positives, 83 False Negatives, 11 True Positives. Even though the False Positive count slighlty went down, the True Positives are significantly less compare to RandomForestClassifier.

# In[ ]:


#precision score for svm
precision_score(y_testset, svc_pred)


# In[ ]:


# recall score for svm
recall_score(y_testset, svc_pred)


# Both precision and recall score is much lower compare to RandomForestClassifier.

# In[ ]:


# probability for each prediction
prob_2=svc_model.predict_proba(X_testset)[:,1]


# In[ ]:


# look at ROC curve
fpr, tpr, thresholds=roc_curve(y_testset, prob_2)
plt.plot(fpr, tpr)


# In[ ]:


# area under the curve
auc=roc_auc_score(y_testset, prob)
auc


# In[ ]:


# find ideal degree for SVM model
param_grid_2={'degree': np.arange(1,50)}
svc_cv=GridSearchCV(SVC(), param_grid_2)
svc_cv.fit(X,y)
svc_cv.best_params_


# ## Conclusion
# 
# Based on my analysis, the top features that we can use in order to predict if the customer will leave the services of the company are; "account length", "international plan", "total day charge", "total night charge", "total intl charge", "customer service calls". Upon development and evaluation of the two models, modified version of the model that uses Random Forest Classifier can be used to predict the customer churn. When I state modified version, I mean removing the state variable as part of the feature set and updating the n_estimator to 49. The current rate of 89% , precision and racall rates can be approved by tuning the Random Forest Model.

# In[ ]:




