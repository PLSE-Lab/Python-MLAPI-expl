#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the libraries to be used in data analysis and churn data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # supress warnings
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

from pickle import dump, load

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.discrete.discrete_model as sm

# With the "na_values" parameter, introducing possible formats of missing data.
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv", na_values=["nan", " ","NaN", "-", "_", ".", "NAN"])

# Firstly copy the original data, because then the data will changed. 
data_original = data.copy()


# **DATA PRE PROCESSING**

# Firstly, for recognize the data, we will do following:
# - print top and last five rows
# - print columns (features) name
# - print types of columns (features)
# - check the missing data

# In[ ]:


# top 5 rows
data.head()


# In[ ]:


# last 5 rows
data.tail()

# Also you can use for top 10 rows; "data.head(10)" or for last 7 rows; "data.tail(7)" etc. 


# In[ ]:


# print the columns name, types of columns and count of total rows.
data.info()

# Also, you can use following code for recognize to data.
# data.columns
# data.dtypes
# data.describe()


# We have the 7043 rows. As you can use above, all of columns are non-null excluding "TotalCharges" feature. 
# "TotalCharges" has 11 missing data. We will fill these later.
# 
# Checking the unique variable of string features. ('object' mentioned above means string)

# In[ ]:


print(
" gender:", data.gender.unique(),"\n",
"Partner:", data.Partner.unique(),"\n",
"Dependents:", data.Dependents.unique(),"\n",
"PhoneService:", data.PhoneService.unique(),"\n",
"MultipleLines:", data.MultipleLines.unique(),"\n",
"InternetService:", data.InternetService.unique(),"\n",
"OnlineSecurity:", data.OnlineSecurity.unique(),"\n",
"OnlineBackup:", data.OnlineBackup.unique(),"\n",
"DeviceProtection:", data.DeviceProtection.unique(),"\n",
"TechSupport:", data.TechSupport.unique(),"\n",
"StreamingTV:", data.StreamingTV.unique(),"\n",
"StreamingMovies:", data.StreamingMovies.unique(),"\n",
"Contract:", data.Contract.unique(),"\n",
"PaperlessBilling:", data.PaperlessBilling.unique(),"\n",
"PaymentMethod:", data.PaymentMethod.unique(),"\n",
"Churn:", data.Churn.unique(),"\n"
)


# Now, we will examine the effect of the each variable on the churn with graphs.
# 
# To be able to compare; we calculate the mean of churn.

# In[ ]:


data.groupby('Churn')["customerID"].count()


# In[ ]:


churn_rate = 1869 / (5174+1869)
print(churn_rate)


# In[ ]:


figure = plt.figure(figsize=(20,15))

plt.subplot2grid((2,3),(0,0))
data.gender[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("Male - churn")

plt.subplot2grid((2,3),(1,0))
data.SeniorCitizen[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("Senior Citizen - churn")

plt.subplot2grid((2,3),(0,1))
data.Partner[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("Partner - churn")

plt.subplot2grid((2,3),(1,1))
data.MultipleLines[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("MultipleLines - churn")

plt.subplot2grid((2,3),(0,2))
data.InternetService[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("InternetService - churn")

plt.subplot2grid((2,3),(1,2))
data.OnlineSecurity[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("OnlineSecurity - churn")

plt.show()


# As seen from the graphs above; 
# - 'being a senior citizen' is decreasing churn
# - 'being a partner' is decreasing churn
# - 'having a fiber optic' are increasing churn
# - 'not having online security' are increasing churn
# 

# In[ ]:


figure = plt.figure(figsize=(20,15))

plt.subplot2grid((2,3),(0,0))
data.OnlineBackup[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("OnlineBackup - churn")

plt.subplot2grid((2,3),(1,0))
data.DeviceProtection[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("DeviceProtection - churn")

plt.subplot2grid((2,3),(0,1))
data.TechSupport[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("TechSupport - churn")

plt.subplot2grid((2,3),(1,1))
data.StreamingTV[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("StreamingTV - churn")

plt.subplot2grid((2,3),(0,2))
data.StreamingMovies[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("StreamingMovies - churn")

plt.subplot2grid((2,3),(1,2))
data.Contract[data.Churn=='Yes'].value_counts(normalize=True).sort_index().plot(kind="bar", grid=True)
plt.title("Contract - churn")

plt.show()


# As it appears;
# - "not have a Tech support" is increasing churn
# - "having month-month contract type" is increasing churn
# - "having one-year or two-year contract type" is increasing churn
# 
# In the same way, it can be analyze other features.

# We will transform features that containing only 'Yes' and 'No' variables from string to integer format.
# We will print "1" instead of 'Yes' and same way '0' instead of 'No'

# In[ ]:


data.loc[data["gender"]=="Male", "gender"] = 1
data.loc[data["gender"]=="Female", "gender"] = 0

data.loc[data["Partner"]=="Yes", "Partner"] = 1
data.loc[data["Partner"]=="No", "Partner"] = 0

data.loc[data["Dependents"]=="Yes", "Dependents"] = 1
data.loc[data["Dependents"]=="No", "Dependents"] = 0

data.loc[data["PhoneService"]=="Yes", "PhoneService"] = 1
data.loc[data["PhoneService"]=="No", "PhoneService"] = 0

data.loc[data["PaperlessBilling"]=="Yes", "PaperlessBilling"] = 1
data.loc[data["PaperlessBilling"]=="No", "PaperlessBilling"] = 0

data.loc[data["Churn"]=="Yes", "Churn"] = 1
data.loc[data["Churn"]=="No", "Churn"] = 0


# Later the transformation, check to types of columns again.

# In[ ]:


data.info()


# We will transform remaining columns that containing 3 or 4 variables to dummy columns.
# 
# For example, "InternetService" has 3 variables; 'DSL', 'Fiber optic' and 'No'.
# Now, three new columns named "InternetService_DSL", "InternetService_Fiber optic" and "InternetService_No" will be created and all of these has containg 2 variables; '1' and '0'
# 
# And, we will drop original columns we've created dummy versions.

# In[ ]:


data.columns


# In[ ]:


MultipleLines = pd.get_dummies(data['MultipleLines'], prefix="MultipleLines", prefix_sep="_")
InternetService = pd.get_dummies(data['InternetService'], prefix="InternetService", prefix_sep="_")
OnlineSecurity = pd.get_dummies(data['OnlineSecurity'], prefix="OnlineSecurity", prefix_sep="_")
OnlineBackup = pd.get_dummies(data['OnlineBackup'], prefix="OnlineBackup", prefix_sep="_")
DeviceProtection = pd.get_dummies(data['DeviceProtection'], prefix="DeviceProtection", prefix_sep="_")
TechSupport = pd.get_dummies(data['TechSupport'], prefix="TechSupport", prefix_sep="_")
StreamingTV = pd.get_dummies(data['StreamingTV'], prefix="StreamingTV", prefix_sep="_")
StreamingMovies = pd.get_dummies(data['StreamingMovies'], prefix="StreamingMovies", prefix_sep="_")
Contract = pd.get_dummies(data['Contract'], prefix="Contract", prefix_sep="_")
PaymentMethod = pd.get_dummies(data['PaymentMethod'], prefix="PaymentMethod", prefix_sep="_")

# concatenating above dummy variables to our actual data. 
data = pd.concat([data, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                  DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, 
                  PaymentMethod], axis=1)

# dropping the original columns.
# "inplace=True" means; make on the original data.
data.drop(["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
           "PaymentMethod"], axis=1, inplace=True)


# Now, we'll rearrange the column names that contain the space character and check column names again.

# In[ ]:


data.columns = [ i.replace(" ","_") if len(i.split()) > 1 else i for i in data.columns ]
data.columns


# We have 11 missing data for "TotalCharges" columns. Instead of this missing data, we fill median of Total Charges group by Monthly Charges. Because "TotalCharges" and "MonthlyCharges" have the highest correlation.

# In[ ]:


data.loc[data.TotalCharges.isnull(),"TotalCharges"] = data.groupby("InternetService_Fiber_optic").TotalCharges.transform("median")


# Now, let's look at the correlation of features.

# In[ ]:


data.corr()["Churn"].sort_values()


# The variables that converge to -1 and + 1 are the most correlated.
# 
# Tenure (inverse ratio),
# Contract_Month-to-month  (inverse ratio),
# Contract_two_year (right ratio),
# OnlineSecurity_No (right ratio)

# In[ ]:


f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# Now, we'll separate our data to test and train data. With "test_size = 0.25" parameter, specified size of test data. (it is usually recommended between 20% and 30%.)
# And after separation process, we will print shape of datas

# In[ ]:


# y = target
# x = features
y = data.Churn.values
X = data[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges',
       'TotalCharges', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One_year',
       'Contract_Two_year', 'PaymentMethod_Bank_transfer_(automatic)',
       'PaymentMethod_Credit_card_(automatic)',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=100)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# After splitting train and test data, now we'll begin to make predictive modeling now. Various machine learning models will be applied and their success will be compared.
# 
# Firstly, let's start with logistic regression.

# **PREDICTIVE MODELS**

# ** 1 - Logistic Regression**

# In[ ]:


param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty': ['l1', 'l2'] }

logreg = LogisticRegression()   # create model that is empty yet
 
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)     # cross validation. with "cv=5" parameter, specified to how many iteration step applied
logreg_cv.fit(X_train, y_train)

# now, the best value of above mentioned 'C' and 'penalty' variables will be choosen.
print(logreg_cv.best_params_)
print("best score", logreg_cv.best_score_) # best score: r**2 (1: the best, 0: the worst)


# Logistic Regression Model Fitting and Performance Metrics
logreg = LogisticRegression(C=logreg_cv.best_params_['C'], penalty=logreg_cv.best_params_['penalty'])

# to fit
a=logreg.fit(X_train, y_train)

# predict to y
y_pred = logreg.predict(X_test)
print(y_pred)


# calculate to 'accuracy'  
logreg_acc_score = round(logreg.score(X_train, y_train) * 100, 2)
print("***Logistic Regression***")
print("Accuracy Score:", logreg_acc_score)


# print to 'confusion matrix'
# TP, FN
# FP, TN
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# print to 'precision','recall', 'support', and 'f1' scores
print("Classification Report:")
print(classification_report(y_test, y_pred))


# calculate to probability and add to 'test_score' dataframe that created new
y_pred_prob_logreg = logreg.predict_proba(X_test)

pred_prob = pd.DataFrame(data=y_pred_prob_logreg, columns=["prob0_logreg","prob1_logreg"])
test_score_x = pd.DataFrame(data=X_test)
test_score_y = pd.DataFrame(data=y_test, columns=['Churn'])
test_score = pd.concat([test_score_x, test_score_y], axis=1)
test_score = pd.concat([test_score['Churn'], pred_prob["prob1_logreg"]], axis=1)



# calculate to 'roc score'
print("ROC_AUC Score:")
roc_score_logistic = roc_auc_score(y_test, y_pred)
print(roc_score_logistic)


print(logreg.coef_)                # Coefficient of the features in the decision function.
print(logreg.intercept_)           # intercept (a.k.a. bias) added to the decision function.

# Actual number of iterations for all classes. If binary or multinomial, it returns only 1 element. 
# For liblinear solver, only the maximum number of iteration across all classes is given.
print(logreg.n_iter_)              

print("MSE(mean squared error):", np.mean((y_pred-y_test)**2)) # mean squared error, MSE
print("RMSE(root mean squared error):", np.sqrt(np.mean((y_pred-y_test)**2))) # root mean squared error, RMSE


# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_logreg[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_score_logistic)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **2 - Support Vector Machine(SVM)**

# In[ ]:


param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1e-3, 1e-4] }
svm = SVC()  # create model that is empty yet
svm_cv = GridSearchCV(svm, param_grid, cv=5)  # cross validation. with "cv=5" parameter, specified to how many iteration step applied
svm_cv.fit(X, y)

# now, the best value of above mentioned 'C' and 'gamma' variables will be choosen.
print(svm_cv.best_params_)
print("best score", svm_cv.best_score_)

# SVM Model Fitting and Performance Metrics
svm = SVC(C=svm_cv.best_params_['C'], gamma=svm_cv.best_params_['gamma'], probability=True)
a=svm.fit(X_train, y_train)

# predict to y
y_pred = svm.predict(X_test)
print(y_pred)

# calculate to accuracy
svm_acc_score = round(svm.score(X_train, y_train) * 100, 2)
print("Accuracy Score:", svm_acc_score)

# print to confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# print to classification report so calculate to 'precision','recall', 'support', and 'f1' scores
print("Classification Report:")
print(classification_report(y_test, y_pred))

# calculate to probability and add to 'test_score' dataframe that created before (in logistic regression part)
y_pred_prob_svm = svm.predict_proba(X_test)
pred_prob = pd.DataFrame(data=y_pred_prob_svm, columns=["prob0_svm","prob1_svm"])
test_score = pd.concat([test_score, pred_prob["prob1_svm"]], axis=1)

# print to roc score
roc_score_svm = roc_auc_score(y_test, y_pred)
print("ROC_AUC Score:", roc_score_svm)

print(svm.support_vectors_)
print(svm.n_support_) #Number of support vectors for each class.
print(svm.intercept_) #Constants in decision function.
print(svm.fit_status_) # 0 if correctly fitted, 1 otherwise (will raise warning)
print(np.mean((y_pred-y_test)**2))

# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_svm[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_score_svm)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **3 - Decision Tree**

# In[ ]:


param_grid = {'max_depth': np.arange(1, 20)} # specify the maximum depth of the tree that will be created.
decision_tree = DecisionTreeClassifier()   # create model that is empty yet
decision_tree_cv = GridSearchCV(decision_tree, param_grid, cv=5)  # with "cv=5" parameter, specified to how many cross validation iteration
decision_tree_cv.fit(X, y)
print(decision_tree_cv.best_params_)
print("best score", decision_tree_cv.best_score_)

# Decision Tree Model Fitting and Performance Metrics
decision_tree = DecisionTreeClassifier(max_depth=decision_tree_cv.best_params_['max_depth'])
decision_tree.fit(X_train, y_train) # fitting
y_pred = decision_tree.predict(X_test) # to predict y

# calculate to accuracy
decision_tree_acc_score = round(decision_tree.score(X_train, y_train) * 100, 2) 
print("Accuracy Score:", decision_tree_acc_score)

# print to confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# calculate to probability and add to 'test_score' dataframe that created before (in logistic regression part)
y_pred_prob_dtree = decision_tree.predict_proba(X_test)
pred_prob = pd.DataFrame(data=y_pred_prob_dtree, columns=["prob0_dtree","prob1_dtree"])
test_score = pd.concat([test_score, pred_prob["prob1_dtree"]], axis=1)

# calculate to roc score
roc_score_decision_tree = roc_auc_score(y_test, y_pred)
print("ROC_AUC Score:", roc_score_decision_tree)


# importances of feature
importance_dtree = pd.DataFrame(data=decision_tree.feature_importances_, columns=['importance'])
features = pd.DataFrame(data=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges',
       'TotalCharges', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One_year',
       'Contract_Two_year', 'PaymentMethod_Bank_transfer_(automatic)',
       'PaymentMethod_Credit_card_(automatic)',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check'])

importance_dtree = pd.concat([features, importance_dtree], axis=1)
importance_dtree = importance_dtree.sort_values(by=['importance'], ascending=False)
print(importance_dtree) 

# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_dtree[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % roc_score_decision_tree)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **4 - Random Forest**

# In[ ]:


param_grid = {'n_estimators': np.arange(1, 100)}
random_forest = RandomForestClassifier()   # create model that is empty yet
random_forest_cv = GridSearchCV(random_forest, param_grid, cv=5)  #cross validation
random_forest_cv.fit(X, y)
print(random_forest_cv.best_params_)
print("best score", random_forest_cv.best_score_)

#Random Forest Model Fitting and Performance Metrics
random_forest = RandomForestClassifier(n_estimators=random_forest_cv.best_params_['n_estimators'], max_depth=6, max_features="sqrt")
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)

# calculate to accuracy
random_forest_acc_score = round(random_forest.score(X_train, y_train) * 100, 2)
print("Accuracy Score:", random_forest_acc_score)

# print to confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# calculate to probability and add to 'test_score' dataframe that created before (in logistic regression part)
y_pred_prob_rforest = random_forest.predict_proba(X_test)
pred_prob = pd.DataFrame(data=y_pred_prob_rforest, columns=["prob0_rforest","prob1_rforest"])
test_score = pd.concat([test_score, pred_prob["prob1_rforest"]], axis=1)

# calculate to roc score
roc_score_rforest = roc_auc_score(y_test, y_pred)
print("ROC_AUC Score:", roc_score_rforest)

# importances of feature
importance_rforest = pd.DataFrame(data=random_forest.feature_importances_, columns=['importance'])
importance_rforest = pd.concat([features, importance_rforest], axis=1) # 'features' dataframe was created before in decision tree part
importance_rforest = importance_rforest.sort_values(by=['importance'], ascending=False)
print(importance_rforest) 

# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rforest[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_score_rforest)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **5 - Gaussian Naive Bayes**

# In[ ]:


nb = GaussianNB()  # create model that is empty yet
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# calculate to accuracy
nb_acc_score = round(nb.score(X_train, y_train) * 100, 2)
print("Accuracy Score:", nb_acc_score)

# print to confusion matrix and classification report so calculate to 'precision','recall', 'support', and 'f1' scores
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# calculate to probability and add to 'test_score' dataframe that created before (in logistic regression part)
y_pred_prob_GaussianNB =nb.predict_proba(X_test)
pred_prob = pd.DataFrame(data=y_pred_prob_GaussianNB, columns=["prob0_GaussianNB","prob1_GaussianNB"])
test_score = pd.concat([test_score, pred_prob["prob1_GaussianNB"]], axis=1)

# calculate to roc score
roc_score_GaussianNB = roc_auc_score(y_test, y_pred)
print("ROC_AUC Score:", roc_score_GaussianNB)

# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_GaussianNB[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Gaussian Naive Bayes (area = %0.2f)' % roc_score_GaussianNB)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **6 - K-Nearest Neighbors**

# In[ ]:


param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier() # create model that is empty yet
knn_cv = GridSearchCV(knn, param_grid, cv=5) # cross validation
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)
print("best score", knn_cv.best_score_)

# KNN Model Fitting and Performance Metrics
knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# calculate to accuracy score
knn_acc_score = round(knn.score(X_train, y_train) * 100, 2)
print("Accuracy Score:", knn_acc_score)

# print to confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# calculate to probability and add to 'test_score' dataframe that created before (in logistic regression part)
y_pred_prob_knn =knn.predict_proba(X_test)
pred_prob = pd.DataFrame(data=y_pred_prob_knn, columns=["prob0_knn","prob1_knn"])
test_score = pd.concat([test_score, pred_prob["prob1_knn"]], axis=1)

# calculate to roc score
roc_score_knn = roc_auc_score(y_test, y_pred)
print("ROC_AUC Score:", roc_score_knn)

# print to roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_knn[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='KNN (area = %0.2f)' % roc_score_knn)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# **CONCLUSION AND MODELS COMPARISON  **

# We will apply six different machine laerning algorithms up to the present. Now, we will compare the success of the models.
# 
# We will evaluate to make comparison:
# - Accuracy
# - Roc Curve
# - Lift

# In[ ]:


# accuracy and roc score

models_all = pd.DataFrame(
    data=["Logistic regression", "Decision Tree", "Random Forest", "SVM", "Gaussian Naive Bayes", "KNN"], 
    columns=["Algorithms"])

accuracy_all = pd.DataFrame(
    data=[logreg_acc_score, decision_tree_acc_score, random_forest_acc_score, svm_acc_score, nb_acc_score, knn_acc_score],
    columns=["Accuracy"])

roc_all = pd.DataFrame(
    data=[roc_score_logistic, roc_score_decision_tree, roc_score_rforest, roc_score_svm, roc_score_GaussianNB, roc_score_knn], 
    columns=["ROC Score"])

comparison_models = pd.concat([models_all, accuracy_all, roc_all], axis=1)
comparison_models = comparison_models.sort_values(by=['Accuracy'], ascending=False)
comparison_models


# In[ ]:


# print roc curve

fpr_logreg, tpr_logreg, thresholds = roc_curve(y_test, y_pred_prob_logreg[:, 1])
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, y_pred_prob_svm[:, 1])
fpr_dtree, tpr_dtree, thresholds = roc_curve(y_test, y_pred_prob_dtree[:, 1])
fpr_rforest, tpr_rforest, thresholds = roc_curve(y_test, y_pred_prob_rforest[:, 1])
fpr_GaussianNB, tpr_GaussianNB, thresholds = roc_curve(y_test, y_pred_prob_GaussianNB[:, 1])
fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_pred_prob_knn[:, 1])

plt.figure(figsize=(10, 10))

plt.plot(fpr_logreg, tpr_logreg, label='Log_Reg (area = %0.2f)' % roc_score_logistic)
plt.plot(fpr_svm, tpr_svm, label='SVM (area = %0.2f)' % roc_score_svm)
plt.plot(fpr_dtree, tpr_dtree, label='Decision_Tree (area = %0.2f)' % roc_score_decision_tree)
plt.plot(fpr_rforest, tpr_rforest, label='Random_Forest (area = %0.2f)' % roc_score_rforest)
plt.plot(fpr_GaussianNB, tpr_GaussianNB, label='GaussianNB (area = %0.2f)' % roc_score_GaussianNB)
plt.plot(fpr_knn, tpr_knn, label='KNN (area = %0.2f)' % roc_score_knn)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# calculating top 10% lift


event_rate = test_score.Churn.mean()

# lift of logistic regression
lift_logregg_sort = test_score.sort_values(by=['prob1_logreg'], ascending=False).head(176).mean()
lift_logregg_mean = lift_logregg_sort.prob1_logreg.mean()
lift_logregg = lift_logregg_mean / event_rate

# lift of svm
lift_svm_sort = test_score.sort_values(by=['prob1_svm'], ascending=False).head(176).mean()
lift_svm_mean = lift_svm_sort.prob1_svm.mean()
lift_svm = lift_svm_mean / event_rate

# lift of decision tree
lift_dtree_sort = test_score.sort_values(by=['prob1_dtree'], ascending=False).head(176).mean()
lift_dtree_mean = lift_dtree_sort.prob1_dtree.mean()
lift_dtree = lift_dtree_mean / event_rate

# lift of random forest
lift_rforest_sort = test_score.sort_values(by=['prob1_rforest'], ascending=False).head(176).mean()
lift_rforest_mean = lift_rforest_sort.prob1_rforest.mean()
lift_rforest = lift_rforest_mean / event_rate

# lift of GaussianNB
lift_GaussianNB_sort = test_score.sort_values(by=['prob1_GaussianNB'], ascending=False).head(176).mean()
lift_GaussianNB_mean = lift_GaussianNB_sort.prob1_GaussianNB.mean()
lift_GaussianNB = lift_GaussianNB_mean / event_rate

# lift of knn
lift_knn_sort = test_score.sort_values(by=['prob1_knn'], ascending=False).head(176).mean()
lift_knn_mean = lift_knn_sort.prob1_knn.mean()
lift_knn = lift_knn_mean / event_rate


# print to lifts of models together and add to 'comparison_models' dataframe that created before (in "accuracy and roc score" part)
lifts = pd.DataFrame(data=[lift_logregg, lift_dtree, lift_rforest, lift_svm, lift_GaussianNB, lift_knn], columns=["Lift_top10"])
comparison_models = pd.concat([comparison_models, lifts], axis=1)

comparison_models

