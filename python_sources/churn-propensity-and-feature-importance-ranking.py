#!/usr/bin/env python
# coding: utf-8

# #                                       Telco Customer Churn
# #                       Focused customer retention programs

# ## OBJECTIVE: Predict churn to retain customers. 
# 1. Calculation of Churn Probability and ranking of CustomerIds based on the Prob(Churn)
# 2. Ranking of Features

# # Data Pre-Processing

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Get the data

# In[ ]:


data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# # Content Analysis

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()                                                  ## data type of each column, missing values, shape of table..

SUMMARY:

1. There are total 7043 Rows and 21 Columns.

2. There are 4 Numeric and 17 Non Numeric type variables/columns.

3. TotalCharges is being treated as Non Numeric. Thus it need be converted into numeric.
# #### Convert TotalCharges column to numeric

# In[ ]:


data.TotalCharges=pd.to_numeric(data.TotalCharges,errors='coerce')


# # Univariate Analysis

# ### Univariate Analysis for Non-Numeric/Categorical type Variables

# In[ ]:


data.describe(include=[np.object])


# #### What are the levels and its distribution within each Categorical Column

# In[ ]:


col_names=list(data.columns)


# In[ ]:


col_names.remove('customerID')


# In[ ]:


col_names.remove('tenure')
col_names.remove('MonthlyCharges')
col_names.remove('TotalCharges')


# In[ ]:


col_names


# In[ ]:


for i in col_names:
    j=data[i].value_counts()
    print('-----------------------------------')
    print(j)


# In[ ]:


for m in col_names:
    data[m].hist()
    plt.show()


# ### Univariate Analysis of the Numeric type Variables

# In[ ]:


data.describe(include=[np.number])


# # Missing Value Treatment

# #### Where are the missing value??

# In[ ]:


data.info()                                     ## Check the Missing Value


# In[ ]:


data.isnull().sum()                               ## Check the number missing value

Column TotalCharges has 11 Missing Values
# #### Replace /Impute the Missing Value.

# In[ ]:


## Calculate the median of the column

q=data.TotalCharges.quantile([0.1,0.5,0.9])


# In[ ]:


type(q)                                                                                 ## one Dimensional labelled Array


# In[ ]:


q


# In[ ]:


TC_median=q[.5]


# In[ ]:


TC_median


# In[ ]:


#data.loc[null_value].index             ## Indexes of the Missing Values


# In[ ]:


column_names=list(data.columns)
column_names


# In[ ]:


column_names[18:20]


# In[ ]:


plt.scatter(data.MonthlyCharges,data.TotalCharges, alpha=0.1)
plt.xlabel(column_names[18])
plt.ylabel(column_names[19])


# In[ ]:


plt.scatter(data.tenure,data.TotalCharges, alpha=0.01)
plt.xlabel(column_names[5])
plt.ylabel(column_names[19])


# #### Replace the missing Value with Median

# In[ ]:


data.TotalCharges =  data.TotalCharges.fillna(TC_median)           


# In[ ]:


data.info()


# # OUTLIER Treatment

# In[ ]:


data.boxplot(column=['MonthlyCharges','tenure'])


# In[ ]:


data.boxplot(column='TotalCharges')

## there are no outliers.
# In[ ]:


sns.kdeplot(data.MonthlyCharges)


# ## Correlation Analysis

# In[ ]:


print(data[['MonthlyCharges','TotalCharges','tenure']].corr())

 
1. Tenure and Total Charges are highly Correlated therefore I will drop Total Charges and keep Tenure.
2. MonthlyCharges and TotalCharges were also highly correlated but we have dropped the TotalCharges, thus now only tenure and 
   MonthlyCharges are left as Numeric Variables. Both of the remaining variables are not highly correlated.
# In[ ]:


print(data.corr())


# ## Create Dummy Variables

# In[ ]:


data_copy=data
data_copy=data_copy.drop(columns=['customerID', 'TotalCharges'])


# In[ ]:


data_dummy=pd.get_dummies(data_copy,drop_first=True)


# In[ ]:


len(data_dummy.columns)


# In[ ]:


data_dummy.head()


# # Building a Predictive Model

# #### PREDICTORS

# In[ ]:


X=data_dummy.iloc[:,0:29]


# #### TARGET VARIABLE

# In[ ]:


y=data_dummy.iloc[:,29]


# ### Test Train Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc=StandardScaler()


# In[ ]:


X_train=sc.fit_transform(X_train)


# In[ ]:


X_test=sc.transform(X_test)


# ### Create the model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train, y_train)


# ### Check for Accuracy

# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


y_pred1 = logreg.predict(X_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))


# ### K Fold Cross Validation

# In[ ]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[ ]:


results.mean()


# In[ ]:


results.std()


# ### ROC Curve

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))


# In[ ]:


logit_roc_auc


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ### Precision and Recall

# In[ ]:


from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# In[ ]:


print('recall score = ',recall_score(y_test,y_pred))
print('precision score = ',precision_score(y_test,y_pred))


# ##### Classification Report on Test Set

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ##### Classification Report on Training Set

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred1))


# #### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# ### HyperParameter Tuning using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Create logistic regression instance
logistic = LogisticRegression()


# In[ ]:


# Regularization penalty space
penalty = ['l1', 'l2']

# Regularization hyperparameter space
C = np.logspace(0, 4, 10)


# In[ ]:


# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)


# In[ ]:


# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)


# In[ ]:


# Fit grid search
best_model = clf.fit(X_train, y_train)


# In[ ]:


print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[ ]:


y_pred_GCV = best_model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(best_model.score(X_test, y_test)))


# In[ ]:


y_pred_GCV = best_model.predict(X_train)
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(best_model.score(X_train, y_train)))


# ## Feature Selection based on Random Forest and Recursive Feature Elimination

# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Create random forest classifer object that uses entropy
rfc = RandomForestClassifier(criterion='entropy', random_state=0, n_jobs=-1,n_estimators=200,max_depth=11)

# Train model
rfc_model = rfc.fit(X_train, y_train)
              
# Predict    
y_pred_rfc=rfc_model.predict(X_test)


# In[ ]:


print('Accuracy of random forest classifier on test set: {:.2f}'.format(rfc_model.score(X_test, y_test)))


# In[ ]:


print(classification_report(y_test,y_pred_rfc))


# In[ ]:


# Create a series with feature importance 

rfc_model.feature_importances_


# In[ ]:


rfc_imp=list(rfc_model.feature_importances_)


# In[ ]:


rfc_colname=list(X.columns)


# In[ ]:


rfc_dict={'Column_Names_rfc':rfc_colname,'feature_imp_rfc':rfc_imp}


# In[ ]:


rfc_feature_imp=pd.DataFrame(rfc_dict)


# In[ ]:


rfc_feature_rank=rfc_feature_imp.sort_values(by='feature_imp_rfc',ascending = False)


# In[ ]:


rfc_feature_rank


# ## RFE Recursive Feature Elimination

# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


model_rfe=LogisticRegression()


# In[ ]:


rfe=RFE(model_rfe,1)


# In[ ]:


rfe_fit=rfe.fit(X_train,y_train)


# In[ ]:


rfe_fit.n_features_


# In[ ]:


rfe_fit.ranking_


# In[ ]:


rank=list(rfe_fit.ranking_)


# In[ ]:


X.columns


# In[ ]:


col_nm=list(X.columns)


# In[ ]:


dict_rank={'Column_Name': col_nm,'Ranking':rank}


# In[ ]:


df_rank=pd.DataFrame(dict_rank)


# #### Ranking of Predictor Variables Based on their importance in predicting the Churn

# In[ ]:


df_rank.sort_values('Ranking')


# ## Churn Probability

# In[ ]:


y_pred_list=list(y_pred)


# In[ ]:


y_prob=logreg.predict_proba(X_test)


# In[ ]:


y_prob_list=list(y_prob)


# In[ ]:


pd.DataFrame(y_prob_list,columns=['No_Churn','Churn']).sort_values(by='Churn', ascending=False).head(20)

