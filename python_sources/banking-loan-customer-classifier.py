#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pandas_profiling as pdp
from sklearn.linear_model import LogisticRegression
pd.set_option('max_rows',1200)
pd.set_option('max_columns',1000)


# In[ ]:


cr=pd.read_csv("../input/Loan payments data.csv")


# In[ ]:


cr.head()


# In[ ]:


# EDA using pandas profiling
cr.profile_report(style={'full_width':True})


# In[ ]:





# In[ ]:


cr.fillna('0',axis=1,inplace=True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corr=cr.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


cr.sample(10)


# In[ ]:


# Printing no of records for different columns
print(cr['Gender'].value_counts())
print(cr['education'].value_counts())
print(cr['Principal'].value_counts())
print(cr['loan_status'].value_counts())
print(cr['terms'].value_counts())


# In[ ]:


# Boxplot to show loan status based on gender
plt.figure(figsize=(10,7))
sns.boxplot(data=cr,x='loan_status',y='age',hue='Gender',linewidth=2,order=['PAIDOFF','COLLECTION_PAIDOFF','COLLECTION'])
plt.show()


# In[ ]:


# Boxplot to show loan status based on education
plt.figure(figsize=(15,10))
sns.boxplot(data=cr,x='loan_status',y='age',hue='education',linewidth=2
            ,order=['PAIDOFF','COLLECTION_PAIDOFF','COLLECTION'])
#           ,order=['High School or Below','college','Bechalor','Master or Above'])
plt.show()


# In[ ]:


# Only male has 'master' degree and paidoff loan on time
plt.figure(figsize=(60,20))
sns.factorplot(data=cr,x='loan_status',y='age',hue='education',col='Gender',kind='box'
               ,order=['PAIDOFF','COLLECTION_PAIDOFF','COLLECTION'],aspect= 1.5)
plt.show()


# In[ ]:


plt.figure(figsize=(60,20))
sns.factorplot(data=cr,x='loan_status',y='age',hue='Gender',col='education',kind='box'
               ,order=['PAIDOFF','COLLECTION_PAIDOFF','COLLECTION'],aspect=1.5)
plt.show()


# In[ ]:


#Count of males/females paying loan after due dates(1,2,3,4.....etc)
# Maximum count is '59' days after due date for both male and female
pd.crosstab(cr['Gender'],cr['past_due_days'],rownames=['gender'], colnames=['Loan paidafter due date'])


# In[ ]:


# Loan status and count based on gender
# Close to 90% defaulters are 'MALE'; naturally because male comprise '85%' of total loan records
pd.crosstab(cr['Gender'],cr['loan_status'],rownames=['gender'], colnames=['loan status'])


# In[ ]:


# Less people took 7 days loan, and hence less defaulters, most defaulters have '30' day loan term
pd.crosstab(cr['terms'],cr['loan_status'],rownames=['terms'], colnames=['loan status'])


# In[ ]:


# Mostly people took '1000' principal/amount of loan and as below data shows most defaulters for this
pd.crosstab(cr['Principal'],cr['loan_status'],rownames=['principal'],colnames=['loan status'])


# In[ ]:


# High school and college ones are among most defaulters
pd.crosstab(cr['education'],cr['loan_status'],rownames=['education'],colnames=['loan status'])


# In[ ]:


# Various education levels for the loan takers based on gender
pd.crosstab(cr['education'],cr['Gender'],rownames=['education'],colnames=['gender'])


# In[ ]:


# Below tab shows in which age(26-30 for men; 26-35 for women) maximum loans are taken
pd.crosstab(cr['Gender'],cr['age'],rownames=['gender'],colnames=['age'])


# In[ ]:


cr.head().append(cr.tail()).append(cr.sample(10))


# In[ ]:



# DummyCoding : Replacing 'loan status' with numerical values
cr['loan_status'].replace('PAIDOFF',0,inplace=True)
cr['loan_status'].replace('COLLECTION_PAIDOFF',1,inplace=True)
cr['loan_status'].replace('COLLECTION',2,inplace=True)
cr.sample(20)


# In[ ]:



education_dummies=pd.get_dummies(cr.education,prefix='education')
education_dummies.sample(4)


# In[ ]:


education_dummies.drop(education_dummies.columns[0],axis=1,inplace=True)


# In[ ]:


education_dummies.head(5)


# In[ ]:


cr=pd.concat([cr,education_dummies],axis=1)
cr.head(5)


# In[ ]:


cr.drop(cr.columns[9],axis=1,inplace=True)
cr.sample(15)


# In[ ]:


# Dummification for gender column
gender_dummies=pd.get_dummies(cr['Gender'],prefix='gender')
gender_dummies.drop(gender_dummies.columns[0],axis=1,inplace=True)
gender_dummies.head()


# In[ ]:



cr=pd.concat([cr,gender_dummies],axis=1)
cr.sample(10)


# In[ ]:


cr.sample(10)


# Modeling

# In[ ]:



# Assigning target variable to y
y=cr['loan_status']


# In[ ]:


# Assigning input variables to X
cols=['Principal','terms','past_due_days','age','education_Master or Above'
         ,'education_college','gender_male']
x=cr[list(cols)].values
x


# In[ ]:


#splitting the data into train and test with 70:30 ratio
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.30)


# In[ ]:


#calling logistic regression
from sklearn import metrics
logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


y_pred=logreg.predict(X_test)


# In[ ]:


# Accuracy score for Logistic regression
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


#creating a confusion matrix to understand the classification
conf = metrics.confusion_matrix(y_pred,y_test)
conf


# In[ ]:


# invoking Support Vector machines(svm)
from sklearn import svm
clf = svm.SVC()
svc_model = clf.fit(X_train,y_train)


# In[ ]:


print(svc_model)


# In[ ]:


target_names=['paidoff','collection_paidoff','collection']
print(metrics.classification_report(y_pred,y_test,target_names=target_names))


# In[ ]:


sv_pred=svc_model.predict(X_test)
print(accuracy_score(sv_pred,y_test))
print(confusion_matrix(sv_pred,y_test))
print(classification_report(sv_pred,y_test))


# **Model: Random Forest Classifier**

# In[ ]:


# Applying decision tree algo
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
tree1=tree.DecisionTreeClassifier()
dt_model=tree1.fit(X_train,y_train)


# In[ ]:


print(dt_model)


# In[ ]:


print(dt_model.feature_importances_)
print(dt_model.score(X_train,y_train))


# In[ ]:


# Accuracy score for Decision tree algo
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
pred=dt_model.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


# In[ ]:


# Grid Search on DT
from sklearn.model_selection import GridSearchCV
dt_params={'criterion':['gini','entropy'],'max_depth':range(3,8),'min_samples_split':range(2,6), 'min_samples_leaf':range(1,3)}
tree2=tree.DecisionTreeClassifier()
dt_grid=GridSearchCV(tree2,dt_params,cv=5)
dec_tree=dt_grid.fit(X_train,y_train)


# Model Optimisation

# In[ ]:


dt_grid.best_params_


# In[ ]:


pred=dec_tree.predict(X_test)
print(accuracy_score(pred,y_test))
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))


# # It seems Decision Tree is giving better results when compared to Logistic Regression and SVM
