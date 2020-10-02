#!/usr/bin/env python
# coding: utf-8

# ## Comprehensive data analysis and predictive modeling with IBM HR Analytics data set . 
# 
# ### Extensive hyperparameter tuning for RandomForest and XGBoost

# ### Introduction
# 
# The data set was obtained from Kaggle.
#   [IBM HR Analytics Data Set](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)  
#   The data set was uploaded by [Pavan Subhash](https://www.kaggle.com/pavansubhasht)  
#   
#   This is a fictional data set created by IBM data scientists.
#   Such a data set will help us uncover the factors that lead to an employee leaving the company.  
#   This is binary classification problem, and I have used Logistic Regression, RandomForest, and XGBoost to build predictive  models.
#   Extensive hyperparameter tuning has been performed for all the models.  
#   More there is class imbalance in the data set.  
#   I have used SMOTE ( oversampling the minority class), and also created a balanced data set that contained equal number of 1s and 0s. The The predictive models were built separately on these datasets.  
#   
# ### Acknowledgement
# 
#   Initially I studied the notebook [Employee attrition via Ensemble tree-based methods](https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods) by [Anisotropic](https://www.kaggle.com/arthurtok)

# In[ ]:


get_ipython().system(' pip install imbalanced-learn')
get_ipython().system(' pip install xgboost')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[ ]:


hr_data=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


hr_data.head()


# In[ ]:


hr_data.info()


# In[ ]:


num=hr_data.select_dtypes(include=['number']).columns.values


# In[ ]:


num


# In[ ]:


num=num.tolist()


# In[ ]:


type(num)


# In[ ]:


num


# In[ ]:


num=[var for var in num if var not in ['EmployeeNumber']]


# In[ ]:


num


# In[ ]:


cat=hr_data.select_dtypes(exclude=['number']).columns.values


# In[ ]:


cat


# In[ ]:


cat=cat.tolist()


# In[ ]:


cat.append('EmployeeNumber')


# In[ ]:


type(cat)


# In[ ]:


cat


# In[ ]:


#Checking for null values
hr_data.isnull().any()


# In[ ]:


# No null values..thats great..


# In[ ]:


# Inspecting the taerget variable ...
hr_data['Attrition'].value_counts(normalize=True).plot(kind='pie',autopct='%1.1f%%')


# In[ ]:


#Class imbalance exists


# #### Univariate Analysis

# In[ ]:


len(hr_data['DailyRate'].unique())


# In[ ]:


sns.distplot(hr_data['DailyRate'])


# In[ ]:


sns.distplot(hr_data['HourlyRate'])


# In[ ]:


sns.distplot(hr_data['MonthlyIncome'])


# In[ ]:


# For some numeric variables we will use the countplot, while for some others we will use the distplot..


# In[ ]:


disc=[]
num_nd=[]
for var in num :
    print(f'Variable : {var}, No. of UniqueValues: {len(hr_data[var].unique())}')
    if len(hr_data[var].unique()) <= 20 :
       disc.append(var)
    else :
       num_nd.append(var)    


# In[ ]:


num_nd


# In[ ]:


disc


# In[ ]:


disc.remove('EmployeeCount')
hr_data.drop(columns=['EmployeeCount'],inplace=True)


# In[ ]:


disc


# In[ ]:


# Plotting the numerical vars..ie vars deemed to be numerical....
num_f=pd.melt(hr_data,value_vars=sorted(num_nd))
g=sns.FacetGrid(num_f,col='variable',sharex=False,sharey=False,col_wrap=4)
g=g.map(sns.distplot,'value')
[plt.setp(ax.get_xticklabels(),rotation=60) for ax in g.axes.flat]
plt.tight_layout()
plt.show()


# In[ ]:


# Plotting the discrete vars..ie vars deemed to be discrete....
disc_f=pd.melt(hr_data,value_vars=sorted(disc))
g=sns.FacetGrid(disc_f,col='variable',sharex=False,sharey=False,col_wrap=4)
g=g.map(sns.countplot,'value')
[plt.setp(ax.get_xticklabels(),rotation=60) for ax in g.axes.flat]
plt.tight_layout()
plt.show()


# In[ ]:


print(len(hr_data['StandardHours'].unique()))


# In[ ]:


#Dropping the variable standard hours as it has only 1 value
hr_data.drop(columns=['StandardHours'],inplace=True)


# In[ ]:


disc.remove('StandardHours')


# In[ ]:


cat.remove('EmployeeNumber')
cat.remove('Attrition')


# In[ ]:


cat_f=pd.melt(hr_data,value_vars=sorted(cat))
g=sns.FacetGrid(cat_f,col='variable',sharex=False,sharey=False,col_wrap=2)
g=g.map(sns.countplot,'value')
[plt.setp(ax.get_xticklabels(),rotation=60) for ax in g.axes.flat]
plt.tight_layout()
plt.show()


# In[ ]:


# The variable OverTime has only one value ..'Yes'...
len(hr_data['Over18'].unique())


# In[ ]:


hr_data.drop(columns=['Over18'],inplace=True)


# In[ ]:


cat.remove('Over18')


# In[ ]:


# Inspecting rare values in categorical variables..
for var in cat :
    print(var)
    print()
    print(hr_data[var].value_counts(normalize=True))
    print()


# In[ ]:


# Rare values are present in JobRole,EducationField, and Department..only Human Resources is the 'Rare' value in these fields...


# #### Bivariate Analysis

# In[ ]:


# Exploring correlation among numerical variables
corr_mat=hr_data[num_nd].corr()
fig=plt.figure(figsize=(20,10))
fig.add_subplot(111)
cmap=sns.diverging_palette(200,10,as_cmap=True)
sns.heatmap(corr_mat,xticklabels=corr_mat.columns.values,yticklabels=corr_mat.columns.values,cmap=cmap,annot=True)
plt.show()


# In[ ]:


corr_mat=hr_data[disc].corr()
fig=plt.figure(figsize=(20,10))
fig.add_subplot(111)
cmap=sns.diverging_palette(200,10,as_cmap=True)
sns.heatmap(corr_mat,xticklabels=corr_mat.columns.values,yticklabels=corr_mat.columns.values,cmap=cmap,annot=True)
plt.show()


# In[ ]:


# As seen from the above plots some features are correlated


# In[ ]:


hr_data.Attrition.replace({'Yes':1,'No':0},inplace=True)


# In[ ]:


hr_data[disc].corrwith(hr_data['Attrition']).plot(kind='bar',title='Correlation of discrete variables with Attrition',figsize=(15,10))


# In[ ]:


# Some variables strongly affect Attrition...For example if en employee is comfortable with his/her manager (YearsWithCurrManager is more), then he/she is likely to stay back.


# In[ ]:


pd.crosstab(index=hr_data['Attrition'],columns=hr_data['BusinessTravel'])


# In[ ]:


from scipy.stats import chi2_contingency


# In[ ]:


# We will use chisq tests to determine which categorical variables are significant


# In[ ]:


for var in cat :
    chisq=chi2_contingency(pd.crosstab(index=hr_data['Attrition'],columns=hr_data[var]))
    if chisq[1] < 0.05 :
        print(f'{var} is significant for Attrition ')
        print(f'p value {chisq[1]}')
        print()
    else :
        print(f'{var} is not significant for Attrition ')
        print(f'p value {chisq[1]}')
        print()      


# In[ ]:


# Looks like all the categorical variables are significant


# In[ ]:


# Encoding the categorical variables
hr_data=pd.get_dummies(hr_data)


# In[ ]:


hr_data.info()


# In[ ]:


# Avoiding the dummy variable trap...
hr_data.drop(columns=['BusinessTravel_Non-Travel','JobRole_Human Resources','MaritalStatus_Divorced','OverTime_Yes','Gender_Female','EducationField_Human Resources','Department_Human Resources'],inplace=True)


# In[ ]:


hr_data['Attrition'].head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(hr_data[hr_data.columns.difference(['EmployeeNumber','Attrition'])],hr_data['Attrition'],test_size=0.3,random_state=10)


# In[ ]:


# Tackling class imbalance using SMOTE
oversampling=SMOTE(random_state=0)


# In[ ]:


SMOTE_train,SMOTE_target=oversampling.fit_sample(X_train,y_train)


# In[ ]:


SMOTE_train=pd.DataFrame(SMOTE_train,columns=X_train.columns)


# #### Predictive Models

# ##### Applying Logistic Regression ( with both oversampling and undersampling)

# ###### We will use SMOTE to oversample the minority class. We will also create another data set that will downsample the majority class such that the no. of samples for both classes are equal

# ##### Logistic Regression with SMOTE 

# In[ ]:


#Applying Logistic Regression
from sklearn.metrics import classification_report,roc_curve,auc,accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,cross_validate


# In[ ]:


parameter_grid={'C':[0.0001,0.001,0.01,.1,1,10],
               'penalty':['l1','l2']}
log_reg=LogisticRegression()


# In[ ]:


classifier_logreg=GridSearchCV(estimator=log_reg,param_grid=parameter_grid,n_jobs=-1,scoring='roc_auc',cv=10)


# In[ ]:


classifier_logreg.fit(SMOTE_train,SMOTE_target)


# In[ ]:


classifier_logreg.best_params_


# In[ ]:


label_pred_logit=classifier_logreg.best_estimator_.predict(X_test)


# In[ ]:


fpr,tpr,thresh=roc_curve(y_test,label_pred_logit)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


cls_report_logreg=classification_report(y_test,label_pred_logit)


# In[ ]:


print(cls_report_logreg)


# In[ ]:


#Though we have obtained 85% accuracy, the recall,and precision scores are quite less where Attrition=1...We can downsample the majority class, and create a sample that has equal number of 1 s and zeroes.


# In[ ]:


# Or we can put a cost on misclassifying a 1 as a 0....


# ##### Logistic Regression after downsampling ( creating a balanced data set)

# In[ ]:


# Creating a balanced data set..


# In[ ]:


hr_att=hr_data[hr_data.Attrition==1]


# In[ ]:


hr_no_att=hr_data[hr_data.Attrition==0]


# In[ ]:


hr_att.info()


# In[ ]:


hr_no_att.info()


# In[ ]:


hr_no_att=hr_no_att.sample(hr_att.shape[0],random_state=10)


# In[ ]:


hr_balanced=pd.concat([hr_no_att,hr_att])


# In[ ]:


hr_balanced.head()


# In[ ]:


hr_balanced=hr_balanced.sample(frac=1,random_state=10)


# In[ ]:


hr_balanced.head()


# In[ ]:


hr_bl_train,hr_bl__test,target_train,target_test=train_test_split(hr_balanced[hr_balanced.columns.difference(['EmployeeNumber','Attrition'])],hr_balanced['Attrition'],test_size=0.3,random_state=10)


# In[ ]:


#Applying Logistic Regression
parameter_grid={'C':[0.0001,0.001,0.01,.1,1,10],
               'penalty':['l1','l2']}
log_reg=LogisticRegression()

classifier_logreg=GridSearchCV(estimator=log_reg,param_grid=parameter_grid,n_jobs=-1,scoring='accuracy',cv=10)

classifier_logreg.fit(hr_bl_train,target_train)


# In[ ]:


classifier_logreg.best_params_


# In[ ]:


logreg_predict=classifier_logreg.best_estimator_.predict(hr_bl__test)


# In[ ]:


fpr,tpr,thresh=roc_curve(target_test,logreg_predict)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


print(classification_report(target_test,logreg_predict))


# In[ ]:


# The accuracy has come down....but area under the curve has improved so has the recall , precision for Attrition=1


# In[ ]:


#For these class imbalanced problems for Employee Attrition ...a penalty should be imposed on classifying a 1 (Attrition ) as 0 (No Attrition)


# ##### Random Forests with balanced data set ( majority class downsampled)

# In[ ]:


import time
import numpy as np


# In[ ]:


parameter_grid={'n_estimators':np.arange(200,600,100),
                'min_samples_leaf':[2,5,10],
                'min_samples_split':[2,5,10],
               }

print(parameter_grid)
rf=RandomForestClassifier(n_jobs=-1,random_state=10)
classifier_rf=GridSearchCV( estimator=rf,
                     param_grid=parameter_grid,
                          scoring='accuracy',
                          cv=5) 
                         
t1=time.time()
classifier_rf.fit(hr_bl_train,target_train)
t2=time.time()
print(t2-t1)


# In[ ]:


classifier_rf.best_params_


# In[ ]:


parameter_grid={'n_estimators':[350,400,450,1000],
                'min_samples_leaf':[2,5,10],
                'min_samples_split':[2,5,10],
               }

print(parameter_grid)
rf=RandomForestClassifier(n_jobs=-1,random_state=10)
classifier_rf=GridSearchCV( estimator=rf,
                     param_grid=parameter_grid,
                          scoring='accuracy',
                          cv=5) 
                         
t1=time.time()
classifier_rf.fit(hr_bl_train,target_train)
t2=time.time()
print(t2-t1)


# In[ ]:


classifier_rf.best_params_


# In[ ]:


#Tuning Max_depth
rf=RandomForestClassifier(n_estimators=350,min_samples_leaf=2,min_samples_split=10,n_jobs=-1,random_state=10)
max_d=np.arange(10,110,10)
#max_d.append(None)
parameter_grid={'max_depth':max_d}
classifier_rf=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5,verbose=1)  

t1=time.time()
classifier_rf.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf.best_params_


# In[ ]:


#Honing in on Max Depth
rf=RandomForestClassifier(n_estimators=350,min_samples_leaf=2,min_samples_split=10,n_jobs=-1,random_state=10)
max_d=[10]
max_d.append(None)
parameter_grid={'max_depth':max_d}
classifier_rf=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5,verbose=1)  

t1=time.time()
classifier_rf.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf.best_params_


# In[ ]:


parameter_grid={'bootstrap':[True,False],
               'max_features':['auto','sqrt'],
               'criterion':['gini','entropy']}
rf=RandomForestClassifier(n_estimators=350, n_jobs=-1,min_samples_leaf=2, min_samples_split=10,max_depth=10,random_state=10)
classifier_rf=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5)

t1=time.time()
classifier_rf.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf.best_params_


# In[ ]:


rf_pred=classifier_rf.best_estimator_.predict(hr_bl__test)


# In[ ]:


fpr,tpr,thresh=roc_curve(target_test,rf_pred)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


print(classification_report(target_test,rf_pred))


# In[ ]:


# Logistic Regression outdoes RandomForest on the balanced data set!!


# ##### RandomForest with SMOTEd data 

# In[ ]:


# Trying Random Forest on SMOTEd data set
parameter_grid={'n_estimators':np.arange(200,600,100),
                'min_samples_leaf':[2,5,10],
                'min_samples_split':[2,5,10],
               }

print(parameter_grid)
rf=RandomForestClassifier(n_jobs=-1,random_state=0)
classifier_rf_smote=GridSearchCV( estimator=rf,
                     param_grid=parameter_grid,
                          scoring='accuracy',
                          cv=5) 
                         
t1=time.time()
classifier_rf_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()
print(t2-t1)


# In[ ]:


classifier_rf_smote.best_params_


# In[ ]:


parameter_grid={'n_estimators':[250,300,350,1000],
                'min_samples_leaf':[2,5,10],
                'min_samples_split':[2,5,10],
               }

print(parameter_grid)
rf=RandomForestClassifier(n_jobs=-1,random_state=0)
classifier_rf_smote=GridSearchCV( estimator=rf,
                     param_grid=parameter_grid,
                          scoring='accuracy',
                          cv=5) 
                         
t1=time.time()
classifier_rf_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()
print(t2-t1)


# In[ ]:


classifier_rf_smote.best_params_


# In[ ]:


#Tuning Max_depth
rf=RandomForestClassifier(n_estimators=1000,min_samples_leaf=2,min_samples_split=2,n_jobs=-1,random_state=10)
max_d=np.arange(10,110,10)
#max_d.append(None)
parameter_grid={'max_depth':max_d}
classifier_rf_smote=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5,verbose=1)  

t1=time.time()
classifier_rf_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf_smote.best_params_


# In[ ]:


rf=RandomForestClassifier(n_estimators=1000,min_samples_leaf=2,min_samples_split=2,n_jobs=-1,random_state=10)
max_d=[20]
max_d.append(None)
parameter_grid={'max_depth':max_d}
classifier_rf_smote=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5,verbose=1)  

t1=time.time()
classifier_rf_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf_smote.best_params_


# In[ ]:


parameter_grid={'bootstrap':[True,False],
               'max_features':['auto','sqrt'],
               'criterion':['gini','entropy']}
rf=RandomForestClassifier(n_estimators=1000, n_jobs=-1,min_samples_leaf=2, min_samples_split=2,max_depth=20,random_state=10)
classifier_rf_smote=GridSearchCV(estimator=rf,param_grid=parameter_grid,scoring='accuracy',cv=5)

t1=time.time()
classifier_rf_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


classifier_rf_smote.best_params_


# In[ ]:


rf_pred_smote=classifier_rf_smote.best_estimator_.predict(X_test)


# In[ ]:


fpr,tpr,thresh=roc_curve(y_test,rf_pred_smote)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


print(classification_report(y_test,rf_pred_smote))


# In[ ]:


# We see that the accuracy is good, but ROC is reduced to 0.61...moreover recall is very low (.32) when Attrition occurs (target=1)


# ###### XGBOOST with balanced data

# In[ ]:


xgb_model=xgboost.XGBClassifier(n_estimators=1000,objective='binary:logistic')


# In[ ]:


#Starting tuning with max_depth and min_child_weight
params=dict(max_depth=np.arange(3,11,2),min_child_weight=np.arange(5,11,2))
xgb_classifier=GridSearchCV(estimator=xgb_model, param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier.best_params_


# In[ ]:


#Honing in...
params=dict(max_depth=[3,4,6,8],min_child_weight=[6,8,9,10])
xgb_classifier=GridSearchCV(estimator=xgb_model, param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier.best_params_


# In[ ]:


xgb_classifier.cv_results_['params']


# In[ ]:


# Tuning colsample_bytree, and subsample...
params=dict(colsample_bytree=np.arange(0.3,1,.1),subsample=np.arange(0.8,1,.1))
print(params)
xgb_classifier=GridSearchCV(estimator=xgboost.XGBClassifier(n_estimators=1000,min_child_weight=9,max_depth=3), param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier.best_params_


# In[ ]:


# Now, let us tune the learning rate
params=dict(learning_rate=[0.3,0.2,0.1,0.05,0.01])
print(params)
xgb_classifier=GridSearchCV(estimator=xgboost.XGBClassifier(n_estimators=1000,min_child_weight=9,max_depth=3,colsample_bytree=0.7,subsample=0.9), param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier.fit(hr_bl_train,target_train)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier.best_params_


# In[ ]:


xgb_classifier.best_estimator_


# In[ ]:


xgb_pred=xgb_classifier.best_estimator_.predict(hr_bl__test)


# In[ ]:


fpr,tpr,thresh=roc_curve(target_test,xgb_pred)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


print(classification_report(target_test,xgb_pred))


# In[ ]:


# For the balanced data set ,Logistic Regression is still the best, followed by XGBoost, and then Random Forest


# ##### XGBoost on the SMOTEd data

# In[ ]:


xgb_model=xgboost.XGBClassifier(n_estimators=1000,objective='binary:logistic')


# In[ ]:


#Starting tuning with max_depth and min_child_weight
params=dict(max_depth=np.arange(3,11,2),min_child_weight=np.arange(5,11,2))
xgb_classifier_smote=GridSearchCV(estimator=xgb_model, param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier_smote.best_params_


# In[ ]:


#Honing in...
params=dict(max_depth=[3,4,5,6,8],min_child_weight=[5,6,8,9,10])
xgb_classifier_smote=GridSearchCV(estimator=xgb_model, param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier_smote.best_params_


# In[ ]:


# Tuning colsample_bytree, and subsample...
params=dict(colsample_bytree=np.arange(0.3,1,.1),subsample=np.arange(0.8,1,.1))
print(params)
xgb_classifier_smote=GridSearchCV(estimator=xgboost.XGBClassifier(n_estimators=1000,min_child_weight=5,max_depth=5), param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier_smote.best_params_


# In[ ]:


# Now, let us tune the learning rate
params=dict(learning_rate=[0.3,0.2,0.1,0.05,0.01])
print(params)
xgb_classifier_smote=GridSearchCV(estimator=xgboost.XGBClassifier(n_estimators=1000,min_child_weight=5,max_depth=5,colsample_bytree=0.9,subsample=0.9), param_grid=params, n_jobs=-1, 
                   cv=5, 
                   scoring='accuracy',
                   verbose=1)
t1=time.time()
xgb_classifier_smote.fit(SMOTE_train,SMOTE_target)
t2=time.time()

print(t2-t1)


# In[ ]:


xgb_classifier_smote.best_params_


# In[ ]:


xgb_pred_smote=xgb_classifier_smote.predict(X_test)


# In[ ]:


fpr,tpr,thresh=roc_curve(y_test,xgb_pred_smote)


# In[ ]:


auc(fpr,tpr)


# In[ ]:


print(classification_report(y_test,xgb_pred_smote))


# In[ ]:


## As again with SMOTEd data sets, the accuracy is high..but the recall is very low for cases where Attrition is 1.....


# ##### Selecting important features based on the Logistic Regression model ( that was built using the balanced data set)

# In[ ]:


#We will select the important features from the Logistic regression model on the balanced data set..
from sklearn.feature_selection import RFE


# In[ ]:


rfe=RFE(estimator=LogisticRegression(C=10,penalty='l2'),n_features_to_select=10)


# In[ ]:


rfe_fit=rfe.fit(hr_data[hr_data.columns.difference(['EmployeeNumber','Attrition'])],hr_data['Attrition'])


# In[ ]:


hr_data.columns.difference(['EmployeeNumber','Attrition'])[rfe_fit.support_].values


# ### Conclusion  
# 
# We saw that though the SMOTE data set had a higher accuracy , the balanced data set had better 'recall' statistics for the 1s..ie,where Attrition occurs. This is true for all the models.  
# 
# LogisticRegression on the balanced data set had the best recall statistics (86%)

# In[ ]:




