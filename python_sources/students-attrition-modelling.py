#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_excel("../input/Student.xlsx")


# In[ ]:


data.head()


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


data.isna().sum()


# In[ ]:


# %age of missing values
((data.isnull().sum()/data.shape[0])*100).round(2)


# In[ ]:


#Student Background
data['STDNT_BACKGROUND'].unique()


# In[ ]:


data['STDNT_MAJOR'].unique()


# In[ ]:


data['STDNT_MINOR'].unique()


# In[ ]:


data['STDNT_TEST_ENTRANCE_COMB'].describe()


# In[ ]:


data['STDNT_TEST_ENTRANCE_COMB'].hist(bins=25,color='red')
plt.title("Student_Entrance_Combination",size=20, color='red')


# In[ ]:


data.boxplot(column='STDNT_TEST_ENTRANCE_COMB')


# In[ ]:


data['STDNT_TEST_ENTRANCE_COMB'].mode()


# In[ ]:


#Missing values are replaced by mode value, as it is normally distributed
data['STDNT_TEST_ENTRANCE_COMB']=data['STDNT_TEST_ENTRANCE_COMB'].fillna(950.0)


# In[ ]:


data['FIRST_TERM'].value_counts()


# In[ ]:


data['SECOND_TERM'].value_counts()


# In[ ]:


data['FIRST_TERM']=data['FIRST_TERM'].replace({200508:2005, 200608:2006, 200708:2007, 200808:2008, 200908:2009, 201008:2010})


# In[ ]:


data['SECOND_TERM']=data['SECOND_TERM'].replace({200602:2006, 200702:2007, 200802:2008, 200902:2009, 201002:2010, 201102:2011})


# In[ ]:


data['HOUSING_STS'].unique()


# In[ ]:


data['RETURNED_2ND_YR'].value_counts()


# In[ ]:


data['DISTANCE_FROM_HOME'].describe()


# In[ ]:


data['DISTANCE_FROM_HOME'].isnull().sum()


# In[ ]:


#Replacing missing values for DISTANCE_FROM_HOME by grouping HIGH_SCHL_NAME
data['DISTANCE_FROM_HOME'] = data['DISTANCE_FROM_HOME'].fillna(data.groupby('HIGH_SCHL_NAME')['DISTANCE_FROM_HOME'].transform('mean'))


# In[ ]:


#Still 2 rows contain NaN values, replacing them by total mean
data['DISTANCE_FROM_HOME'] = data['DISTANCE_FROM_HOME'].fillna(data['DISTANCE_FROM_HOME'].mean())


# In[ ]:


data['HIGH_SCHL_GPA'].isnull().sum()


# In[ ]:


data.boxplot(column='HIGH_SCHL_GPA')


# In[ ]:


data['HIGH_SCHL_GPA'].describe()


# In[ ]:


#Replacing HIGH_SCHL_GPA with mean values
data['HIGH_SCHL_GPA']=data['HIGH_SCHL_GPA'].fillna(data['HIGH_SCHL_GPA'].mean())


# In[ ]:


#HIGH_SCHL_NAME contains 1 row with NaN value, hence removing the record
data[data['HIGH_SCHL_NAME'].isnull()]


# In[ ]:


data = data[pd.notnull(data['HIGH_SCHL_NAME'])]


# In[ ]:


data['FATHER_HI_EDU_CD'].value_counts()


# In[ ]:


data['MOTHER_HI_EDU_CD'].value_counts()


# In[ ]:


data['FATHER_HI_EDU_DESC'].value_counts(dropna=False)


# In[ ]:


data['MOTHER_HI_EDU_DESC'].value_counts()


# In[ ]:


data['FATHER_HI_EDU_CD'].isnull().sum()


# In[ ]:


data['MOTHER_HI_EDU_CD'].isnull().sum()


# In[ ]:


#Replacing all of them with a new value 0.0 as they dont provide any info
data['FATHER_HI_EDU_CD']=data['FATHER_HI_EDU_CD'].fillna(0.0)
data['MOTHER_HI_EDU_CD']=data['MOTHER_HI_EDU_CD'].fillna(0.0)


# In[ ]:


data['DEGREE_GROUP_CD'].value_counts()


# In[ ]:


data['DEGREE_GROUP_DESC'].value_counts()


# In[ ]:


data['FIRST_TERM_PERF']=data['FIRST_TERM_EARNED_HRS']/data['FIRST_TERM_ATTEMPT_HRS']
data['SECOND_TERM_PERF']=data['SECOND_TERM_EARNED_HRS']/data['SECOND_TERM_ATTEMPT_HRS']


# In[ ]:


data['FIRST_TERM_PERF'].describe()


# In[ ]:


data['SECOND_TERM_PERF'].describe()


# In[ ]:


data['FIRST_TERM_PERF'].isnull().sum()


# In[ ]:


data['SECOND_TERM_PERF'].isnull().sum()


# In[ ]:


data['SECOND_TERM_PERF'].hist(bins=15,color='red')
plt.title("Second_term Preferred",size=20, color='red')


# In[ ]:


data["SECOND_TERM_PERF"].mode()


# In[ ]:


#Substituting NaN with mode value
data['SECOND_TERM_PERF']=data['SECOND_TERM_PERF'].fillna(1.0)


# In[ ]:


#Substituting values that exceeds 1.0 with 1.0
data['FIRST_TERM_PERF'] = data['FIRST_TERM_PERF'].apply(lambda x: 1.0 if x > 1.0 else x)
data['SECOND_TERM_PERF'] = data['SECOND_TERM_PERF'].apply(lambda x: 1.0 if x > 1.0 else x)


# In[ ]:


#It is in numeric and all the values seems to be fine
data['GROSS_FIN_NEED'].describe()


# In[ ]:


#It is in numeric and all the values seems to be fine
data['COST_OF_ATTEND'].describe()


# In[ ]:


#It is in numeric and all the values seems to be fine
data['EST_FAM_CONTRIBUTION'].describe()


# In[ ]:


data['UNMET_NEED'].describe()


# In[ ]:


#As financial needs cannot be in negative, hence imputing those to 0
data['UNMET_NEED'] = data['UNMET_NEED'].apply(lambda x: 0.0 if x <0 else x)


# In[ ]:


#Deriving DV from RETURNED_2ND_YR column
data['RETURNED_2ND_YR'].value_counts()


# In[ ]:


data['STDNT_ATT']=data['RETURNED_2ND_YR'].map(lambda x:1 if x==0 else 0)


# In[ ]:


data['CORE_COURSE_GRADE_1_F'].unique()


# In[ ]:


data[['CORE_COURSE_GRADE_2_F','CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S','CORE_COURSE_GRADE_3_S']]=data[['CORE_COURSE_GRADE_2_F','CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S','CORE_COURSE_GRADE_3_S']].fillna(value="NG")


# In[ ]:


data[['CORE_COURSE_NAME_2_F','CORE_COURSE_NAME_3_F','CORE_COURSE_NAME_1_S','CORE_COURSE_NAME_2_S','CORE_COURSE_NAME_3_S']]=data[['CORE_COURSE_NAME_2_F','CORE_COURSE_NAME_3_F','CORE_COURSE_NAME_1_S','CORE_COURSE_NAME_2_S','CORE_COURSE_NAME_3_S']].fillna(value="NC")


# In[ ]:


data['CORE_COURSE_NAME_1_F']=data['CORE_COURSE_NAME_1_F'].str.slice(0,4)
data['CORE_COURSE_NAME_2_F']=data['CORE_COURSE_NAME_2_F'].str.slice(0,4)
data['CORE_COURSE_NAME_3_F']=data['CORE_COURSE_NAME_3_F'].str.slice(0,4)

data['CORE_COURSE_NAME_1_S']=data['CORE_COURSE_NAME_1_S'].str.slice(0,4)
data['CORE_COURSE_NAME_2_S']=data['CORE_COURSE_NAME_2_S'].str.slice(0,4)
data['CORE_COURSE_NAME_3_S']=data['CORE_COURSE_NAME_3_S'].str.slice(0,4)


# In[ ]:


data.describe()


# In[ ]:


#It is already in numeric, hence not converting it
#data['UNMET_NEED']=pd.to_numeric(data['UNMET_NEED'],errors='coerce')


# In[ ]:


#Changeing to categorical variable
data['FIRST_TERM']=pd.Categorical(data['FIRST_TERM'])
data['SECOND_TERM']=pd.Categorical(data['SECOND_TERM'])
data['FATHER_HI_EDU_CD']=pd.Categorical(data['FATHER_HI_EDU_CD'])
data['MOTHER_HI_EDU_CD']=pd.Categorical(data['MOTHER_HI_EDU_CD'])
data['STDNT_ATT']=pd.Categorical(data['STDNT_ATT'])


# In[ ]:


X=data.drop(['STDNT_ATT','STUDENT IDENTIFIER','CORE_COURSE_NAME_4_F','CORE_COURSE_GRADE_4_F',
             'CORE_COURSE_NAME_5_F','CORE_COURSE_GRADE_5_F','CORE_COURSE_NAME_6_F','CORE_COURSE_GRADE_6_F',
             'CORE_COURSE_NAME_4_S','CORE_COURSE_GRADE_4_S','CORE_COURSE_GRADE_5_S','CORE_COURSE_NAME_5_S',
             'CORE_COURSE_GRADE_6_S','CORE_COURSE_NAME_6_S','RETURNED_2ND_YR','FIRST_TERM_ATTEMPT_HRS',
             'FIRST_TERM_EARNED_HRS','SECOND_TERM_ATTEMPT_HRS','SECOND_TERM_EARNED_HRS','STDNT_TEST_ENTRANCE1',
             'STDNT_TEST_ENTRANCE2','FATHER_HI_EDU_CD','MOTHER_HI_EDU_CD','DEGREE_GROUP_CD'],axis=1)


# In[ ]:


y=data['STDNT_ATT']


# In[ ]:


X=pd.get_dummies(X)


# In[ ]:


X.head()


# In[ ]:


import sklearn.model_selection as model_selection
X_train,X_test, y_train, y_test=model_selection.train_test_split(X,y, test_size=0.3, random_state=400)


# #### Random Forest Classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf=RandomForestClassifier(n_estimators=100,oob_score=True,n_jobs=-1,random_state=200)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.oob_score_


# In[ ]:


for w in range(30,500,25):
    clf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')


# In[ ]:


for w in range(170,200,2):
    clf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob))
    print('************************')


# In[ ]:


#Taking n_estimators as 174
clf=RandomForestClassifier(n_estimators=174,oob_score=True,n_jobs=-1,random_state=200)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.oob_score_


# In[ ]:


clf.feature_importances_


# In[ ]:


imp_feat=pd.Series(clf.feature_importances_,index=X.columns.tolist())


# In[ ]:


imp_feat.sort_values(ascending=False).head(30)


# In[ ]:


imp_feat.sort_values(ascending=False).head(10).plot(kind='bar')


# In[ ]:


import sklearn.metrics as metrics
mod2=clf.predict(X_test)


# In[ ]:


clf.score(X_test,y_test)


# In[ ]:


metrics.accuracy_score(y_test, clf.predict(X_test))


# #### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbc=GradientBoostingClassifier(n_estimators=40,random_state=200, max_depth=4)
gbc.fit(X_train,y_train)


# In[ ]:


#Getting the n_estimators
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(gbc,param_grid={'n_estimators':[40,60,80,100,120,140]})
mod.fit(X_train,y_train)


# In[ ]:


mod.best_estimator_


# In[ ]:


mod.best_params_


# In[ ]:


#For depth
mod=GridSearchCV(gbc,param_grid={'max_depth':[2,3,4,5,6,7,8]})


# In[ ]:


mod.fit(X_train,y_train)


# In[ ]:


mod.best_estimator_


# In[ ]:


model=clf.predict(X_test)


# In[ ]:


metrics.accuracy_score(y_test,model)


# In[ ]:


#Since, the accuracy score is 82.64%, this is a good model


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


y_pred=logreg.predict(X_test)


# In[ ]:


X_test['y_pred']=logreg.predict(X_test)


# In[ ]:


#Confusion Matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[ ]:


#Confusion matrix heatmap
import seaborn as sns
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_test,y_pred)))
plt.show()


# In[ ]:


#Confusion Matrix Evaluation Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 
print("Precision:",metrics.precision_score(y_test, y_pred)) 
print("Recall:",metrics.recall_score(y_test, y_pred))
print(metrics.classification_report(y_test,y_pred))


# In[ ]:


#ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred) 
auc = metrics.roc_auc_score(y_test,y_pred) 
plt.plot(fpr,tpr,label="data, auc="+str(auc)) 
plt.legend(loc=4) 
plt.show()


# In[ ]:




