#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### PREDICTING THE SURVIVAL OF PASSENGERS IN THE TITANIC
# The famous dataset to start learning the aspects of Exploratory Data Analysis and Machine Learning Models.
# 
# Features of the dataset
# - Survival: Whether the passenger has survived or not
# - pclass: Class in which a passenger travelled.
# - sex: 	Gender of the passenger	
# - Age: 	Age of the passengers in years	
# - sibsp: number of siblings / spouses in the Titanic of a passenger
# - parch: number of parents / children in the Titanic of a passenger
# - ticket: Ticket number	
# - fare: ticket fare	
# - cabin: Cabin in which passengers travelled	
# - embarked: 	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


#Importing the necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing the dataset
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
# For easy identification of rows from the train and test, we create a feature called label and set 0 for rows from train and 1 for rows from test
train['Label']='0'
test['Label']='1'


# In[ ]:


# Concatenating train and test data.
df=pd.concat([train,test])


# In[ ]:


df.shape


# In[ ]:


# As we have the label feature, we can reset the index values.
df.index=np.arange(0,1309)


# In[ ]:


# Checking for null values
df.isna().sum()


# In[ ]:


# Cabin feature has the most null values (77%)
df['Cabin'].isna().sum()*100/len(df['Cabin'])


# In[ ]:


# Converting the feature to string type as the nan values also get converted to string values.
df['Cabin']=df['Cabin'].astype(str)


# In[ ]:


df['Cabin'].value_counts()


# In[ ]:


# Extracting the first letter of each value as it as it provides the different cabin classes in which the passengers have travelled.
list_cabin=[]
for i in df['Cabin']:
    if i!='nan':
        i=i[0]
        list_cabin.append(i)
    else:
        list_cabin.append(i)
list_cabin


# In[ ]:


df['Cabin']=list_cabin


# In[ ]:


# Converting nan to a seperate class called NA
df['Cabin']=df['Cabin'].apply(lambda x:'NA' if x=='nan' else x)


# In[ ]:


df['Cabin'].value_counts()


# - With the categories of NA and others, we can segregate the passengers as with cabin if cabin has values other than NA and without cabin if it is NA 

# In[ ]:


df['With_without_cabin']=df['Cabin'].apply(lambda x: 0 if x=='NA' else 1)


# In[ ]:


sns.countplot(df['With_without_cabin'],hue=df['Survived'])


# - Out of the 1300 passengers, around 700 have travelled without a cabin and only 200 out of them have survived.

# In[ ]:


# Imputing the missing values with the most recurring value ('S')
df['Embarked']=df['Embarked'].apply(lambda x: x if x=='S' or x=='C' or x=='Q' else 'S')


# In[ ]:


# To impute the missing values of age, we consider the most corelated features with age.
df.corr()['Age']


# In[ ]:


# Creating a feature called married from the name feature
name=list(df['Name'].values)


# In[ ]:


title=[]
for i in name:
    i=i.split(' ')[1]
    title.append(i)


# In[ ]:



married=[]
for i in title:
    if 'Mr.' in i or 'Mrs.' in i:
        married.append(1)
    else:
        married.append(0)
        


# In[ ]:


df['Married']=married


# In[ ]:


df=df.drop(['Name','Cabin','Ticket'],1)


# In[ ]:


df.head()


# In[ ]:


# Married and Pclass have the highest corelation with Age and hence we use the information from those features to impute
# the missing values.
abs(df.corr()['Age'])


# In[ ]:


age=df[['Age','Married','Pclass']]


# In[ ]:


# We have different combinations of information in the married and pclass features and we take all the combinations 
# and impute the missing values with those median values.
m0_cl1=age[(age['Married']==0) & (age['Pclass']==1)]
m0_cl2=age[(age['Married']==0) & (age['Pclass']==2)]
m0_cl3=age[(age['Married']==0) & (age['Pclass']==3)]
m1_cl1=age[(age['Married']==1) & (age['Pclass']==1)]
m1_cl2=age[(age['Married']==1) & (age['Pclass']==2)]
m1_cl3=age[(age['Married']==1) & (age['Pclass']==3)]


# In[ ]:


m0_cl1.fillna(m0_cl1['Age'].median(),inplace=True)
m0_cl2.fillna(m0_cl2['Age'].median(),inplace=True)
m0_cl3.fillna(m0_cl3['Age'].median(),inplace=True)
m1_cl1.fillna(m1_cl1['Age'].median(),inplace=True)
m1_cl2.fillna(m1_cl2['Age'].median(),inplace=True)
m1_cl3.fillna(m1_cl3['Age'].median(),inplace=True)


# In[ ]:


age_df=pd.concat([m0_cl1,m0_cl2,m0_cl3,m1_cl1,m1_cl2,m1_cl3],0)


# In[ ]:


age_df['index']=age_df.index


# In[ ]:


age_df.sort_values('index',inplace=True)


# In[ ]:


df['Age']=age_df['Age']


# In[ ]:


# As we have only 1 missing value, we can impute it with the mean value.
df['Fare'].fillna(df['Fare'].mean(),inplace=True)


# In[ ]:


# Only Survived feature has missing values and the 418 values are from the test data which we have to predict.
df.isna().sum()


# In[ ]:


# We can combine the parch and SibSp to calculate the number of family members they had on board.
df['Dependents']=df['Parch']+df['SibSp']


# In[ ]:


# Mapping the different values obtained to categorize it.
df['Dependents']=df['Dependents'].map({0:'No',1:'Few',2:'Few',3:'Few',4:'Few',5:'Few',6:'Many',7:'Many',10:'Many'})


# In[ ]:


# Dropping the Parch and SibSp features as the information has been obtained from it.
df=df.drop(['Parch','SibSp'],1)


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


# As the location in which a passenger boarded does not affect whether they survived or not, we can drop it.
df=df.drop('Embarked',1)


# In[ ]:


# As the passengerID is also a feature not required for the model, we can drop it 
df=df.drop('PassengerId',1)


# In[ ]:


# Convering class to string type in order to create dummies for this feature too.
df['Pclass']=df['Pclass'].astype(str)


# In[ ]:


df=pd.get_dummies(df,drop_first=True)


# In[ ]:


# Splitting indepenndent and dependent features
X=df.drop('Survived',1)
y=df['Survived']


# In[ ]:


# Scaling the features to bring them all to one scale.
from sklearn.preprocessing import StandardScaler
X=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)


# In[ ]:


X.head()


# In[ ]:


#Dropping the Label feature as we can split the data into train and test based on the index values. 
X=X.drop('Label_1',1)


# In[ ]:


X_train=X.iloc[:891]
X_test=X.iloc[891:]
y_train=df['Survived'].iloc[:891]
y_test=df['Survived'].iloc[891:]


# In[ ]:


# Splitting the train data into train and validation to check the performance of different models.
from sklearn.model_selection import train_test_split
train_X,X_val,train_y,y_val=train_test_split(X_train,y_train,test_size=0.25,random_state=42)


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score


# In[ ]:


# We tune the value of C as it gives penalty to different features.
for i in [0.0001,0.001,0.1,1,10,100,1000]:
    lr=LogisticRegression(C=i).fit(train_X,train_y)
    y_pred_lr=lr.predict(X_val)
    print('For C value',i,'f1 score is: ',f1_score(y_val,y_pred_lr))


# In[ ]:


lr=LogisticRegression(C=10).fit(train_X,train_y)
y_pred_lr=lr.predict(X_val)
print('For C value 10 f1 score is: ',f1_score(y_val,y_pred_lr))


# In[ ]:


from sklearn.preprocessing import binarize


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


# Calcutating the threshold value of where to convert the probable values as 0 and 1.
for i in range(1,11):
    y_pred2=lr.predict_proba(X_val)
    bina=binarize(y_pred2,threshold=i/10)[:,1]
    cm2=confusion_matrix(y_val,bina)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    print('f1 score: ',f1_score(y_val,bina))
    print('accuracy score: ',accuracy_score(y_val,bina))
    print('\n')


# In[ ]:


#0.4 is where we have the least misclassified values.
y_pred2=lr.predict_proba(X_val)
bina=binarize(y_pred2,threshold=0.4)[:,1]
print(confusion_matrix(y_val,bina))
print('f1_score: ',f1_score(y_val,bina))
print('accuracy_score: ',accuracy_score(y_val,bina))


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier().fit(train_X,train_y)
y_pred_dt=dt.predict(X_val)
f1_score(y_val,y_pred_dt)


# In[ ]:


# Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV
dt=DecisionTreeClassifier()
param_grid = {

    'criterion': ['gini','entropy'],
    'max_depth': [10,15,20,25],
    'min_samples_split' : [5,10,15,20],
    'min_samples_leaf': [2,5,7],
    'random_state': [42,135,777],
}

rf_grid=GridSearchCV(estimator=dt,param_grid=param_grid,n_jobs=-1,return_train_score=True)

rf_grid.fit(train_X,train_y)


# In[ ]:


rf_grid.best_params_


# In[ ]:


cv_res_df=pd.DataFrame(rf_grid.cv_results_)


# In[ ]:


cv_res_df.head()


# In[ ]:


# We take the point where the test score is high and also where the difference between the train and test score is minimal
plt.figure(figsize=(20,5))
plt.plot(cv_res_df['mean_train_score'])
plt.plot(cv_res_df['mean_test_score'])
plt.xticks(np.arange(0,250,5),rotation=90)
plt.show()


# In[ ]:


cv_res_df[['mean_train_score','mean_test_score']].iloc[240:246]


# In[ ]:


pd.DataFrame(cv_res_df.iloc[240]).T


# In[ ]:


# Creating a decision tree model with the optimal hyperparameters
dt=DecisionTreeClassifier(max_depth=20,min_samples_leaf=7,min_samples_split=5,criterion='entropy',random_state=42).fit(train_X,train_y)


# In[ ]:


y_pred_dtc=dt.predict(X_val)


# In[ ]:


accuracy_score(y_val,y_pred_dtc)


# In[ ]:


f1_score(y_val,y_pred_dtc)


# In[ ]:


confusion_matrix(y_val,y_pred_dtc)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier().fit(train_X,train_y)
y_pred_rf=rf.predict(X_val)
print(accuracy_score(y_val,y_pred_rf),'\t',f1_score(y_val,y_pred_rf))


# In[ ]:


# Hyper parameter tuning
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
param_grid = {
    
    'n_estimators':[10,20,30],
    'criterion': ['gini','entropy'],
    'max_depth': [10,15,20,25],
    'min_samples_split' : [5,10,15],
    'min_samples_leaf': [2,5,7],
    'random_state': [42,135,777],
    'class_weight': ['balanced' ,'balanced_subsample']
}

rf_grid=GridSearchCV(estimator=rf,param_grid=param_grid,n_jobs=-1,return_train_score=True)

rf_grid.fit(train_X,train_y)


# In[ ]:


cv_res_df=pd.DataFrame(rf_grid.cv_results_)


# In[ ]:


plt.figure(figsize=(20,5))
plt.plot(cv_res_df['mean_train_score'])
plt.plot(cv_res_df['mean_test_score'])
plt.xticks(np.arange(0,1200,50),rotation=90)
plt.show()


# In[ ]:


pd.DataFrame(cv_res_df.iloc[330])


# In[ ]:


rfc=RandomForestClassifier(class_weight='balanced',criterion='gini',max_depth=10,min_samples_leaf=2,min_samples_split=5,n_estimators=30,random_state=42).fit(train_X,train_y)


# In[ ]:


y_pred_rfc=rfc.predict(X_val)
print(accuracy_score(y_val,y_pred_rfc),'\t',f1_score(y_val,y_pred_rfc))


# In[ ]:


confusion_matrix(y_val,y_pred_rfc)


# ### XG Boost

# In[ ]:


#Converting the dataset into matrix.
import xgboost as xgb
dtrain=xgb.DMatrix(train_X,train_y)
dval=xgb.DMatrix(X_val,y_val)


# In[ ]:


param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dval)


# In[ ]:


for i in range(1,11):
    bina=binarize(preds.reshape(-1,1),threshold=i/10)
    cm2=confusion_matrix(y_val,bina)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    print('f1 score: ',f1_score(y_val,bina))
    print('accuracy score: ',accuracy_score(y_val,bina))
    print('\n')


# ### Gradient Boost

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gr_boost=GradientBoostingClassifier().fit(train_X,train_y)
y_pred_gr=gr_boost.predict(X_val)
print(accuracy_score(y_val,y_pred_gr),'\t',f1_score(y_val,y_pred_gr))


# In[ ]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4,6,8,10],
              'min_samples_leaf': [20,50,100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=5, scoring="accuracy", n_jobs= -1, verbose = 1)

gsGBC.fit(train_X,train_y)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# In[ ]:


gsGBC.best_params_


# In[ ]:


gr_boost1=GradientBoostingClassifier(**gsGBC.best_params_).fit(train_X,train_y)
y_pred_gr1=gr_boost1.predict(X_val)
print(f1_score(y_val,y_pred_gr1),accuracy_score(y_val,y_pred_gr1))


# ### Ada Boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ada_boost=AdaBoostClassifier().fit(train_X,train_y)
y_pred_ada=ada_boost.predict(X_val)
print(accuracy_score(y_val,y_pred_ada),'\t',f1_score(y_val,y_pred_ada))


# ### Submission:

# #### Logistic Regression

# In[ ]:


y_pred_test_lr=lr.predict(X_test)
y_pred_test2=lr.predict_proba(X_test)


# In[ ]:


y_pred_test_lr=binarize(y_pred_test2,threshold=0.4)[:,1]


# In[ ]:


log_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
log_pred['Survived']=y_pred_test_lr
log_pred.to_csv('Logistic pred.csv',index=False)


# #### Decision Tree

# In[ ]:


y_pred_test_dt=dt.predict(X_test)
dt_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
dt_pred['Survived']=y_pred_test_dt
dt_pred.to_csv('Decision_tree_pred.csv',index=False)


# #### Random Forest

# In[ ]:


y_pred_test_rfc=rfc.predict(X_test)
rfc_pred=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
rfc_pred['Survived']=y_pred_test_rfc
rfc_pred.to_csv('Random_forest_pred.csv',index=False)


# #### XGBoost

# In[ ]:


dtest=xgb.DMatrix(X_test)


# In[ ]:


pred_xgb=bst.predict(dtest)


# In[ ]:


y_pred_xgb=binarize(pred_xgb.reshape(-1,1),threshold=0.4)


# In[ ]:


pred_test_xgb=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
pred_test_xgb['Survived']=y_pred_xgb


# In[ ]:


pred_test_xgb.to_csv('XGBoost_pred.csv',index=False)


# #### Gradient Boost

# In[ ]:


y_pred_test_gr=gr_boost.predict(X_test)
pred_gr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
pred_gr['Survived']=y_pred_test_gr
pred_gr.to_csv('GradientBoost_pred.csv',index=False)


# #### Ada Boost

# In[ ]:


y_pred_test_ada=ada_boost.predict(X_test)
pred_ada=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
pred_ada['Survived']=y_pred_test_ada
pred_ada.to_csv('AdaBoost_pred.csv',index=False)


# ### PCA
# - We try to perform PCA on the data to check if it improves the performance.

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca=PCA().fit(X)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()


# In[ ]:


# From the graph, we see that 95% of the variance is explained by 6 principle components and hence we can take 6 Principle
# components.
X1=PCA(n_components=6).fit_transform(X)
X1


# In[ ]:


# Splitting train and test independent features.
X_test_pca,X_train_pca=X1[891:],X1[:891]


# ### Train and validation

# In[ ]:


# We repeat the same steps for the converted data.
trai_x,val_x,trai_y,val_y=train_test_split(X_train_pca,y_train,test_size=0.3,random_state=42)


# #### Logistic Regression

# In[ ]:


for i in [0.00001,0.0001,0.001,0.1,1,10,100,1000]:
    lr=LogisticRegression(C=i).fit(trai_x,trai_y)
    y_pred_pca_lr=lr.predict(val_x)
    print('For C value',i,'f1 score is: ',f1_score(val_y,y_pred_pca_lr))


# In[ ]:


lr=LogisticRegression(C=1).fit(trai_x,trai_y)
y_pred_pca_lr=lr.predict(val_x)
print(f1_score(val_y,y_pred_pca_lr),'\t',accuracy_score(val_y,y_pred_pca_lr))


# In[ ]:


for i in range(1,11):
    y_pred2=lr.predict_proba(val_x)
    bina=binarize(y_pred2,threshold=i/10)[:,1]
    cm2=confusion_matrix(val_y,bina)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    print('f1 score: ',f1_score(val_y,bina))
    print('accuracy score: ',accuracy_score(val_y,bina))
    print('\n')


# In[ ]:


y_pred2=lr.predict_proba(X_test_pca)
y_pred_pca_lr=binarize(y_pred2,threshold=0.4)[:,1]


# In[ ]:


sub_lr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_lr['Survived']=y_pred_pca_lr
sub_lr.to_csv('Logistic_Regression_PCA.csv',index=False)


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier().fit(trai_x,trai_y)
y_pred_dt=dt.predict(val_x)
f1_score(val_y,y_pred_dt)


# In[ ]:



from sklearn.model_selection import GridSearchCV
dt=DecisionTreeClassifier()
param_grid = {

    'criterion': ['gini','entropy'],
    'max_depth': [5,10,15,20,25],
    'min_samples_split' : [5,10,15,20],
    'min_samples_leaf': [2,5,7,10],
    'random_state': [42,135,777],
}

rf_grid=GridSearchCV(estimator=dt,param_grid=param_grid,n_jobs=-1,return_train_score=True)

rf_grid.fit(trai_x,trai_y)


# In[ ]:


rf_grid.best_params_


# In[ ]:


cv_res_df=pd.DataFrame(rf_grid.cv_results_)


# In[ ]:


plt.figure(figsize=(12,6))
cv_res_df['mean_train_score'].plot()
cv_res_df['mean_test_score'].plot()
plt.xticks(np.arange(0,500,20),rotation=90)
plt.show()


# In[ ]:


cv_res_df.iloc[267]


# In[ ]:


dt=DecisionTreeClassifier(max_depth=5,min_samples_leaf=7,min_samples_split=10,criterion='entropy',random_state=42).fit(trai_x,trai_y)


# In[ ]:


y_pred_val_pca_dt=dt.predict(val_x)
print(f1_score(val_y,y_pred_val_pca_dt),accuracy_score(val_y,y_pred_val_pca_dt))


# In[ ]:


y_pred_pca_dt=dt.predict(X_test_pca)


# In[ ]:


sub_dt=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_dt['Survived']=y_pred_pca_dt
sub_dt.to_csv('Decision_Tree_PCA.csv',index=False)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier().fit(trai_x,trai_y)
y_pred_rf=rf.predict(val_x)
print(accuracy_score(val_y,y_pred_rf),'\t',f1_score(val_y,y_pred_rf))


# In[ ]:


from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
param_grid = {
    
    'n_estimators':[10,20,30],
    'criterion': ['gini','entropy'],
    'max_depth': [10,15,20,25],
    'min_samples_split' : [5,10,15],
    'min_samples_leaf': [2,5,7],
    'random_state': [42,135,777],
    'class_weight': ['balanced' ,'balanced_subsample']
}

rf_grid=GridSearchCV(estimator=rf,param_grid=param_grid,n_jobs=-1,return_train_score=True)

rf_grid.fit(trai_x,trai_y)


# In[ ]:


rf_grid.best_params_


# In[ ]:


cv_res_df=pd.DataFrame(rf_grid.cv_results_)
cv_res_df


# In[ ]:


plt.figure(figsize=(12,6))
cv_res_df['mean_train_score'].plot()
cv_res_df['mean_test_score'].plot()


# In[ ]:


cv_res_df['diff']=cv_res_df['mean_train_score']-cv_res_df['mean_test_score']


# In[ ]:


cv_res_df[['mean_train_score','mean_test_score']][(cv_res_df['mean_test_score']>0.80) & (cv_res_df['mean_train_score']<0.90)]


# In[ ]:


cv_res_df.iloc[11]


# In[ ]:


rf=RandomForestClassifier(n_estimators=10,max_depth=10,min_samples_leaf=2,min_samples_split=10,random_state=777,criterion='gini').fit(trai_x,trai_y)
y_pred_rf=rf.predict(val_x)
print(accuracy_score(val_y,y_pred_rf),'\t',f1_score(val_y,y_pred_rf))


# In[ ]:


y_pred_pca_rf=rf.predict(X_test_pca)


# In[ ]:


sub_rf=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_rf['Survived']=y_pred_pca_rf
sub_rf.to_csv('Random_Forest_PCA.csv',index=False)


# ### XGBoost

# In[ ]:


dtrain_pca=xgb.DMatrix(trai_x,trai_y)
dval_pca=xgb.DMatrix(val_x)
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain_pca, num_round)
# make prediction
preds = bst.predict(dval_pca)


# In[ ]:


for i in range(1,11):
    bina=binarize(preds.reshape(-1,1),threshold=i/10)
    cm2=confusion_matrix(val_y,bina)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    print('f1 score: ',f1_score(val_y,bina))
    print('accuracy score: ',accuracy_score(val_y,bina))
    print('\n')


# In[ ]:


dtest_pca=xgb.DMatrix(X_test_pca)
preds=bst.predict(dtest_pca)
bina=binarize(preds.reshape(-1,1),threshold=0.5)


# In[ ]:


sub_xgb=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_xgb['Survived']=bina
sub_xgb.to_csv('XGBoost_PCA.csv',index=False)


# ### GradientBoost

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gr_boost=GradientBoostingClassifier().fit(trai_x,trai_y)
y_pred_gr=gr_boost.predict(val_x)
print(accuracy_score(val_y,y_pred_gr),'\t',f1_score(val_y,y_pred_gr))


# In[ ]:


y_pred_pca_gr=gr_boost.predict(X_test_pca)


# In[ ]:


sub_gr=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_gr['Survived']=y_pred_pca_gr
sub_gr.set_index('PassengerId')
sub_gr.to_csv('GradientBoost_PCA.csv',index=False)


# ### AdaBoost

# In[ ]:


ada_boost=AdaBoostClassifier().fit(trai_x,trai_y)
y_pred_val=ada_boost.predict(val_x)
print(accuracy_score(val_y,y_pred_val),'\t',f1_score(val_y,y_pred_val))


# In[ ]:


y_pred_pca_ab=ada_boost.predict(X_test_pca)


# In[ ]:


sub_ab=pd.DataFrame(np.arange(892,1310),columns=['PassengerId'])
sub_ab['Survived']=y_pred_pca_ab
sub_ab.to_csv('Adaboost_PCA.csv',index=False)


# ### Upon submission, I got the highest score for the Gradient Boosting Model without PCA. For different data pre processing techniques, we get different results. My best result was from the above mentioned model.

# In[ ]:




