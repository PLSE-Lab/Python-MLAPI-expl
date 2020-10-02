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


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as st 
import os


# In[ ]:


df=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# In[ ]:


df.info()


# # Exploratory Data Analysis
# 
# We will check the data for trends and draw insights. We will separate nominal and ordinal categorical variables. 

# In[ ]:


pd.set_option('display.max_columns',40)
df.describe()


# In[ ]:


df1=df.drop(['EmployeeCount','EmployeeNumber','StandardHours'],axis=1)
ordinal=['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','PerformanceRating',
 'RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']


# In[ ]:


df.describe(include='object')


# In[ ]:


cat=[i for i in df1.columns if df[i].dtype=='O']
numerical=[i for i in df1.columns if i not in cat+ordinal]


# In[ ]:


len(cat)+len(numerical)+len(ordinal)


# Now we use plots to check for significant features based on which we may be able to create new features to improve accuracy
# We use Box plots for numerical variables and count plots for categorical variables

# In[ ]:


fig, axes=plt.subplots(3,6,figsize=(15,10))
axes=axes.ravel()
ind=0
for i in numerical:
    sns.boxplot(x=df1.Attrition,y=df[i],ax=axes[ind])
    ind=ind+1
    #plt.show()
    


# In[ ]:





# In[ ]:


fig, axes=plt.subplots(3,6,figsize=(20,10))
axes=axes.ravel()
ind=0
for i in cat+ordinal:
    plt.xticks(rotation=90)
    sns.countplot(df[i],hue=df.Attrition,ax=axes[ind])
    ind=ind+1
   # plt.show()


# # Building Models

# Let us build an initial base model without any feature elimination or feature engineering to evalute and establish a baseline model which can be improved upon

# In[ ]:


df1['Attrition']=df1['Attrition'].map({'Yes':1,'No':0})
df1.Attrition.value_counts()


# In[ ]:


x=df1.drop('Attrition',axis=1)
y=df1.Attrition


# In[ ]:


cat.remove('Attrition')
x=pd.get_dummies(x,columns=cat,drop_first=True)


# In[ ]:


x.info()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold,cross_val_score,StratifiedKFold
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,classification_report,confusion_matrix,recall_score


# In[ ]:


def mod_score(algo,x,y):
    sfold=StratifiedKFold(random_state=7,shuffle=True)
    mod=algo.fit(x,y)
    cv1=cross_val_score(algo,x,y,cv=sfold,scoring='accuracy')
    cv2=cross_val_score(algo,x,y,cv=sfold,scoring='roc_auc')
    cv3=cross_val_score(algo,x,y,cv=sfold,scoring='recall')
    print('\nAccuracy : ',cv1.mean())
    print('ROC AUC score : ',cv2.mean())
    print('Recall: ',cv3.mean() )
    return mod

    


# In[ ]:


def rand_search(algo,params,x,y):
    rs=RandomizedSearchCV(algo,param_distributions=params,random_state=0,n_jobs=-1,n_iter=100,scoring='roc_auc',cv=10)
    mod=rs.fit(x,y)
    print(mod.best_score_)
    return mod.best_params_


# In[ ]:


def get_models(x,y):
    rbp=rand_search(RandomForestClassifier(),rfc_params,x,y)
    lbp=rand_search(LGBMClassifier(),lgb_params,x,y)
    kbp=rand_search(KNeighborsClassifier(),knn_params,x,y)
    lr=LogisticRegression(solver='liblinear')
    rfc=RandomForestClassifier(**rbp)
    lgb=LGBMClassifier(**lbp)
    ss=StandardScaler()
    knn=KNeighborsClassifier(**kbp)
    x_ss=ss.fit_transform(x)
    models={'Logistic Regression':lr,'Random Forest':rfc,'Light GBM':lgb,'knn':knn}
    m={}
    for i in models:
        if i!='knn':
            print('\n',i)
            m[i]=mod_score(models[i],x,y)
        else:
            print('\n',i)
            m[i]=mod_score(models[i],x_ss,y)
            
    return m


# In[ ]:


rfc_params={'n_estimators':st.randint(50,300),
    'criterion':['gini','entropy'],
    'max_depth':st.randint(2,20),
    'min_samples_split':st.randint(2,100),
    'min_samples_leaf':st.randint(2,100)}
lgb_params={ 'num_leaves':st.randint(31,60),
   'max_depth':st.randint(2,20),
    'learning_rate':st.uniform(0,1),
    'n_estimators':st.randint(50,300),
    'min_split_gain':st.uniform(0,0.3)}
knn_params={'n_neighbors': st.randint(5,30),
    'leaf_size':st.randint(20,70) }


# In[ ]:


m0=get_models(x,y)


# In[ ]:


x1=x
ss=StandardScaler()
d=ss.fit_transform(x)
x1['sum']=d.sum(axis=1)
x1['min']=d.min(axis=1)
x1['max']=d.max(axis=1)
x1['skew']=st.skew(d,axis=1)
x1['kurt']=st.kurtosis(d,axis=1)
x1['std']=d.std(axis=1)


# In[ ]:


m1=get_models(x1,y)


# In[ ]:


plt.figure(figsize=[10,20])
plt.xticks(rotation=90)
sns.barplot(x=x1.columns,y=m1['Random Forest'].feature_importances_)


# In[ ]:


plt.figure(figsize=[10,20])
plt.xticks(rotation=90)
sns.barplot(x=x1.columns,y=m1['Light GBM'].feature_importances_)


# Now we try to make new features and see if they may have any affect on the results

# In[ ]:



x2=x
x2['MonInc/ed']=x['MonthlyIncome']/x['Education']
x2['SalHike']=(x['PercentSalaryHike']/100)*x['MonthlyIncome']


# In[ ]:


cat.append('EducationField')


# In[ ]:


m2=get_models(x2,y)


# # Fixing Imbalance

# ## SMOTE

# The main problem as we see is that recall is not increasing. This is because of imbalance in the data set. The models are biased towards 0 and are not predicting any 1. This is because the model is giving prominence to only 0. This is a common problem in imbalanced datasets. 
# 
# This can be solved by under sampling or over sampling.
# In undersampling we may lose important data that may be required for creating rules. Hence we will do over sampling
# Here we use a technique called SMOTE(Synthetic Minority oversampling technique) which will create duplicate points for target 1 which will be close to the existing ones and solve the imbalance.

# In[ ]:


from imblearn.over_sampling import SMOTE,KMeansSMOTE,SMOTENC


# In[ ]:


def score(mod,x,y,samp_frac=0.5):
    sfold=StratifiedKFold(random_state=7,shuffle=True)
    #mod=algo.fit(x,y)
    smote=SMOTE(sampling_strategy=samp_frac,random_state=7)
    fold_auc=[]
    fold_acc=[]
    fold_recall=[]
    for i,j in sfold.split(x,y):
        xtr=x.iloc[i]
        ytr=y.iloc[i]
        xts=x.iloc[j]
        yts=y.iloc[j]
        x_sm,y_sm=smote.fit_resample(xtr,ytr)
        m=mod.fit(x_sm,y_sm)
        pred=m.predict(xts)
        prob=m.predict_proba(xts)[:,1]
        fold_acc.append(accuracy_score(yts,pred))
        fold_auc.append(roc_auc_score(yts,prob))
        fold_recall.append(recall_score(yts,pred))
    print('\nAccuracy : ',np.array(fold_acc).mean())
    print('ROC AUC score : ',np.array(fold_auc).mean())
    print('Recall: ',np.array(fold_recall).mean() )
    

def smote_models(x,y):
    rbp=rand_search(RandomForestClassifier(),rfc_params,x,y)
    lbp=rand_search(LGBMClassifier(),lgb_params,x,y)
    kbp=rand_search(KNeighborsClassifier(),knn_params,x,y)
    lr=LogisticRegression(solver='liblinear')
    rfc=RandomForestClassifier(**rbp)
    lgb=LGBMClassifier(**lbp)
    ss=StandardScaler()
    knn=KNeighborsClassifier(**kbp)
    
    models={'Logistic Regression':lr,'Random Forest':rfc,'Light GBM':lgb,'knn':knn}
    m={}
    for i in models:

        print('\n',i)
        m[i]=score(models[i],x,y)


# In[ ]:


smote_models(x,y)


# We can pick the required model and use it as required. 

# In[ ]:


smote_models(x1,y)


# In[ ]:


x


# In[ ]:





# In[ ]:




