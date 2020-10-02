#!/usr/bin/env python
# coding: utf-8

# # If you found this as a worthy notebook please upvote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Loading datasets

# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submision=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head()


# # Simple EDA and feature engineering.

# In[ ]:


train.shape,test.shape


# In[ ]:


gender_submision.head()


# In[ ]:


train.info()


# In[ ]:


train.describe(exclude='number')#Name,Tciket,Cabin having high number of unique values,need some feature engineering


# # Feature engineering on Name

# In[ ]:


train.Name.head()


# In[ ]:


train['titlelst']=train.Name.str.split(' ').apply(lambda x:x[-1])
test['titlelst']=test.Name.str.split(' ').apply(lambda x:x[-1])

train['title1st']=train.Name.str.split(',').apply(lambda x:x[1]).str.split('.').apply(lambda x:x[0])
test['title1st']=test.Name.str.split(',').apply(lambda x:x[1]).str.split('.').apply(lambda x:x[0])


# In[ ]:


dict(test.title1st.value_counts())# we can further reduce this by setting 'Rare' value to some title.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='title1st',data=train,hue='Survived')
plt.xticks(rotation=60)


# In[ ]:


#Rather than assigning a string 'Rare' values i have encoded them dirrectly ,'Rare' got 4
train.title1st=train.title1st.apply(lambda x:1 if x==' Mr' else 2 if x==' Mrs' else 3 if x==' Miss' else 4)
test.title1st=test.title1st.apply(lambda x:1 if x==' Mr' else 2 if x==' Mrs' else 3 if x==' Miss' else 4)


# In[ ]:


print(train.titlelst.value_counts())#Still have many unique values, not so important
droplist=['titlelst']


# In[ ]:


droplist.append('Name')


# # Exploring ticket

# In[ ]:


print(train.Ticket.str.isnumeric().value_counts())
train.Ticket[train.Ticket.str.isnumeric().apply(lambda x:not x)]


# In[ ]:


train.Ticket=train.Ticket.apply(lambda x:'num' if x.isnumeric() else x)
train.Ticket=train.Ticket.str.split('/').apply(lambda x:x[0]).str.split(' ').apply(lambda x:x[0]).str.split('.').apply(lambda x:x[0])


# In[ ]:


test.Ticket=test.Ticket.apply(lambda x:'num' if x.isnumeric() else x)
test.Ticket=test.Ticket.str.split('/').apply(lambda x:x[0]).str.split(' ').apply(lambda x:x[0]).str.split('.').apply(lambda x:x[0])


# In[ ]:


train.Ticket.value_counts()


# In[ ]:


train.Ticket.replace(dict(zip(['WE','P','SW','Fa','F4','SO','SCO'],['Rare']*7)),inplace=True)
test.Ticket.replace(dict(zip(['WE','P','SW','Fa','F4','SO','SCO','LP','AQ'],['Rare']*9)),inplace=True)


# In[ ]:


sns.countplot('Ticket',data=train,hue='Survived')


# # Feature engineering on Sibsp,Parch

# In[ ]:


train['family_size']=train.SibSp+train.Parch+1# for himself
train['issingle']=train.family_size.apply(lambda x:1 if x==1 else 0)
train['smallfamily']=train.family_size.apply(lambda x:1 if x<=3 and x>1 else 0)
train['middfamily']=train.family_size.apply(lambda x:1 if x<=8 and x>3 else 0)
train['largefamily']=train.family_size.apply(lambda x:1 if x>8 else 0)


# In[ ]:


test['family_size']=test.SibSp+test.Parch+1
test['issingle']=test.family_size.apply(lambda x:1 if x==1 else 0)
test['smallfamily']=test.family_size.apply(lambda x:1 if x<=3 and x>1 else 0)
test['middfamily']=test.family_size.apply(lambda x:1 if x<=8 and x>3 else 0)
test['largefamily']=test.family_size.apply(lambda x:1 if x>8 else 0)


# In[ ]:


sns.countplot('family_size',data=train,hue='Survived')


# # Missing values

# In[ ]:


train.isnull().sum()


# In[ ]:


train.title1st.value_counts()


# In[ ]:


for _ in range(1,5):
    g=sns.distplot(train.loc[train.title1st==_,'Age'],label=_)
    g.axvline(train.loc[train.title1st==_,'Age'].mean())

plt.legend()


# In[ ]:


train.Age.fillna(28,inplace=True)
test.Age.fillna(28,inplace=True)


# In[ ]:


train.Age=train.Age.apply(lambda x:1 if x<=10 else 2 if x<=25 else 3 if x<=45 else 4 if x<=60 else 5)
test.Age=test.Age.apply(lambda x:1 if x<=10 else 2 if x<=25 else 3 if x<=45 else 4 if x<=60 else 5)


# # Outlier detection

# In[ ]:


train.head()


# In[ ]:


_,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
ax=ax.flatten()
sns.distplot(train.Fare,ax=ax[0])
sns.boxplot(x='Pclass',y='Fare',data=train,ax=ax[1])
sns.boxplot(x='Age',data=train,y='Fare',ax=ax[2])
sns.boxplot(x='title1st',data=train,y='Fare',ax=ax[3])


# In[ ]:


train=train.loc[train.Fare<300]


# # Exploring cabin

# In[ ]:


train.Cabin.value_counts()#we can convert it into floor rather than specifying each room on the floor.


# In[ ]:


train['Cabin']=train.Cabin.map(str).apply(lambda x:x[0])
test['Cabin']=test.Cabin.map(str).apply(lambda x:x[0])


# In[ ]:


test.head(1)


# In[ ]:


train.Cabin.replace({'n':np.nan},inplace=True)
test.Cabin.replace({'n':np.nan},inplace=True)


# In[ ]:


train.Cabin.replace({"T":'U'},inplace=True)


# In[ ]:


train.Cabin.value_counts()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.Cabin.fillna('U',inplace=True)
#test.Cabin.fillna('U',inplace=True)


# In[ ]:


df=train[['Fare','Age','Cabin']].dropna()
df.head()


# # Imputing missing value of test data by KNN.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['Fare','Age']],df.Cabin)
knn=KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
knn.score(X_test,y_test)


# In[ ]:


test.Fare=test.Fare.fillna(train.Fare.median())


# In[ ]:


_=test[['Fare','Age','Cabin']]
_=_.loc[_.isnull().Cabin==True,['Fare','Age']]


# In[ ]:


y_pred=knn.predict(_)


# In[ ]:


test.isnull().sum()


# In[ ]:


_=test.copy()
_=_.loc[_.isnull().Cabin==True]
_.head()


# In[ ]:


_.Cabin=y_pred
_.head()


# In[ ]:


test.dropna(inplace=True)
test=pd.concat((test,_),axis=0)


# In[ ]:


test.sort_index(inplace=True)


# In[ ]:


test.shape


# In[ ]:


train.drop(droplist,axis=1,inplace=True)
test.drop(droplist,axis=1,inplace=True)


# In[ ]:


cat_features=train.columns[train.dtypes==object]
cat_features


# In[ ]:


train.describe(exclude='number')


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in cat_features:
    le.fit(train[col])
    test[col]=le.transform(test[col].map(str))
    train[col]=le.transform(train[col].map(str))


# In[ ]:


test.drop('PassengerId',axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


test.head()


# In[ ]:


feature=train.drop(['Survived'],axis=1)
label=train.Survived
feature.head()


# # Modling

# In[ ]:


import xgboost as xgb
dmatrix=xgb.DMatrix(feature,label)
params={'objective':'binary:logistic','max_depth':5,'colsample_bytree':.6,'eta':.1,'alpha':1}
result=xgb.cv(params,dmatrix,early_stopping_rounds=10,as_pandas=True,num_boost_round=100,nfold=5,metrics='error')
result


# In[ ]:


xgb_cl=xgb.train(params,dmatrix,num_boost_round=19)
xgb.plot_importance(xgb_cl)


# In[ ]:


feature.index=list(range(len(feature)))
label.index=list(range(len(label)))


# # A function for validation (StratifiedKfold)

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=123)
def validate(estimator,feature,label):
    score=[]
    for trainind,testind in skf.split(feature,label):
        X_train,y_train=feature.loc[trainind],label[trainind]
        X_test,y_test=feature.loc[testind],label[testind]
        estimator.fit(X_train,y_train)
        score.append(accuracy_score(y_test,estimator.predict(X_test)))
    return score    


# In[ ]:


xgb_cl=xgb.XGBClassifier(max_depth=10,colsample_bytree=.6,learning_rate=.01,alpha=1,)
validate(xgb_cl,feature,label)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=3,n_estimators=500)
validate(rf,feature,label)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
validate(knn,feature,label)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
validate(lr,feature,label)


# In[ ]:


from sklearn.svm import SVC
svm=SVC(C=15)
validate(svm,feature,label)


# In[ ]:


from lightgbm import LGBMClassifier
lgb=LGBMClassifier(max_depth=5,colsample_bytree=.6,num_leaves=30,subsample_for_bin=800)
validate(lgb,feature,label)


# # Making a voting classifier with top 3 performing model.

# In[ ]:


from sklearn.ensemble import VotingClassifier
models=(['xgboost',xgb_cl],['lgb',lgb],['lr',lr])
vc=VotingClassifier(estimators=models)
validate(vc,feature,label)


# # Final model training.

# vc.fit(feature,label)

# In[ ]:


xgb_cl.fit(feature,label)


# # Prediction

# y_pred=vc.predict(test)

# In[ ]:


y_pred=xgb_cl.predict(test)


# In[ ]:


t=pd.read_csv('/kaggle/input/titanic/test.csv')
submission=t[['PassengerId']]
submission['Survived']=y_pred


# In[ ]:


submission.head(10)


# In[ ]:


submission.to_csv('submission_new_features_lgb.csv',index=False)


# In[ ]:


submission.Survived.value_counts(),label.value_counts()


# # If you found this as a worthy notebook please upvote.
# Finaly i chose xgboost as final model,its performance is better than the votting classifier.
