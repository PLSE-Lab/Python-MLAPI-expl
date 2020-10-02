#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin,BaseEstimator


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')
# y = train.pop('Survived')
test = pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')


# In[ ]:


train.head()


# In[ ]:


train.isna().sum()


# In[ ]:


train['HasCabin'] = np.where(train['Cabin'].isna(),1,0)
test['HasCabin'] = np.where(test['Cabin'].isna(),1,0)


# In[ ]:


train['IsAlone'] = np.where((train['SibSp']+train['Parch'])==0,1,0)
test['IsAlone'] = np.where((test['SibSp']+test['Parch'])==0,1,0)


# In[ ]:


train['FamilySize'] = train['SibSp']+train['Parch']
test['FamilySize'] = test['SibSp']+test['SibSp']


# In[ ]:


train['FamilySize'] = train['FamilySize'].astype('object')
test['FamilySize'] = test['FamilySize'].astype('object')


# In[ ]:


train['Pclass'] = train['Pclass'].astype('object')
test['Pclass'] = test['Pclass'].astype('object')


# In[ ]:


sns.distplot(train['Survived'],kde=False)


# In[ ]:


sns.barplot(x = 'IsAlone',y = 'Survived',data= train)


# In[ ]:


sns.barplot(x='HasCabin',y='Survived',data=train)


# In[ ]:


sns.barplot(x='Sex',y='Survived',data=train)


# In[ ]:


sns.barplot(x='Embarked',y='Survived',data=train)


# In[ ]:


sns.barplot(x='FamilySize',y='Survived',data=train)


# In[ ]:


train['Embarked'].fillna('C',inplace=True)
test['Embarked'].fillna('C',inplace=True)


# In[ ]:


train['Title'] = pd.DataFrame(((pd.DataFrame((train['Name'].str.split(', ')).tolist(),index=train.index)[1]).str.split(' ')).tolist(),index=train.index)[0]


# In[ ]:


test['Title'] = pd.DataFrame(((pd.DataFrame((test['Name'].str.split(', ')).tolist(),index=test.index)[1]).str.split(' ')).tolist(),index=test.index)[0]


# In[ ]:


train['Title'].value_counts()


# In[ ]:


train['Title'] = np.where(train['Title']=='Mr.','Mr','').astype('object')+np.where(train['Title']=='Mrs.','Mrs','').astype('object')+np.where(train['Title']=='Miss.','Miss','').astype('object')+np.where(train['Title']=='Master.','Master','').astype('object')


# In[ ]:


test['Title'] = np.where(test['Title']=='Mr.','Mr','').astype('object')+np.where(test['Title']=='Mrs.','Mrs','').astype('object')+np.where(test['Title']=='Miss.','Miss','').astype('object')+np.where(test['Title']=='Master.','Master','').astype('object')


# In[ ]:


np.where(train['Title']=='')


# In[ ]:


train['Title'] = train['Title'].replace(r'','Others')


# In[ ]:


test['Title'] = test['Title'].replace(r'','Others')


# In[ ]:


sns.barplot(x='Title',y='Survived',data=train)


# In[ ]:


train['Age'].fillna(train['Age'].mode()[0],inplace=True)


# In[ ]:


test['Age'].fillna(test['Age'].mode()[0],inplace=True)


# In[ ]:


train['AgeGroup'] = np.where(train['Age']<=3,'Babies','').astype('object')+np.where((train['Age']>3) & (train['Age']<=16),'Children','').astype('object')+np.where((train['Age']>16) & (train['Age']<=30),'Young Adults','').astype('object')+np.where((train['Age']>30) & (train['Age']<=60),'Middle Aged','').astype('object')+np.where(train['Age']>60,'Senior Citizen','').astype('object')


# In[ ]:


test['AgeGroup'] = np.where(test['Age']<=3,'Babies','').astype('object')+np.where((test['Age']>3) & (test['Age']<=16),'Children','').astype('object')+np.where((test['Age']>16) & (test['Age']<=30),'Young Adults','').astype('object')+np.where((test['Age']>30) & (test['Age']<=60),'Middle Aged','').astype('object')+np.where(test['Age']>60,'Senior Citizen','').astype('object')


# In[ ]:


sns.barplot(x='AgeGroup',y='Survived',data=train)


# In[ ]:


train['Fare'].value_counts().iloc[0:50]


# In[ ]:


train['Fare'].isna().sum()


# In[ ]:


train['FareGroup'] = pd.cut(train['Fare'],[-1,10,50,100,200,1000],labels=['Low','Mid','UpperMid','High','VeryHigh'])


# In[ ]:


test['FareGroup'] = pd.cut(test['Fare'],[-1,10,50,100,200,1000],labels=['Low','Mid','UpperMid','High','VeryHigh'])


# In[ ]:


sns.barplot(x='FareGroup',y='Survived',data=train)


# In[ ]:


train['SexCode'] = np.where(train['Sex']=='male',1,0)


# In[ ]:


test['SexCode'] = np.where(test['Sex']=='male',1,0)


# In[ ]:


train.head(3)


# In[ ]:


test.head()


# In[ ]:


X = train[['Pclass','Embarked','HasCabin','IsAlone','Title','AgeGroup','FareGroup','SexCode','FamilySize']]
y = train['Survived']


# In[ ]:


predict = test[['Pclass','Embarked','HasCabin','IsAlone','Title','AgeGroup','FareGroup','SexCode','FamilySize']]


# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.base import TransformerMixin,BaseEstimator,RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,PolynomialFeatures,StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier,RidgeClassifierCV,LassoCV,LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB,GaussianNB,ComplementNB,CategoricalNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


seed = 64
rfc = RandomForestClassifier(n_jobs=-1,random_state=seed)
abc = AdaBoostClassifier(random_state=seed,n_estimators=250)
gbc = GradientBoostingClassifier(random_state=seed)
lr = LogisticRegression(random_state=seed,max_iter=5000)
lrcv = LogisticRegressionCV(random_state=seed)
rc = RidgeClassifier(random_state=seed)
rccv = RidgeClassifierCV()
lc = LassoCV(random_state=seed,n_jobs=-1)
dtc = DecisionTreeClassifier(random_state=seed)
knc = KNeighborsClassifier(n_jobs=-1)
svc = SVC(random_state=seed)
mlp = MLPClassifier(random_state=-1)
bnb = BernoulliNB()
gnb = GaussianNB()
cnb = ComplementNB()
canb = CategoricalNB()
lgbm = LGBMClassifier(random_state=seed,n_jobs=-1)
xgb = XGBClassifier(random_state=seed,n_jobs=-1)
ss = StandardScaler()


# In[ ]:


cen = ce.WOEEncoder()
X_coded = cen.fit_transform(X,y)


# In[ ]:


ss = MinMaxScaler()
X_coded = ss.fit_transform(X_coded)


# In[ ]:


predict_coded = ss.transform(cen.transform(predict))


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_coded,y,test_size=0.1,random_state =seed)


# In[ ]:


model = [rfc,abc,gbc,lr ,lrcv ,rc,rccv,lc,dtc,knc,svc,lgbm,xgb]
training_score = []


# In[ ]:


def modelselect(model):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)


# In[ ]:


for i in model:
    training_score.append(modelselect(i))
training_score


# In[ ]:


cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)


# In[ ]:


parameters = {'hyperthread':[4],
#               'objective':['binary:logistic'],
              'learning_rate': [0.05,0.01,0.1,1], 
              'max_depth': [2,4,6,8,10],
              'min_child_weight': [8,10,12,14,16],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [250,500,1000,2000],
              'missing':[-999],
              'seed': [1337]}


# In[ ]:


grid = GridSearchCV(lgbm,parameters,n_jobs=-1,cv=cv,verbose=1,scoring='roc_auc')


# In[ ]:


# grid = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#               hyperthread=4, importance_type='gain',
#               interaction_constraints=None, learning_rate=1, max_delta_step=0,
#               max_depth=2, min_child_weight=16, missing=-999,
#               monotone_constraints=None, n_estimators=500, n_jobs=-1,
#               num_parallel_tree=1, objective='binary:logistic', random_state=64,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1337,
#               silent=1, subsample=0.8, tree_method=None,
#               validate_parameters=False, verbosity=None)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


best_lgbm = grid.best_estimator_


# In[ ]:


grid.best_score_


# In[ ]:


best_lgbm.score(X_test,y_test)


# In[ ]:


cross_val_score(best_lgbm,X_coded,y,cv=cv,scoring='roc_auc',verbose=2,n_jobs=-1)


# In[ ]:


pre = best_lgbm.predict(predict_coded)


# In[ ]:


pd.DataFrame({'PassengerId':test.index,'Survived':pre}).to_csv('submission.csv',index=False)


# In[ ]:


from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# In[ ]:


X.shape


# In[ ]:


classifier = Sequential()
classifier.add(Dense(5,activation='relu',kernel_initializer='uniform',input_dim=9))
classifier.add(Dense(8,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(5,activation='relu',kernel_initializer='uniform'))
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))


# In[ ]:


opt = Adam(learning_rate=0.01)


# In[ ]:


classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


hist = classifier.fit(X_train,y_train,epochs=8,validation_data=(X_test,y_test))


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(hist.history['val_loss'],label='val')
plt.plot(hist.history['loss'],label='loss')


# In[ ]:


plt.plot(hist.history['val_accuracy'],label='val')
plt.plot(hist.history['accuracy'],label='loss')


# In[ ]:


X_train.shape


# In[ ]:


X_train = X_train.reshape(801,3,3,1)
X_test = X_test.reshape(-1,3,3,1)


# In[ ]:


from keras import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D


# In[ ]:


classifier = Sequential()
classifier.add(Conv2D(64,(1,1),activation='relu',input_shape=(3,3,1)))
classifier.add(Conv2D(64,(1,1),activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(1,activation='sigmoid'))


# In[ ]:


classifier.compile(opt,'binary_crossentropy',['accuracy'])


# In[ ]:


hist = classifier.fit(X_train,y_train,epochs=15,batch_size=1,validation_data=(X_test,y_test))


# In[ ]:


plt.plot(hist.history['val_loss'],label='val')
plt.plot(hist.history['loss'],label='loss')


# In[ ]:


plt.plot(hist.history['val_accuracy'],label='val')
plt.plot(hist.history['accuracy'],label='loss')


# In[ ]:


pred = ss.transform(cen.transform(predict))


# In[ ]:


pred = pred.reshape(-1,3,3,1)


# In[ ]:


pre = classifier.predict_classes(pred)


# In[ ]:


pre.reshape(-1,)


# # **ACCURACIES** - 
# # 1.   LightGBM - 0.81791831
# # 2.   ANN - 0.7889
# # 3.   CNN - 0.8111

# 

# 

# In[ ]:


# pd.DataFrame({'PassengerId':test.index,'Survived':pre.reshape(-1,)}).to_csv('submission.csv',index=False)


# In[ ]:




