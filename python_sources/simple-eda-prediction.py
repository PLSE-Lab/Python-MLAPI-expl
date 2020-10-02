#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


data= pd.read_csv('../input/train.csv')
data2 = pd.read_csv('../input/test.csv')


# In[ ]:


data.head()


# In[ ]:


data2.head()


# In[ ]:


data.shape


# In[ ]:


data2.shape


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


data2.isnull().sum()


# **We can delete 'Cabin' as there too many null values**

# In[ ]:


del data['Cabin']
del data2['Cabin']


# **Columns- PassengerId, Name, Ticket, are of no use in the prediction, as they are just values indicating nothing**

# In[ ]:


del data['PassengerId']
del data2['PassengerId']


# In[ ]:


del data['Name']
del data2['Name']


# In[ ]:


del data['Ticket']
del data2['Ticket']


# In[ ]:


sns.boxplot('Pclass','Age',data=data)


# In[ ]:


sns.boxplot('Embarked','Age',data=data)


# **We can replace 'Age' null values, with 'Pclass' each category median values as indicated by the Boxplot**<br>
# 'Embarked' is not the right variable for 'Age' null values, as all median are nearly same.

# In[ ]:


data['Age'].fillna(0,inplace=True)
data2['Age'].fillna(0,inplace=True)


# In[ ]:


for i in range(891):
    if data['Pclass'][i]==1 and data['Age'][i]==0:
        data['Age'][i]=data['Pclass'].median()
    elif data['Pclass'][i]==2 and data['Age'][i]==0:
        data['Age'][i]=data['Pclass'].median()
    elif data['Pclass'][i]==3 and data['Age'][i]==0:
        data['Age'][i]=data['Pclass'].median()


# In[ ]:


for i in range(418):
    if data2['Pclass'][i]==1 and data2['Age'][i]==0:
        data2['Age'][i]=data2['Pclass'].median()
    elif data2['Pclass'][i]==2 and data2['Age'][i]==0:
        data2['Age'][i]=data2['Pclass'].median()
    elif data2['Pclass'][i]==3 and data2['Age'][i]==0:
        data2['Age'][i]=data2['Pclass'].median()


# In[ ]:


data.isnull().sum()


# In[ ]:


data2.isnull().sum()


# In[ ]:


sns.countplot(data['Embarked'])


# **'Embarked' column has much more 'S' category, thus we can replace Null values by S**

# In[ ]:


data['Embarked'].fillna('S',inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


plt.figure(figsize=(7,7))
sns.heatmap(data.corr(),annot=True,cmap='magma')


# **SibSp**

# In[ ]:


sns.factorplot('SibSp','Survived',data=data,kind='bar',palette='muted')


# Person having 0 or 1 sibling/spouse have more chances to survive.

# **Pclass**

# In[ ]:


sns.countplot('Pclass',data=data,hue='Survived')


# Clearly, we can see that higher the class of passenger, more the probabilty that he/she will be saved

# Since, Higher the class, more will be the Fair. Thus, we can remove the 'Fare' column from the dataset

# In[ ]:


del data['Fare']
del data2['Fare']


# In[ ]:


sns.countplot('Sex',hue='Survived',data=data)


# **Male is less Survived as compared to Females...**

# In[ ]:


data.head()


# Now, lets Encode the 'Sex' and 'Embarked' column

# In[ ]:


data2.head()


# In[ ]:


X= data.iloc[:,1:].values
y= data.iloc[:,0].values
X_test= data2.iloc[:,:].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X_1= LabelEncoder()
X[:,1]=le_X_1.fit_transform(X[:,1])
# data2
le_Xtest_1= LabelEncoder()
X_test[:,1]=le_Xtest_1.fit_transform(X_test[:,1])


# In[ ]:


le_X_2= LabelEncoder()
X[:,5]=le_X_2.fit_transform(X[:,5])
ohe= OneHotEncoder(categorical_features=[5])
X= ohe.fit_transform(X).toarray()
X= X[:,1:]
# data2
le_Xtest_2= LabelEncoder()
X_test[:,5]=le_Xtest_2.fit_transform(X_test[:,5])
ohe= OneHotEncoder(categorical_features=[5])
X_test= ohe.fit_transform(X_test).toarray()
X_test= X_test[:,1:]


# In[ ]:


from sklearn.model_selection import GridSearchCV,StratifiedKFold


# In[ ]:


kfold = StratifiedKFold(n_splits=10)


# **RandomForest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
rfc = RandomForestClassifier()
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [3,4,5,6],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(rfc,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X,y)

RFC_best=gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# In[ ]:


'''rfc=RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=5, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=10, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rfc.fit(X,y)
y_pred=rfc.predict(X_test)'''


# **XGBOOST**

# In[ ]:


'''from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV
params = {
     'learning_rate': [0.05],
     'n_estimators': [1000,1100],
     'max_depth':[7,8],
     'reg_alpha':[0.3,0.4,0.5]
    }
 
# Initializing the XGBoost Regressor
xgb_model = XGBClassifier()
 
# Gridsearch initializaation
gsearch = GridSearchCV(xgb_model, params,
                    verbose=True,
                    cv=5,
                    n_jobs=-1)
gsearch.fit(X,y) 
#Printing the best chosen params
XGB_best=gsearch.best_params_'''


# In[ ]:


'''xgb_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=7, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0.4, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
print(XGBClassifier())
 
# Cross validation scores
f1_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='f1')
print("F1-score = ",f1_scores," Mean F1 score = ",np.mean(f1_scores))
 
# Training the models
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)'''


# **Adaboost**

# In[ ]:


# Adaboost

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X,y)

ada_best=gsadaDTC.best_estimator_


# In[ ]:


'''from sklearn.tree import DecisionTreeClassifier
adc=AdaBoostClassifier(algorithm='SAMME.R',
                   base_estimator=DecisionTreeClassifier(class_weight=None,
                                                         criterion='entropy',
                                                         max_depth=None,
                                                         max_features=None,
                                                         max_leaf_nodes=None,
                                                         min_impurity_decrease=0.0,
                                                         min_impurity_split=None,
                                                         min_samples_leaf=1,
                                                         min_samples_split=2,
                                                         min_weight_fraction_leaf=0.0,
                                                         presort=False,
                                                         random_state=None,
                                                         splitter='best'),
                   learning_rate=1.5, n_estimators=2, random_state=7)
adc.fit(X,y)
y_pred=adc.predict(X_test)'''


# **Gradient Boosting Classifier**

# In[ ]:


# Gradient boosting tunning
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X,y)

GBC_best=gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# **SVC**

# In[ ]:


from sklearn.svm import SVC
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X,y)

SVMC_best=gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# In[ ]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X, y)
y_pred = votingC.predict(X_test)


# In[ ]:


data3=pd.read_csv('../input/test.csv')


# In[ ]:


submission = pd.DataFrame({'PassengerId': data3.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here
submission.to_csv('FirstCompetition_self.csv', index=False)


# In[ ]:




