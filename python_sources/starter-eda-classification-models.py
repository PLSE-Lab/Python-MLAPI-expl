#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe(include="all")


# In[ ]:


df_test.head()


# In[ ]:


df_test.info()


# In[ ]:


df_test.describe(include="all")


# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


print(df_train.Survived.sum()/df_train.Survived.count())


# In[ ]:


df_train.groupby(['Survived','Sex'])['Survived'].count()


# In[ ]:


sns.catplot(x='Sex', col='Survived', kind='count', data=df_train);


# In[ ]:


print("% of women survived: " , df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print("% of men survived: " , df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(16,7))
df_train['Survived'][df_train['Sex']=='male'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
df_train['Survived'][df_train['Sex']=='female'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')
plt.show()


# In[ ]:


pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


print("% of survivals in")
print("Pclass=1 : ", df_train.Survived[df_train.Pclass == 1].sum()/df_train[df_train.Pclass == 1].Survived.count())
print("Pclass=2 : ", df_train.Survived[df_train.Pclass == 2].sum()/df_train[df_train.Pclass == 2].Survived.count())
print("Pclass=3 : ", df_train.Survived[df_train.Pclass == 3].sum()/df_train[df_train.Pclass == 3].Survived.count())


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(21,7))
df_train['Survived'][df_train['Pclass']==1].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
df_train['Survived'][df_train['Pclass']==2].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
df_train['Survived'][df_train['Pclass']==3].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[2],shadow=True)
ax[0].set_title('Survived (Class 1 Passenger)')
ax[1].set_title('Survived (Class 2 Passenger)')
ax[2].set_title('Survived (Class 3 Passenger)')
plt.show()


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(21,7))
df_train['Survived'][df_train['Embarked']=='S'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
df_train['Survived'][df_train['Embarked']=='C'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)
df_train['Survived'][df_train['Embarked']=='Q'].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',ax=ax[2],shadow=True)
ax[0].set_title('Survived (Embarked S)')
ax[1].set_title('Survived (Embarked C)')
ax[2].set_title('Survived (Embarked Q)')
plt.show()


# In[ ]:


pd.crosstab([df_train.Survived], [df_train.Sex, df_train.Pclass, df_train.Embarked], margins=True)


# In[ ]:


sns.catplot(x='Embarked', col='Pclass', kind='count', data=df_train[df_train['Survived']==1],order=['Q','C','S']);


# In[ ]:


sns.catplot(x='Embarked', col='Pclass', kind='count', data=df_train[df_train['Survived']==0],order=['Q','C','S']);


# In[ ]:


#sort the ages into logical categories
bins = [-1, 8, 15, 30, np.inf]
labels = ['<8', '8-15', '15-31', '>31']
for df in [df_train, df_test]:
    df["Fare"] = df["Fare"].fillna(-0.5)
    df['FareGroup'] = pd.cut(df["Fare"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="FareGroup", y="Survived", data=df_train)
plt.xticks(np.linspace(0,5,6), labels, rotation=45, ha="right")
plt.xlim(-0.6,3.6)
plt.show()

# print values
df_train[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(
    by='Survived', ascending=False)


# In[ ]:


for df in [df_train, df_test]:
    df['FamilySize'] = (df['SibSp'] + df['Parch'] + 1)
    df.loc[df['FamilySize'] > 4, 'FamilySize'] = 5

#draw a bar plot of Age vs. survival
sns.barplot(x="FamilySize", y="Survived", data=df_train)
plt.xticks(np.linspace(0,5,6), rotation=45, ha="right")
plt.xlim(-0.6,4.9)
plt.show()

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(
        by='Survived', ascending=False)


# In[ ]:


df_train_ml = df_train.copy()
df_test_ml = df_test.copy()


# In[ ]:


df_train_ml = pd.get_dummies(df_train_ml, columns=['FareGroup','Sex', 'Embarked','Pclass'], drop_first=True)


# In[ ]:


passenger_id = df_test_ml['PassengerId']
df_test_ml = pd.get_dummies(df_test_ml, columns=['FareGroup','Sex', 'Embarked', 'Pclass'],drop_first=True)


# In[ ]:


corr = df_train_ml.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[ ]:


df_test_ml.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df_train_ml.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df_train_ml.dropna(inplace=True)


# In[ ]:


df_train_ml.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# for df_train_ml
scaler.fit(df_train_ml.drop('Survived',axis=1))
scaled_features = scaler.transform(df_train_ml.drop('Survived',axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features, columns=df_train_ml.columns[:-1])
# for df_test_ml
df_test_ml.fillna(df_test_ml.mean(), inplace=True)
# scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features, columns=df_test_ml.columns)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_ml.drop('Survived',axis=1), df_train_ml['Survived'], test_size=0.30,random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_train_ml_sc.drop('Survived',axis=1),df_train_ml['Survived'], test_size=0.30, random_state=101)


# In[ ]:


df_train_ml.drop('Survived',axis=1)


# In[ ]:


# unscaled
X_train_all = df_train_ml.drop('Survived',axis=1)
y_train_all = df_train_ml['Survived']
X_test_all = df_test_ml
# scaled
X_train_all_sc = df_train_ml_sc.drop('Survived',axis=1)
y_train_all_sc = df_train_ml['Survived']
X_test_all_sc = df_test_ml_sc


# In[ ]:


X_test_all.fillna(X_test_all.mean(), inplace=True)


# In[ ]:


X_test_all.head()


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 

logreg = LogisticRegression(random_state=100, solver='lbfgs')
logreg.fit(X_train_sc,y_train_sc)
pred_logreg = logreg.predict(X_test_sc)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test_sc, pred_logreg))
print(accuracy_score(y_test_sc, pred_logreg))


# In[ ]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[ ]:


randomforest = RandomForestClassifier(max_depth=6, n_estimators=600)
scores = cross_val_score(randomforest,X_train_all, y_train_all, cv=5)
acc_randomforest = round(scores.mean() * 100, 2)
print(acc_randomforest)


# In[ ]:


models = [KNeighborsClassifier(n_neighbors=4,n_jobs=-1),
          RandomForestClassifier(random_state=4, n_jobs=-1,n_estimators=1000, max_depth=5),
          XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.01,n_estimators=1000, max_depth=5),
          LogisticRegression(random_state=0, solver='lbfgs'),
          GradientBoostingClassifier(n_estimators=1000, learning_rate=.01,max_depth=5, random_state=0)
         ]
S_train, S_test = stacking(models,                   
                           X_train_sc,y_train_sc, X_test_sc,   
                           regression=False, 
                           mode='oof_pred_bag', 
                           needs_proba=False,
                           save_dir=None,
                           metric=accuracy_score, 
                           n_folds=6, 
                           stratified=True,
                           shuffle=True,  
                           random_state=0, 
                           verbose=2)
model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=3)
fitting = model.fit(S_train, y_train)
print("test accuracy: {} ".format(fitting.score(S_test,y_test_sc)))
print("train accuracy: {} ".format(fitting.score(S_train,y_train_sc)))


# In[ ]:


linear_svc = LinearSVC(max_iter=3000)
scores = cross_val_score(linear_svc, X_train_all, y_train_all, cv=5)
acc_linear_svc = round(scores.mean() * 100, 2)
print(acc_linear_svc)


# In[ ]:


logreg = LogisticRegression(solver='lbfgs',max_iter=30000)
scores = cross_val_score(logreg, X_train_all, y_train_all, cv=5)
acc_logreg = round(scores.mean() * 100, 2)
print(acc_logreg)


# In[ ]:


sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = df_test['PassengerId']
sub_logreg['Survived'] = pred_all_logreg
sub_logreg.to_csv('final_submission.csv',index=False)

