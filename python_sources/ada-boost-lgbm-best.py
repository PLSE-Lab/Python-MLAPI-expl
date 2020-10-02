#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data= pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


data


# In[ ]:


### Checking for null values


# In[ ]:


data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


### EDA--


# In[ ]:


#sns.pairplot(data,hue='target')


# In[ ]:


### There is no relateable relationship between the varible in


# In[ ]:


sns.countplot(data['target'])


# In[ ]:


sns.scatterplot(data['age'],data['trestbps'],hue=data['target'])


# In[ ]:


data.columns


# In[ ]:


data.plot(kind='box',y='age')


# In[ ]:


#plt.figure(figsize=(12,8))
data.plot(kind='box',figsize=(12,8))


# In[ ]:


for i in data.columns:
    sns.boxplot(data[i])
    #plt.xlabel(data(i))
    plt.show()


# In[ ]:


### We can observe few outliers in the data set


# In[ ]:


sns.boxplot(data['cp'],data['age'])


# In[ ]:


### We can see the variation of cp with age


# In[ ]:


data['fbs'].value_counts()


# In[ ]:


sns.distplot(data['age'])


# In[ ]:


# Distplot to check about the skewness of the data
for i in data.columns:
    sns.distplot(data[i])
    #plt.xlabel(data(i))
    plt.show()


# In[ ]:


sns.lmplot(x='age',y='chol',data=data,hue='target')


# In[ ]:


sns.catplot('target',data=data,hue='sex',kind='count')
# male are more affected 5 time more than the female


# In[ ]:


# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with postivie examples
plt.scatter(data.age[data.target==1],
            data.thalach[data.target==1],
            c="salmon")

# Scatter with negative examples
plt.scatter(data.age[data.target==0],
            data.thalach[data.target==0],
            c="lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[ ]:


# Person with the heart beat less than 140 have less chance of getting the diease


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


X= data.drop("target",axis=1)
y= data['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


y_train_pred =lr.predict(X_train)


# In[ ]:


confusion_matrix(y_train,y_train_pred)


# In[ ]:


accuracy_score(y_train,y_train_pred)


# In[ ]:


y_train_prob = lr.predict_proba(X_train)[:,1]
y_train_prob


# In[ ]:


print(classification_report(y_train,y_train_pred))   # Train and Predicted


# In[ ]:


y_test_pred= lr.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve


# In[ ]:


confusion_matrix(y_test,y_test_pred)


# In[ ]:


accuracy_score(y_test,y_test_pred)


# In[ ]:


y_test_prob = lr.predict_proba(X_test)[:,1]
y_test_prob


# In[ ]:


print(classification_report(y_test,y_test_pred))   # Train and Predicted


# In[ ]:


roc_auc_score(y_test,y_test_pred)                 # Train and Probability


# In[ ]:


fpr,tpr,thresholds= roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,"r-")
plt.show()


# In[ ]:


ytest_prob = pd.DataFrame([y_test.values,y_test_prob]).T
ytest_prob


# In[ ]:


df0 =ytest_prob[ytest_prob[0]==0.0]
df1 =ytest_prob[ytest_prob[0]==1.0]

sns.distplot(df0[1],color='r')
sns.distplot(df1[1],color='g')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_split=5,random_state=100)
dtc.fit(X_train,y_train)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
dtc.fit(X_train,y_train)


# In[ ]:


y_train_pred= dtc.predict(X_train)


# In[ ]:


confusion_matrix(y_train,y_train_pred)


# In[ ]:


accuracy_score(y_train,y_train_pred)


# In[ ]:


print(" AUC of the train",roc_auc_score(y_train,y_train_prob))


# In[ ]:


## Trying for test
# Similary do it for test
y_test_prob = dtc.predict_proba(X_test)[:,1]
y_test_predict = dtc.predict(X_test)
print (" Accuary of the test :",accuracy_score(y_test,y_test_predict))
print("confusion matrix on test : ")
print(confusion_matrix(y_test,y_test_predict))
print(" AUC of the test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[ ]:


dtc =DecisionTreeClassifier()
max_depth = [2,3,4,5,6,7,8]
min_samples_split = [2,3,4,5,6,7,8,9,10]
min_samples_leaf = [6,7,8,9,10,11,12,13,14,15]
criteria = ['gini','entropy']

params ={ 'max_depth':max_depth,
           'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'criterion':criteria}

gsearch =GridSearchCV(dtc,param_grid=params, scoring="roc_auc",cv=3,n_jobs=-1)

gsearch.fit(X,y)


# In[ ]:


gsearch.best_params_


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc = DecisionTreeClassifier(**gsearch.best_params_)


# In[ ]:


dtc.fit(X_train,y_train)

y_train_prob = dtc.predict_proba(X_train)[:,1]
y_train_predict = dtc.predict(X_train)
print (" Accuary of the train :",accuracy_score(y_train,y_train_predict))
print("confusion matrix on train : ")
print(confusion_matrix(y_train,y_train_predict))
print(" AUC of the test",roc_auc_score(y_train,y_train_prob))


# In[ ]:


dtc.fit(X_test,y_test)

y_test_prob = dtc.predict_proba(X_test)[:,1]
y_test_predict = dtc.predict(X_test)
print (" Accuary of the test :",accuracy_score(y_test,y_test_predict))
print("confusion matrix on test : ")
print(confusion_matrix(y_test,y_test_predict))
print(" AUC of the test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


# In[ ]:


y_train_prob = rfc.predict_proba(X_train)[:,1]
y_train_predict = rfc.predict(X_train)
y_test_prob = rfc.predict_proba(X_test)[:,1]
y_test_predict = rfc.predict(X_test)
print (" Accuary of the Random_forest train :",accuracy_score(y_train,y_train_predict))
print (" Accuary of the Random_forest test :",accuracy_score(y_test,y_test_predict))

print("confusion matrix on Random_forest train : ")
print(confusion_matrix(y_train,y_train_predict))

print("confusion matrix on Random_forest test : ")
print(confusion_matrix(y_test,y_test_predict))

print(" AUC of the Random_forest train",roc_auc_score(y_train,y_train_prob))
print(" AUC of the Random_forest test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


from scipy.stats import randint as sp_randint


# In[ ]:


params={'n_estimators':sp_randint(5,150),
        'max_features':sp_randint(1,14),
        'max_depth':sp_randint(2,10),
        'min_samples_leaf':sp_randint(1,50),
        'min_samples_split':sp_randint(2,50),
        'criterion':['gini','entropy']}
rsearch_rf= RandomizedSearchCV(rfc,param_distributions=params,n_iter=100,n_jobs=-1,cv=5,scoring='roc_auc')
rsearch_rf.fit(X,y)


# In[ ]:


rsearch_rf.best_params_


# In[ ]:


rfc=RandomForestClassifier(**rsearch_rf.best_params_)
rfc.fit(X_train,y_train)


# In[ ]:


y_test_pred= rfc.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_test_pred)


# In[ ]:


y_train_pred=rfc.predict(X_train)


# In[ ]:


accuracy_score(y_train,y_train_pred)


# In[ ]:


## KNN
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


# In[ ]:


X_trains=ss.fit_transform(X_train)
X_tests =ss.transform(X_test)


# In[ ]:


knc.fit(X_trains,y_train)

y_train_prob = knc.predict_proba(X_trains)[:,1]
y_train_predict = knc.predict(X_trains)
y_test_prob = knc.predict_proba(X_tests)[:,1]
y_test_predict = knc.predict(X_tests)
print (" Accuary of the KNN train :",accuracy_score(y_train,y_train_predict))
print (" Accuary of the KNN test :",accuracy_score(y_test,y_test_predict))

print("confusion matrix on KNN train : ")
print(confusion_matrix(y_train,y_train_predict))

print("confusion matrix on KNN test : ")
print(confusion_matrix(y_test,y_test_predict))

print(" AUC of the KNN train",roc_auc_score(y_train,y_train_prob))
print(" AUC of the KNN test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()


# In[ ]:


X_traint=X_train
X_testt=X_test


# In[ ]:


bnb.fit(X_traint,y_train)

y_train_prob = bnb.predict_proba(X_traint)[:,1]
y_train_predict = bnb.predict(X_traint)
y_test_prob = bnb.predict_proba(X_testt)[:,1]
y_test_predict = bnb.predict(X_testt)
print (" Accuary of the BNB- train :",accuracy_score(y_train,y_train_predict))
print (" Accuary of the BNB test :",accuracy_score(y_test,y_test_predict))

print("confusion matrix on BNB train : ")
print(confusion_matrix(y_train,y_train_predict))

print("confusion matrix on BNB test : ")
print(confusion_matrix(y_test,y_test_predict))

print(" AUC of the BNB train",roc_auc_score(y_train,y_train_prob))
print(" AUC of the BNB test",roc_auc_score(y_test,y_test_prob))


# In[ ]:





# In[ ]:


dtcy_train_prob=dtc.predict_proba(X_train)[:,1]
dtcy_test_prob=dtc.predict_proba(X_test)[:,1]


# In[ ]:


rfcy_train_prob=rfc.predict_proba(X_train)[:,1]
rfcy_test_prob=rfc.predict_proba(X_test)[:,1]


# In[ ]:


kncy_train_prob=knc.predict_proba(X_trains)[:,1]
kncy_test_prob=knc.predict_proba(X_tests)[:,1]


# In[ ]:


bnby_train_prob=bnb.predict_proba(X_train)[:,1]
bnby_test_prob=bnb.predict_proba(X_test)[:,1]


# In[ ]:


roc_auc_train=[roc_auc_score(y_train,dtcy_train_prob),roc_auc_score(y_train,rfcy_train_prob),roc_auc_score(y_train,kncy_train_prob),roc_auc_score(y_train,bnby_train_prob)]
roc_auc_test=[roc_auc_score(y_test,dtcy_test_prob),roc_auc_score(y_test,rfcy_test_prob),roc_auc_score(y_test,kncy_test_prob),roc_auc_score(y_test,bnby_test_prob)]
a=pd.DataFrame({'roc_auc_train':roc_auc_train,'roc_auc_test':roc_auc_test})
plt.figure(figsize=[10,5])
a.plot(kind='bar')
plt.xticks(np.arange(4),['dtc','rfc','knc','bnb'])
plt.show()


# In[ ]:


dfpr, dtpr, dthres = roc_curve(y_test,dtcy_test_prob)
rfpr, rtpr, rthres = roc_curve(y_test,rfcy_test_prob)
kfpr, ktpr, kthres = roc_curve(y_test,kncy_test_prob)
bfpr, btpr, bthres = roc_curve(y_test,bnby_test_prob)
plt.figure(figsize=[10,5])
plt.plot(dfpr,dtpr,c='r',label='DTC')
plt.plot(rfpr,rtpr,c='g',label='RFC')
plt.plot(kfpr,ktpr,c='b',label='KNC')
plt.plot(bfpr,btpr,c='y',label='BNB')
plt.legend()
plt.show()


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ada= AdaBoostClassifier(random_state=1)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve


# In[ ]:


ada.fit(X_train,y_train)


ob = ada.predict_proba(X_train)[:,1]
y_train_predict = ada.predict(X_train)
y_test_prob = ada.predict_proba(X_test)[:,1]
y_test_predict = ada.predict(X_test)
print (" Accuary of the Stack train :",accuracy_score(y_train,y_train_predict))
print (" Accuary of the Stack test :",accuracy_score(y_test,y_test_predict))

print("\n")
print("confusion matrix on Stack train : ")
print(confusion_matrix(y_train,y_train_predict))

print("confusion matrix on Stack test : ")
print(confusion_matrix(y_test,y_test_predict))
print("\n")

print(" AUC of the Stack train",roc_auc_score(y_train,y_train_prob))
print(" AUC of the Stack test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


roc_auc_train=[roc_auc_score(y_train,y_train_prob)]
roc_auc_test=[roc_auc_score(y_test,y_test_prob)]
a=pd.DataFrame({'roc_auc_train':roc_auc_train,'roc_auc_test':roc_auc_test})
plt.figure(figsize=[10,5])
a.plot(kind='bar')
plt.xticks(np.arange(1),['adboost'])
plt.show()


# In[ ]:


plt.figure(figsize=[12,8])
fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr,c='r',label="Adaboost")
plt.plot(fpr,fpr,c='b',label="threshold")
plt.legend()
plt.show()


# In[ ]:


import lightgbm as lgb


# In[ ]:


lgbc =lgb.LGBMClassifier()


# In[ ]:


lgbc.fit(X_train,y_train)

y_train_prob = lgbc.predict_proba(X_train)[:,1]
y_train_predict = lgbc.predict(X_train)
y_test_prob = lgbc.predict_proba(X_test)[:,1]
y_test_predict = lgbc.predict(X_test)
print (" Accuary of the Stack train :",accuracy_score(y_train,y_train_predict))
print (" Accuary of the Stack test :",accuracy_score(y_test,y_test_predict))

print("\n")
print("confusion matrix on Stack train : ")
print(confusion_matrix(y_train,y_train_predict))

print("confusion matrix on Stack test : ")
print(confusion_matrix(y_test,y_test_predict))
print("\n")

print(" AUC of the Stack train",roc_auc_score(y_train,y_train_prob))
print(" AUC of the Stack test",roc_auc_score(y_test,y_test_prob))


# In[ ]:


roc_auc_train=[roc_auc_score(y_train,y_train_prob)]
roc_auc_test=[roc_auc_score(y_test,y_test_prob)]
a= pd.DataFrame({"train":roc_auc_train,"test":roc_auc_test})
plt.figure(figsize=[10,5])
a.plot(kind='bar')
plt.xticks(np.arange(1),['adboost'])
plt.show()


# In[ ]:


plt.figure(figsize=[12,8])
fpr,tpr,thresholds=roc_curve(y_test,y_test_prob)
plt.plot(fpr,tpr,c='r',label="Adaboost")
plt.plot(fpr,fpr,c='b',label="threshold")
plt.legend()
plt.show()


# In[ ]:


# Ada Boost and Lg boost performes best with the scores- but its kind of over fitting.


# In[ ]:




