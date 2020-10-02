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
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train['set'] = 'train'
test['set'] = 'test'


# In[ ]:


test.info()


# In[ ]:


test['Survived'] = np.nan


# In[ ]:


data = pd.concat([train,test],sort=True)


# In[ ]:


#finding out missing values
missing = data.isna().sum().sort_values(ascending=False)
percentage = (data.isna().sum()/data.isna().count()).sort_values(ascending=False)
values = pd.concat([missing,percentage],axis=1,keys=('missing','percentage'))
values


# In[ ]:


#cabin variable is not having any signifinace and having a lot of NA's So drop this variable
del data['Cabin']
del data['PassengerId']


# In[ ]:


#EDA
sns.countplot(data['Survived'])


# In[ ]:


#Distribution of survival according to Age
sns.boxplot(data['Survived'],data['Age'])


# In[ ]:


#Distribution of survival according to Age and Gender
sns.boxplot(data['Survived'],data['Age'],hue=data['Sex'])


# In[ ]:


#Distribution of survival according to Fare
sns.boxplot(data['Survived'],data['Fare'])
#There seems to be more survivals according to more fare


# In[ ]:


sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=data)


# In[ ]:


g = sns.catplot(x="Fare", y="Survived", row="Pclass",
                kind="box", orient="h", height=1.5, aspect=4,
                data=data.query("Fare > 0"))
g.set(xscale="log")


# In[ ]:


#NA treatment for Age
#Distribution of age according to Pclass
sns.boxplot(data['Pclass'],data['Age'])


# In[ ]:


#Imputing NA in Age variable
data['Age'].fillna(data.groupby(["Sex","Pclass"])['Age'].transform('median'),inplace=True)


# In[ ]:


#Check NA in Age variable
print(data['Age'].isna().sum())


# In[ ]:


#Imputing NA in Fare variable
data['Fare'].fillna(data.groupby(["Pclass"])['Age'].transform('median'),inplace=True)


# In[ ]:


print(data['Fare'].isna().sum())


# In[ ]:


#Distribution of Embarked with Pclass
sns.countplot(data['Embarked'],hue=data['Pclass'])


# In[ ]:


#Replace NA in Embarked with highest frequency Pclass
data['Embarked'].fillna('S',inplace=True)


# In[ ]:


data = pd.get_dummies(data, columns=['Embarked'],drop_first=True)


# In[ ]:


data = pd.get_dummies(data, columns=['Sex'],drop_first=True)


# In[ ]:


train_set = data[data['set']=='train']


# In[ ]:


del train_set['set']
train_set.info()


# In[ ]:


sns.heatmap(train_set.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


test_set = data[data['set']=='test']


# In[ ]:


del test_set['set']
del test_set['Survived']
del train_set['Name']
del test_set['Name']
del train_set['Ticket']
del test_set['Ticket']


# In[ ]:


#Finally, lets look the correlation of df_train
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(train_set.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()


# Model Building

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_set.drop('Survived',axis=1), 
                                                    train_set['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:





# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


test_predictions = logmodel.predict(test_set)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':test_predictions})


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic_Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


classifier = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=1)
model_rf = classifier.fit(X_train,y_train)


# In[ ]:


prediction_rf = model_rf.predict(X_test)


# In[ ]:


print(classification_report(y_test,prediction_rf))


# In[ ]:


sub_predict_rf = model_rf.predict(test_set)


# In[ ]:


submission1 = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':sub_predict_rf})


# In[ ]:


submission1.to_csv('Titanic_prediction1.csv',index=False)


# In[ ]:


from sklearn import linear_model
import numpy as np
reg = linear_model.RidgeClassifier(alpha=0.5, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)
model_rid = reg.fit(X_train,y_train)


# In[ ]:


predict_rid = model_rid.predict(X_test)
print(classification_report(y_test,predict_rid))


# In[ ]:


sub_prd_rid = model_rid.predict(test_set)


# In[ ]:


submission2 = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':sub_prd_rid})


# In[ ]:


submission2.to_csv("Titanic_prediction2.csv",index=False)


# In[ ]:


from sklearn.linear_model import RidgeClassifierCV
reg_CV = RidgeClassifierCV(alphas=np.logspace(-6,6,13))


# In[ ]:


model_CV = reg_CV.fit(X_train,y_train)


# In[ ]:


predict_cv = reg_CV.predict(X_test)


# In[ ]:


print(classification_report(y_test,predict_cv))


# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'kernel' : ('linear','rbf'), 'C' : [1,10]}


# In[ ]:


svc = svm.SVC()
gsv = GridSearchCV(svc,parameters)


# In[ ]:


model_gsv = gsv.fit(X_train,y_train)


# In[ ]:


predict_gvm = model_gsv.predict(X_test)


# In[ ]:


print(classification_report(y_test,predict_gvm))


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
model_sgd = sgd.fit(X_train,y_train)
predict_sgd = model_sgd.predict(X_test)


# In[ ]:


print(class)

