#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Nianyi Wang(Barry) The first Hw: Titanic Survival Prediction 


# In[ ]:


import pandas as pd                
import numpy as np         


# In[ ]:


import os
os.getcwd()  #I cannot input the dataset, just check the path..


# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head(2)
#train.info()
#test.head(2)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


#train.describe()


# In[ ]:


train['Age'].describe()


# In[ ]:


train['Sex'].unique()


# In[ ]:


train= train.drop('Cabin',1)
train = train.drop('Embarked',1)
train = train.drop(columns = ['Name'])
train = train.drop(columns = ['Ticket'])
train = train.drop(columns = ['Age'])


# In[ ]:


train.head(2)


# In[ ]:


test= test.drop('Cabin',1)
test = test.drop('Embarked',1)
test= test.drop(columns = ['Name'])
test = test.drop(columns = ['Ticket'])
test = test.drop(columns = ['Age'])


# In[ ]:


test.head(2)


# In[ ]:


def qqq(gender):
    if gender=="male":
        gender=1
    else:
        gender=0


# In[ ]:


train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


# In[ ]:


train.head(6)


# In[ ]:


test.head(6)


# In[ ]:


X = train.drop('Survived',1)
y= train['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
testframe = std_scaler.fit_transform(test)
testframe.shape


# In[ ]:



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)
print(X_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,confusion_matrix
from sklearn.metrics import accuracy_score


# In[ ]:


#logistic regression
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(X_train,y_train)
predict=logreg.predict(X_test)
print(accuracy_score(y_test,predict))
print(confusion_matrix(y_test,predict))
print(precision_score(y_test,predict))
print(recall_score(y_test,predict))


# In[ ]:


#decision tree
import sklearn.tree as sk_tree
model = sk_tree.DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,max_leaf_nodes=None,min_impurity_decrease=0)
model.fit(X_train,y_train)
acc=model.score(X_test,y_test) 
print('accuracy:',acc)


# In[ ]:


#nueral network
import sklearn.neural_network as sk_nn
model = sk_nn.MLPClassifier(activation='tanh',solver='adam',alpha=0.0001,learning_rate='adaptive',learning_rate_init=0.001,max_iter=200)
model.fit(X_train,y_train)
acc=model.score(X_test,y_test) 
print('accuracy:',acc)

