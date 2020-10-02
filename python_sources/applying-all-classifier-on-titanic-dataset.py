#!/usr/bin/env python
# coding: utf-8

# Lets take a look at data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
tdata=pd.read_csv('../input/train.csv')
x_test=pd.read_csv('../input/test.csv')
y_test=pd.read_csv('../input/gender_submission.csv')
train=pd.DataFrame(tdata)
train.info()


# Lets see how my datatable looks like.

# In[ ]:


train.head()


# Lets first encode the data for sex using LabelEncoder.
# 
# Then, Fill the empty values with Zero using Fillna.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(train['Sex'])
Sex=le.transform(train['Sex'])
sex=pd.DataFrame(Sex,)
train=pd.concat([train,sex],axis=1)
train.columns=['PassengerId','Survived',  'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','sex']


# Lets take look at columns.

# In[ ]:


train.columns


# We have to deal with test values so Lets do the same process what we did for training set.

# In[ ]:


le=LabelEncoder()
le.fit(x_test['Sex'])
Sex=le.transform(x_test['Sex'])
sex=pd.DataFrame(Sex)
X_test=pd.concat([x_test,sex],axis=1)
X_test.columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','sex']
X_test=X_test.fillna(0)
xtest=X_test[['Pclass','Age','SibSp','Parch','Fare','sex']]


# Now PassengerId, Name, Ticket, Cabin and Embarked will not make much impact on survival rate. So We will not take those columns into consideration.
# Lets define our testing set as below.

# In[ ]:


data=train[['Survived','Pclass','Age','SibSp','Parch','Fare','sex']]
xtr=train[['Pclass','Age','SibSp','Parch','Fare','sex']]
ytr=train['Survived']
Ytr=pd.DataFrame(ytr)


# Now Lets train the model on above data and lets see what is the accuracy.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error,mean_absolute_error
model=DecisionTreeClassifier()
model.fit(xtr,Ytr)
predict=model.predict(xtest)
predict


# Lets evaluate our test data.

# In[ ]:


Ytest=y_test['Survived']
Ytest.head()


# Lets check my accuracy score.

# In[ ]:


print("accuracy_score:", accuracy_score(Ytest,predict))


#  which is not bad. But I want more.

# In[ ]:


confusion_matrix(Ytest,predict)


# Lets apply other classification method and lets see the results

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(xtr,Ytr)
predict=model.predict(xtest)
accuracy_score(Ytest,predict)
print("accuracy_score:", accuracy_score(Ytest,predict))


# I can see slight improvement but lets apply other classifiers.

# In[ ]:


from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier()
model.fit(xtr,Ytr)
predict=model.predict(xtest)
accuracy_score(Ytest,predict)
print("accuracy_score:", accuracy_score(Ytest,predict))


# I can see slight improvement but lets apply other classifiers.

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(xtr,Ytr)
predict=model.predict(xtest)
accuracy_score(Ytest,predict)
print("accuracy_score:", accuracy_score(Ytest,predict))


# This is also much better than previous one but not upto the expection.

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(xtr,Ytr)
predict1=model.predict(xtest)
accuracy_score(Ytest,predict)
print("accuracy_score:", accuracy_score(Ytest,predict))


# Wowww this is really nice accuracy score.
# 
# I never expected such a geat score.
# Lets check with array,

# In[ ]:


confusion_matrix(Ytest,predict1)


# Accuracy score nearly to 93%. We will try to optimise it in the upcoming versions.
# 
# For now Lets submit the result.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": x_test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:


filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# Thank you for your time.
# 
# **Let me know If you have any suggestion.
# 
# Please upvote if you like.**
