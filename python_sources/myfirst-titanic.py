#!/usr/bin/env python
# coding: utf-8

# It's my first notbook so its so basic ,has a simple data preprocessing then I used logistic regression and SVM for prediction

# In[ ]:


#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#load our data
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


train


# **Separating and Splitting data**
# 
# I assumed that "PassengerId","Name","Ticket","Cabin" and "Fare" columns will not add usefull information to our prediction, so I dropped it.
# also gender_submission ( as a test lable ) has an additional column which is "PassengerId", so I also dropped it.

# In[ ]:


y_train=pd.DataFrame(train["Survived"])
x_train=train.drop(["PassengerId","Name","Ticket","Cabin","Fare"],1)
x_test=test.drop(["PassengerId","Name","Ticket","Cabin","Fare"],1)
y_test=gender_submission.drop(["PassengerId"],1)


# In[ ]:


x_train


# In[ ]:


y_train


# In[ ]:


x_test


# In[ ]:


y_test


# In[ ]:


x_train.info()


# In[ ]:


x_test.info()


# In[ ]:


y_train.info()


# In[ ]:


y_test.info()


# - from info we can see that age column in each of training and testing set has NaN values, and also Embarked column in traing set as we can see in the following cell .

# In[ ]:


print(x_train["Age"].isna().sum())
print(x_test["Age"].isna().sum())
print(x_train["Embarked"].isna().sum())


# - Here I replaced the NaN values in age columns with the mean of ages, and replaced the NaN values in Embarked columns with the mode of Emparked columns

# In[ ]:


x_train["Age"]=x_train["Age"].fillna(value=x_train["Age"].mean())
x_test["Age"]=x_test["Age"].fillna(value=x_test["Age"].mean())
x_train["Embarked"]=x_train["Embarked"].fillna(value=x_train["Embarked"].mode()[0])
x_test["Embarked"]=x_test["Embarked"].fillna(value=x_test["Embarked"].mode()[0])


# - Now we can print a gain and see that there is no NaN values

# In[ ]:


print(x_train["Age"].isna().sum())
print(x_test["Age"].isna().sum())
print(x_train["Embarked"].isna().sum())


# - Embarked and Sex columns are not numerical so I converted it here to catigorical values

# In[ ]:


x_train


# In[ ]:


x_test


# In[ ]:


x_train.Embarked.unique()


# In[ ]:


sex={'male':0,'female':1}
emb={'Q':0,'S':1,'C':2}


# In[ ]:


x_train["Sex"]=x_train["Sex"].map(sex)
x_train["Embarked"]=x_train["Embarked"].map(emb)
x_test["Sex"]=x_test["Sex"].map(sex)
x_test["Embarked"]=x_test["Embarked"].map(emb)


# In[ ]:


x_train


# In[ ]:


x_test


# **Visualizing our data**
# 
# - The number of unsurvived peaple is greater than The number of survived.

# In[ ]:


y_train.hist(column='Survived')


# In[ ]:


x_train.hist(column='Pclass')


# - The number of survived males is greater than The number of survived females

# In[ ]:


x_train.hist(column='Sex')


# In[ ]:


x_train.hist(column='Embarked')


# In[ ]:


x_train.hist(column='Parch')


# In[ ]:


x_train.hist(column='SibSp')


# - Most of unsurvived peaple are at the age of thirty

# In[ ]:


x_train.hist(column='Age',bins=20)


# **Correlations**

# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Pclass"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Pclass"],kde_kws={"label":"died"})


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Age"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Age"],kde_kws={"label":"died"})


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Sex"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Sex"],kde_kws={"label":"died"})


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["SibSp"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["SibSp"],kde_kws={"label":"died"})


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Parch"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Parch"],kde_kws={"label":"died"})


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Embarked"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Embarked"],kde_kws={"label":"died"})


# - To read ages more easy I splitted it to four ranges

# In[ ]:


pd.cut(x_train["Age"],bins=4).unique()


# In[ ]:


def factorize_age(col):
    for i,e in enumerate(col):
        if col[i] > 0.34 and col[i] <=20.315:
            col[i]=0
        elif col[i] > 20.315 and col[i] <=40.21:
             col[i]=1
        elif col[i] >40.21 and col[i] <=60.105:
            col[i]=2
        elif col[i] > 60.105 and col[i] <=80.0:
            col[i]=3
    return col


# - Convert Age here to catigorical values

# In[ ]:


factorize_age(x_train["Age"])
factorize_age(x_test["Age"])


# In[ ]:



x_train


# In[ ]:


x_test


# In[ ]:


fig=plt.figure(figsize=(10,10))
sns.distplot(x_train.loc[x_train["Survived"]==1]["Age"],kde_kws={"label":"Survived"})
sns.distplot(x_train.loc[x_train["Survived"]==0]["Age"],kde_kws={"label":"died"})


# - We don't need "Survived" column in training set as it is in testing set

# In[ ]:


x_train=x_train.drop(["Survived"],1)


# **Prediction**
# - Now we will use logistic regression and SVM to predict our lables 

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[ ]:


x_train


# In[ ]:


from sklearn.linear_model import LogisticRegression
regressor1=LogisticRegression(random_state=0)


# In[ ]:


regressor1.fit(x_train,y_train)


# In[ ]:


y_pred1=regressor1.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)


# In[ ]:


cm1


# - Get 95.5% accuracy with logistic regrission

# In[ ]:


print((255+144)/(266+152))


# In[ ]:


from sklearn.svm import SVC
regressor2=SVC(kernel='rbf',random_state=0)
regressor2.fit(x_train,y_train)


# In[ ]:


y_pred2=regressor2.predict(x_test)
cm2=confusion_matrix(y_test,y_pred2)


# In[ ]:


cm2


# - Get 90.2% accuracy with SVM

# In[ ]:


print((254+123)/(254+123+12+29))


# In[ ]:


gender_submission


# In[ ]:


y_pred1=pd.DataFrame(y_pred1)
y_pred1


# In[ ]:


y_pred1["PassengerId"]=gender_submission["PassengerId"]
y_pred1["Survived"]=y_pred1.loc[:,0]
y_pred1=y_pred1.drop(y_pred1.columns[0],1)


# In[ ]:


y_pred1


# In[ ]:


y_pred1.to_csv('SurvivedPrediction.csv',index=False)

