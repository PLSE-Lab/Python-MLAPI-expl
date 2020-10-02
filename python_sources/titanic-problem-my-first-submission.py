#!/usr/bin/env python
# coding: utf-8

# **Titanic solve problem********

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
train_data.head()

# Any results you write to the current directory are saved as output.


# In[ ]:


test_data.head() #for displaying first five datas


# In[ ]:


train_data.info()  # gives information about all the columns, about its data types and no of observation


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


train_data.describe()# Gives the descriptive statistics of all the columns


# In[ ]:


test_data.describe()


# In[ ]:


sns.countplot(x='Survived', data=train_data);


# Here it is shown that total 342 passenger survived out of 891

# In[ ]:


train_data.Survived.sum()


# 342 in training set

# In[ ]:


train_data.Survived.count()


# In[ ]:


print(train_data.Survived.sum()/train_data.Survived.count())


# 38.3 percentage of people survived 

# In[ ]:


train_data.groupby(['Survived','Sex'])['Survived'].count()


# **    Female passengers survived more than the male passengers**

# In[ ]:


sns.catplot(x='Sex', col='Survived', kind='count', data=train_data);


# In[ ]:


print("percentage of women survived: " ,train_data[train_data.Sex == 'female'].Survived.sum()/train_data[train_data.Sex == 'female'].Survived.count())
print("percentage of men survived:   " , train_data[train_data.Sex == 'male'].Survived.sum()/train_data[train_data.Sex == 'male'].Survived.count())


# **survival percentage of women is more than the men.**

# In[ ]:


pd.crosstab(train_data.Pclass, train_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# **from this we can inferred that the survival rate decreases with the class.******

# In[ ]:


print("% of survivals in") 
print("Pclass=1 : ", train_data.Survived[train_data.Pclass == 1].sum()/train_data[train_data.Pclass == 1].Survived.count())
print("Pclass=2 : ", train_data.Survived[train_data.Pclass == 2].sum()/train_data[train_data.Pclass == 2].Survived.count())
print("Pclass=3 : ", train_data.Survived[train_data.Pclass == 3].sum()/train_data[train_data.Pclass == 3].Survived.count())


# In[ ]:


pd.crosstab([train_data.Sex, train_data.Survived], train_data.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# **From this we can infer that women of class 1 and class 2 survived most and nearly all men of class 1 and class 2 died.**

# In[ ]:


pd.crosstab([train_data.Survived], [train_data.Sex, train_data.Pclass, train_data.Embarked], margins=True)


# In[ ]:


for df in [train_data, test_data]:
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i


# In[ ]:


print(train_data[['Age' , 'Age_bin']].head(10))


# In[ ]:


test_data['Survived'] = 0
test_data.loc[ (test_data.Sex == 'female'), 'Survived'] = 1
test_data.loc[ (test_data.Sex == 'female') & (test_data.Pclass == 3) & (test_data.Embarked == 'S') , 'Survived'] = 0


# In[ ]:


sns.distplot(train_data['Fare'])
plt.show()


# In[ ]:


for df in [train_data, test_data]:
    df['Fare_bin']=np.nan
    for i in range(12,0,-1):
        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i


# In[ ]:


sns.catplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=train_data)
plt.show()


# In[ ]:


pd.crosstab([train_data.Sex, train_data.Survived], [train_data.Fare_bin, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


pd.crosstab([train_data.Sex, train_data.Survived], [train_data.Age_bin, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


pd.crosstab([train_data.Sex, train_data.Survived], [train_data.SibSp, train_data.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


test_data.loc[ (test_data.Sex == 'female') & (test_data.SibSp > 7) , 'Survived'] = 0


# In[ ]:


sns.catplot('Parch','Survived', col='Pclass' , row = 'Sex', kind='point', data=train_data)
plt.show()


# In[ ]:



test_data.loc[ (test_data.Sex == 'female') & (test_data.SibSp > 7) , 'Survived'] = 0


# In[ ]:


test_data.drop(['Survived'],axis=1,inplace=True)


# **Now Data Wrangling.** 
# i am copying the train_data and test_data in train1 and test1

# In[ ]:


train1 = train_data.copy()
test1 = test_data.copy()


# **Get Dummies to convert categorical data into Numerical data**

# In[ ]:


train1 = pd.get_dummies(train1, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)


# In[ ]:


train1.head()


# **Now we don't need Passenger id, Name, Ticket, Cabin, Age bin, Fare bin,**

# In[ ]:


train1.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)


# In[ ]:


train1.head()


# In[ ]:


train1.dropna(inplace=True)


# **We have deleted all the nan values**

# In[ ]:


train1.info()


# In[ ]:


passenger_id = test1['PassengerId']
test1 = pd.get_dummies(test1, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
test1.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)


# In[ ]:


test1.head()


# In[ ]:


test1.info()


# **Correlation matrix**

# In[ ]:


correlation = train1.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(correlation, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# **From this matrix we can infer that survived vs age,survived vs survived vs sex_male and survived vs pclass_3  are negatively correlated**

# **Standard scalar for training data**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train1.drop('Survived',axis=1))
scaled_features = scaler.transform(train1.drop('Survived',axis=1))
train1.sc = pd.DataFrame(scaled_features, columns=train1.columns[:-1])


# In[ ]:


test1.info()


# **Standard scalar for test data**

# In[ ]:


test1.fillna(test1.mean(), inplace=True)
scaled_features = scaler.transform(test1)
test1.sc = pd.DataFrame(scaled_features, columns=test1.columns)


# **train and test data split,  70% for training and 30% for testing**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train1.drop('Survived',axis=1), train1['Survived'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train1.sc,train1['Survived'], test_size=0.30, random_state=101)  


# In[ ]:





# **submitting the data**

# **first for unscaled data**

# In[ ]:


X_train_all = train1.drop('Survived',axis=1)
y_train_all = train1['Survived']
X_test_all = test1


# **for scaled data**

# In[ ]:


X_test_all.fillna(X_test_all.mean(), inplace=True)


# In[ ]:


X_test_all.head()


# **Now Fitting the model using scikit learn library**

# 1) **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)


# In[ ]:


pred_logreg


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[ ]:


logreg.fit(X_train_all, y_train_all) #Train for all data
pred_all_logreg = logreg.predict(X_test_all)


# In[ ]:


sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = test_data['PassengerId']
sub_logreg['Survived'] = pred_all_logreg


# **2) Gaussian Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# **3) KNN Classifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[ ]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[ ]:


knn.fit(X_train_all, y_train_all)
pred_all_knn = knn.predict(X_test_all)


# In[ ]:


sub_knn = pd.DataFrame()
sub_knn['PassengerId'] = test_data['PassengerId']
sub_knn['Survived'] = pred_all_knn


# **4) Decision Tree classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# **5) Random forest classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)


# In[ ]:


pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))

