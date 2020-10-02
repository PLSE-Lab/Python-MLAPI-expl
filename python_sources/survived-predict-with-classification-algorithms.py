#!/usr/bin/env python
# coding: utf-8

# ## Survived Predict with Classification Algorithms
# 
# Hello :) I'm recently learning data science and machine learning algorithms. I'm working making practice what I learn. I used python language and Titanic Disaster dataset. I hope it's benefit for you. Please consider upvoting if this is useful to you! :) 
# 
# ### CONTENTS:
# [1. Exploratory Data Analysis](#1) <br/>
# [2. Cleaning Data](#2) <br/>
# [3. Classification Algorithms](#3) <br/>
# &nbsp;&nbsp;&nbsp; [A. Logistic Regression Algorithm](#3.1) <br/>
# &nbsp;&nbsp;&nbsp; [B. K-Nearest Neighbour (KNN) Algorithm](#3.2) <br/>
# &nbsp;&nbsp;&nbsp; [C. Support Vector Machine (SVM) Algorithm](#3.3) <br/>
# &nbsp;&nbsp;&nbsp; [D. Naive Bayes Algorithm](#3.4) <br/>
# &nbsp;&nbsp;&nbsp; [E. Decision Tree Algorithm](#3.5) <br/>
# &nbsp;&nbsp;&nbsp; [F. Random Forest Algorithm](#3.6) <br/>
# &nbsp;&nbsp;&nbsp; [G. Evaluation Classification Models](#3.7) <br/>

# <a id="1"></a>
# ### 1. Exploratory Data Analysis

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import seaborn as sns 
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# In[ ]:


#read data
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")


# In[ ]:


#train sample
data_train.sample(5)


# In[ ]:


#test sample
data_test.sample(5)


# In[ ]:


data_train.info()
print("---------------------------------")
data_test.info()


# In[ ]:


#train columns
data_train.columns


# In[ ]:


#test column
data_test.columns


# In[ ]:


data_train.describe(include="all")


# In[ ]:


data_test.describe(include="all")


# In[ ]:


#missing values
print(pd.isnull(data_train).sum())
print("-------------------------")
print(pd.isnull(data_test).sum())


# In[ ]:


# train survived count
survived = data_train.Survived
plt.figure(figsize=(7,5))
sns.countplot(survived)
plt.title("Survived",color='blue',fontsize=15)
plt.show()


# In[ ]:


passanger_class = data_train.Pclass
plt.figure(figsize=(7,5))
sns.countplot(passanger_class)
plt.title("data_train Passanger Class",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


passanger_class = data_test.Pclass
plt.figure(figsize=(7,5))
sns.countplot(passanger_class)
plt.title("data_test Passanger Class",color = 'blue',fontsize=15)
plt.show()


# I've splitted according to prefix of name. I've added them becoming a new column. 

# In[ ]:


data_train['Title'] = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data_train['Title'] = data_train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
data_train['Title'] = data_train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')
data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')
data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')

passanger_name = data_train.Title
plt.figure(figsize=(10,7))
sns.countplot(passanger_name)
plt.title("data_train Passanger Name",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


data_test['Title'] = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data_test['Title'] = data_test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
data_test['Title'] = data_test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')
data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')
data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')

passanger_name = data_test.Title
plt.figure(figsize=(10,7))
sns.countplot(passanger_name)
plt.title("data_test Passanger Name",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


gender = data_train.Sex
plt.figure(figsize=(7,5))
sns.countplot(gender)
plt.title("data_train Gender",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


gender = data_test.Sex
plt.figure(figsize=(7,5))
sns.countplot(gender)
plt.title("data_test Gender",color = 'blue',fontsize=15)
plt.show()


# I've created age groups. I've added them becoming a new column. 

# In[ ]:


data_train['AgeGroup'] = ["Baby" if (i>=0 and i<5) else "Child" if (i>=5 and i<12) else "Teenager" if (i>=12 and i<18) 
                          else "Student" if(i>=18 and i<24) else "Young Adult" if(i>=24 and i<35) 
                          else "Adult" if(i>=35 and i<60) else "Senior" if(i>=60) else "Unknown" 
                          for i in data_train.Age ]

passanger_ageGroup = data_train.AgeGroup
plt.figure(figsize=(10,7))
sns.countplot(passanger_ageGroup)
plt.title("data_train Passanger AgeGroup",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


data_test['AgeGroup'] = ["Baby" if (i>=0 and i<5) else "Child" if (i>=5 and i<12) else "Teenager" if (i>=12 and i<18) 
                          else "Student" if(i>=18 and i<24) else "Young Adult" if(i>=24 and i<35) 
                          else "Adult" if(i>=35 and i<60) else "Senior" if(i>=60) else "Unknown" 
                          for i in data_test.Age ]

passanger_ageGroup = data_test.AgeGroup
plt.figure(figsize=(10,7))
sns.countplot(passanger_ageGroup)
plt.title("data_test Passanger AgeGroup",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


passanger_sibsp = data_train.SibSp
plt.figure(figsize=(10,7))
sns.countplot(passanger_sibsp)
plt.title("data_train Passanger SibSp")
plt.show()


# In[ ]:


passanger_sibsp = data_test.SibSp
plt.figure(figsize=(10,7))
sns.countplot(passanger_sibsp)
plt.title("data_test Passanger SibSp")
plt.show()


# In[ ]:


passanger_parch = data_train.Parch
plt.figure(figsize=(10,7))
sns.countplot(passanger_parch)
plt.title("data_train Passanger Parch")
plt.show()


# In[ ]:


passanger_parch = data_test.Parch
plt.figure(figsize=(10,7))
sns.countplot(passanger_parch)
plt.title("data_test Passanger Parch")
plt.show()


# In[ ]:


data_train.Fare.describe()


# In[ ]:


passanger_fare = ['above100$' if i>=100 else '32between100$' if (i<100 and i>=32) else 'Free' if i==0 else 'below32$' for i in data_train.Fare]
plt.figure(figsize=(10,7))
sns.countplot(passanger_fare)
plt.title("data_train Passanger Fare",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


data_test.Fare.describe()


# In[ ]:


passanger_fare_test = ['above100$' if i>=100 else '35between100$' if (i<100 and i>=35) else 'Free' if i==0 else 'below35$' for i in data_test.Fare]
plt.figure(figsize=(10,7))
sns.countplot(passanger_fare_test)
plt.title("data_test Passanger Fare",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


passanger_embarked = data_train.Embarked
plt.figure(figsize=(10,7))
sns.countplot(passanger_embarked)
plt.title("data_train Passanger Embarked",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


passanger_embarked = data_test.Embarked
plt.figure(figsize=(10,7))
sns.countplot(passanger_embarked)
plt.title("data_test Passanger Embarked",color = 'blue',fontsize=15)
plt.show()


# <a id=2></a>
# ### Cleaning Data

# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# I've given a number to classify.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
data_train['Title'] = data_train['Title'].map(title_mapping)
data_train['Title'] = data_train['Title'].fillna(0)

data_test['Title'] = data_test['Title'].map(title_mapping)
data_test['Title'] = data_test['Title'].fillna(0)

#data_test.Title.head()
#data_train.Title.head()


# In[ ]:


data_train.Sex = [0 if i=="male" else 1 for i in data_train.Sex]
data_test.Sex = [0 if i=="male" else 1 for i in data_test.Sex]
data_test.Sex.head()
data_train.Sex.head()


# In[ ]:


data_train['Age'] = data_train['Age'].fillna(0)
data_test['Age'] = data_test['Age'].fillna(0)
print("Missing train age value count:",pd.isnull(data_test.Age).sum())
print("Missing test age value count:",pd.isnull(data_train.Age).sum())


# In[ ]:


title_mapping_age = {"Baby":1, "Child":2, "Teenager":3, "Student":4, "Young Adult":5, "Adult":6, "Senior":7, "Unknow":0}
data_train['AgeGroup'] = data_train['AgeGroup'].map(title_mapping_age)
data_train['AgeGroup'] = data_train['AgeGroup'].fillna(0)
data_test['AgeGroup'] = data_test['AgeGroup'].map(title_mapping_age)
data_test['AgeGroup'] = data_test['AgeGroup'].fillna(0)
#data_test.AgeGroup.head()
#data_train.AgeGroup.head()


# I found passengers, which joined with their family or alone, by picking up counts of sibsp and parch.

# In[ ]:


#train
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']
data_train['IsAlone'] = [0 if i==0 else 1 for i in data_train['FamilySize']]# 0 equals alone 1 equals family
data_train["CabinBool"] = (data_train["Cabin"].notnull().astype('int'))
data_train['FareBand'] = [4 if i=='above100$' else 3 if i=='32between100$' else 2 if i=='Free' else 1 for i in passanger_fare]
data_train.Embarked = [0 if i=="S" else 1 if i=="C" else 2 if i=="Q" else 0 for i in data_train.Embarked]
data_train['Embarked'] = data_train['Embarked'].fillna(0)
print(pd.isnull(data_train.Embarked).sum())

#test
data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch']
data_test['IsAlone'] = [0 if i==0 else 1 for i in data_test['FamilySize']]# 0 equals alone 1 equals family
data_test["CabinBool"] = (data_test["Cabin"].notnull().astype('int'))
data_test['FareBand'] = [4 if i=='above100$' else 3 if i=='35between100$' else 2 if i=='Free' else 1 for i in passanger_fare_test]

data_test.Embarked = [0 if i=="S" else 1 if i=="C" else 2 if i=="Q" else 0 for i in data_test.Embarked]
print(pd.isnull(data_test.Embarked).sum())

data_train.head()


# In[ ]:


data_train_x = data_train.drop(['PassengerId','Survived','Name','Cabin','SibSp','Parch','Age','Fare','Ticket'],axis=1)
data_train_y = data_train.Survived
data_train_x.head()


# In[ ]:


data_test_x = data_test.drop(['PassengerId','Name','Cabin','SibSp','Parch','Age','Fare','Ticket'],axis=1)
data_test_x.head()


# In[ ]:


#normalization
data_train_x = (data_train_x - np.min(data_train_x))/(np.max(data_train_x)-np.min(data_train_x)).values
data_train_x.head()


# In[ ]:


#normalization
data_test_x = (data_test_x - np.min(data_test_x))/(np.max(data_test_x)-np.min(data_test_x)).values
data_test_x.head()


# <a id=3.1></a>
# ###  Logistic Regression Algorithm

# In[ ]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size=0.2,random_state=42)
column = ["Logistic Regression","KNN","SVM","Native Bayes","Decision Tree","Random Forest"]
accuracy_list = []
predict_list = []


# In[ ]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train,y_train)
print("test accuracy {}".format(reg.score(x_test,y_test)))
accuracy_list.append(reg.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = reg.predict(x_test)
y_true = y_test

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2))

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.2"></a>
# ### K-Nearest Neighbour (KNN) Algorithm

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
# find best k value 
score_list = []
for each in range(1,15):
    knn = KNeighborsClassifier(n_neighbors = each)
    knn.fit(x_train,y_train)
    score_list.append(knn.score(x_test,y_test))
 
plt.figure(figsize=(10,7))
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))
accuracy_list.append(knn.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = knn.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2))

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.3"></a>
# ### Support Vector Machine (SVM) Algorithm

# In[ ]:


from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
print("print accuracy of svm algo: ",svm.score(x_test,y_test))
accuracy_list.append(svm.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = svm.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2))

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.4"></a>
# ### Naive Bayes Algorithm

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
accuracy_list.append(nb.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = nb.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2)) 

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.5"></a>
# ### Decision Tree Algorithm

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score: ", dt.score(x_test,y_test))
accuracy_list.append(dt.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = dt.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2))

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.6"></a> 
# ### Random Forest Algorithm

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200,random_state = 42)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))
accuracy_list.append(rf.score(x_test,y_test))


# In[ ]:


#Estimated number of survivors
y_pred = rf.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
predict_list.append(cm.item(0)+cm.item(2))

#cm visualization
f, ax = plt.subplots(figsize =(10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# <a id="3.7"></a>
# ### Evaluation Classification Models

# In[ ]:


#Classifier Accuracy
f,ax = plt.subplots(figsize = (15,7))
sns.barplot(x=accuracy_list,y=column,palette = sns.cubehelix_palette(len(accuracy_list)))
plt.xlabel("Accuracy")
plt.ylabel("Classifier")
plt.title('Classifier Accuracy')
plt.show()


# In[ ]:


#Classifier Predict Survived Count
f,ax = plt.subplots(figsize = (15,7))
sns.barplot(x=predict_list,y=column,palette = sns.cubehelix_palette(len(accuracy_list)))
plt.xlabel("Predict Survived Count")
plt.ylabel("Classifier")
plt.title('Classifier Predict Survived Count')
plt.show()


# In[ ]:


#set ids as PassengerId and predict survival 
ids = data_test['PassengerId']
predict = knn.predict(data_test_x)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predict})
output.to_csv('submission.csv', index=False)

