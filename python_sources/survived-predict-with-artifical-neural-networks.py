#!/usr/bin/env python
# coding: utf-8

# # Survived Predict with Artifical Neural Networks
# 
# Hi everybody! This kernel I'm gonna analyze Titanic Disaster Dataset by using ANN(Artifical Neural Network) I'm gonna use the Python language. I hope it's benefit for you. If this kernel is useful for you, please don't forget upvote it. I will be waiting your comment this kernel. Your comment is important in order to improving myself. So Let's go! :)
# 
# <img src="https://i.milliyet.com.tr/MolatikDetayBig/2020/04/14/fft371_mf33115214.Jpeg"/>
# 
# 
# ## CONTENTS
# 
# [1. Libraries](#1) <br/>
# [2. Exploratory Data Analysis](#2) <br/>
# [3. Preparing Data](#3) <br/>
# [4. Train and Test Split](#4) <br/>
# [5. Create Artifical Neural Network Model](#5) <br/>
# [6. Model Evaluation using Confusion Matrix](#6) <br/>
# [7. Conclusion](#7)

# <a id="1"></a>
# ## Libraries

# You can find the libraries which I use. :)

# In[ ]:


# EDA and Preparing Data libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Spliting data and creating model libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential #initialize neural network library
from keras.layers import Dense #build our layers library

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="2"></a>
# ## Exploratory Data Analysis
# 
# 
# In this section, We will analyze feature of disaster data. We will visualize our datas and will realize some information extractions. 

# In[ ]:


data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_train.head()


# In[ ]:


data_train.info()


# The section code in above, You can find some information. There are 12 feature in the disaster data. It has information about 891 people. Five feature are integer type and other five feature are object type. Lastly, two feature are float type. Some features contain missing value.  

# <a id="2.1.1"></a>
# #### Survived
# 
# We can observe number of death more than number of survive. Well Is number of death/survived related to passenger class? 

# In[ ]:


sns.countplot(data_train["Survived"])
plt.show()


# <a id="2.2.2"></a>
# #### Pclass
# 
# Waoow, There are the most third class passenger. 

# In[ ]:


sns.countplot(data_train["Pclass"])


# <a id="2.2.3"></a>
# #### Title / Name
# 
# 

# In[ ]:


data_train['Title'] = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data_train['Title'] = data_train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
data_train['Title'] = data_train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')
data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')
data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')

plt.figure(figsize=(10,7))
sns.countplot(data_train.Title)
plt.title("data_train Passanger Name",color = 'blue',fontsize=15)
plt.show()


# #### Sex

# In[ ]:


sns.countplot(data_train["Sex"])


# In[ ]:


g = sns.FacetGrid(data_train,row="Sex",col="Pclass")
g.map(sns.countplot,"Survived")
plt.show()


# In the graphs above, we see survived number according to sex and passenger class. We can observe death man number more than death woman. Also, passanger class is effective in the death number. You know the disaster. First class passengers are saved, Third class passengers are sunk into water. :(

# #### Age

# In[ ]:


data_train["Age"].describe()


# In[ ]:


sns.distplot(data_train["Age"])


# #### SibSp

# In[ ]:


sns.countplot(data_train["SibSp"])


# #### Parch

# In[ ]:


sns.countplot(data_train["Parch"])


# As we see SibSp and Parch features. The number of alone passangers are quite more. Parch and SibSp represent parent, child(Parch) and sibling,spouse(SibSp). 

# #### Fare

# In[ ]:


data_train["Fare"].describe()


# In[ ]:


fare = ['above100$' if i>=100 else '32between100$' if (i<100 and i>=32) else 'Free' if i==0 else 'below32$' for i in data_train["Fare"]]
plt.figure(figsize=(10,7))
sns.countplot(fare)
plt.title("data_train Passanger Fare",color = 'blue',fontsize=15)
plt.show()


# Mean train fare is 32$. In the graph above we see most passanger who buy ticket below 32$. Probably this passangers can be 3th. class. 

# #### Embarked

# In[ ]:


sns.countplot(data_train["Embarked"])


# <a id="3"></a>
# ## Preparing Data

# In this part, We gonna prepare train and test dataset.

# ### Train Dataset

# In[ ]:


data_train.head()


# In[ ]:


#missing value
print(pd.isnull(data_train).sum())


# In[ ]:


# drop name and ticket
data_train = data_train.drop(["PassengerId","Name","Ticket"],axis=1)

# Sex
data_train["Sex"] = data_train["Sex"].replace("male",1)
data_train["Sex"] = data_train["Sex"].replace("female",2)

# Age
data_train["Age"] = data_train["Age"].replace(np.nan,data_train["Age"].median())

# Fare
data_train["Fare"] = data_train["Fare"].replace(np.nan,data_train["Fare"].median())

# Cabin
data_train.loc[data_train["Cabin"].str[0] == 'A', 'Cabin'] = 1
data_train.loc[data_train["Cabin"].str[0] == 'B', 'Cabin'] = 2
data_train.loc[data_train["Cabin"].str[0] == 'C', 'Cabin'] = 3
data_train.loc[data_train["Cabin"].str[0] == 'D', 'Cabin'] = 4
data_train.loc[data_train["Cabin"].str[0] == 'E', 'Cabin'] = 5
data_train.loc[data_train["Cabin"].str[0] == 'F', 'Cabin'] = 6
data_train.loc[data_train["Cabin"].str[0] == 'G', 'Cabin'] = 7
data_train.loc[data_train["Cabin"].str[0] == 'T', 'Cabin'] = 8
data_train["Cabin"] = data_train["Cabin"].fillna(data_train["Cabin"].mean())

# Embarked
data_train["Embarked"] = data_train["Embarked"].replace("S",1)
data_train["Embarked"] = data_train["Embarked"].replace("C",2)
data_train["Embarked"] = data_train["Embarked"].replace("Q",3)
data_train["Embarked"] = data_train["Embarked"].replace(np.nan,data_train["Embarked"].median())

# Title
data_train["Title"] = data_train["Title"].replace("Mr",1)
data_train["Title"] = data_train["Title"].replace("Mrs",2)
data_train["Title"] = data_train["Title"].replace("Miss",3)
data_train["Title"] = data_train["Title"].replace("Master",4)
data_train["Title"] = data_train["Title"].replace("Rare",5)
data_train["Title"] = data_train["Title"].replace("Royal",6)

#Family Size
data_train['FamilySize'] = data_train['SibSp'] + data_train['Parch']

data_train.drop(["SibSp","Parch"],axis=1,inplace=True)


# ### Test Dataset

# In[ ]:


data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
data_test.head()


# In[ ]:


print(pd.isnull(data_test).sum())


# In[ ]:


#Title
data_test['Title'] = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data_test['Title'] = data_test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
data_test['Title'] = data_test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')
data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')
data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')

# Sex
data_test["Sex"] = data_test["Sex"].replace("male",1)
data_test["Sex"] = data_test["Sex"].replace("female",2)

# Age
data_test["Age"] = data_test["Age"].replace(np.nan,data_test["Age"].median())

# Fare
data_test["Fare"] = data_test["Fare"].replace(np.nan,data_test["Fare"].median())

# Cabin
data_test.loc[data_test["Cabin"].str[0] == 'A', 'Cabin'] = 1
data_test.loc[data_test["Cabin"].str[0] == 'B', 'Cabin'] = 2
data_test.loc[data_test["Cabin"].str[0] == 'C', 'Cabin'] = 3
data_test.loc[data_test["Cabin"].str[0] == 'D', 'Cabin'] = 4
data_test.loc[data_test["Cabin"].str[0] == 'E', 'Cabin'] = 5
data_test.loc[data_test["Cabin"].str[0] == 'F', 'Cabin'] = 6
data_test.loc[data_test["Cabin"].str[0] == 'G', 'Cabin'] = 7
data_test.loc[data_test["Cabin"].astype(str).str[0] == 'T', 'Cabin'] = 8
data_test["Cabin"] = data_test["Cabin"].fillna(int(data_test["Cabin"].mean()))

# Embarked
data_test["Embarked"] = data_test["Embarked"].replace("S",1)
data_test["Embarked"] = data_test["Embarked"].replace("C",2)
data_test["Embarked"] = data_test["Embarked"].replace("Q",3)

# Title
data_test["Title"] = data_test["Title"].replace("Mr",1)
data_test["Title"] = data_test["Title"].replace("Mrs",2)
data_test["Title"] = data_test["Title"].replace("Miss",3)
data_test["Title"] = data_test["Title"].replace("Master",4)
data_test["Title"] = data_test["Title"].replace("Rare",5)
data_test["Title"] = data_test["Title"].replace("Royal",6)

#Family Size
data_test['FamilySize'] = data_test['SibSp'] + data_test['Parch']

# drop passenger id, name,ticket, sibsp and parch
data_test_x = data_test.drop(["PassengerId","Name","Ticket","SibSp","Parch"],axis=1)


# <a id="4"></a>
# ## Train and Test Split
# 
# Our train data splitted as train and test data in order to educate correctly. Train data creates %80 percent of main train data and test data creates %20 percent of main train data. In the below, you see number of passenger train and test datas. You can set percentage datas according to yourself model.

# In[ ]:


X = data_train.drop(["Survived"],axis=1)
Y = data_train["Survived"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)


# <a id="5"></a>
# ## Create Artifical Neural Network Model
# 
# I used to artifical neural network model. I wrote about deep learning beginner tutorial on Kaggle. You can reach it when you click [this](https://www.kaggle.com/ecemboluk/deep-learning-tutorial-on-sign-language-digits). I used to 6 layer together input and output layer. My model has 4 hidden layer. My hidden layers contain 60 units in total. As activation function I prefered relu. Because it is more faster than other activation function. I prefer adam algorithm as optimizer algorithm. Epochs, layer number, unit number are hyperparameters. Nobody don't know which values is true for their model. You can find this by trying. :)

# In[ ]:


classifier = Sequential() # initialize neural network
classifier.add(Dense(units = 128, activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model = classifier.fit(x_train,y_train,epochs=600)
mean = np.mean(model.history['accuracy'])
print("Accuracy mean: "+ str(mean))


# <a id="6"></a>
# ## Model Evaluation using Confusion Matrix

# In[ ]:


y_predict = classifier.predict(x_test)
cm = confusion_matrix(y_test,np.argmax(y_predict, axis=1))

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, ax=ax)


# ## Conclusion
# 
# Artifical Neural Netwok succeed quite prediction survived.    

# In[ ]:


ids = data_test['PassengerId']
predict = classifier.predict(data_test_x)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': np.argmax(predict,axis=1)})
output.to_csv('submission.csv', index=False)

