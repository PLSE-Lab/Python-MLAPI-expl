#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Data Loading**

# Loading the datasets ie: train.csv and test.csv 

# In[ ]:


ttrain = pd.read_csv("../input/titanic/train.csv")
ttrain.head()


# In[ ]:


ttest = pd.read_csv("../input/titanic/test.csv")
ttest.head()


# Data cleaning has to be done by identifying number of missing values within the datasets.

# In[ ]:


print("Shape of Training set")
print(ttrain.shape)
print("\nShape of Testing set")
print(ttest.shape)


# In[ ]:


print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())


# # **Data Cleaning**

# The training set consists of 891 values and testing set consists of 418. 
# From the above output we know that values from 'Age and Cabin' are missing from the ttrain as well as ttest dataset.
# Along with 'Age and Cabin' the values from features 'Embarked' and 'Fare' are missing from the ttrain and ttest dataset respectively.
# The output below shows the percentage of missing values from each of the dataset.

# In[ ]:


print("Missing value in training set")
train = round(((ttrain.isnull().sum()*100)/(ttrain.shape[0])),3).sort_values(ascending=False).head(5)
print(train)

print("\nMissing value in test set")
test = round(((ttest.isnull().sum()*100)/(ttrain.shape[0])),3).sort_values(ascending=False).head(5)
print(test)


# The training set consists of 891 values.
# The missing values of the features from the training set are handelled as follows:
# * Age = Since the percentage of missing values is less, the mean of the column is substitued with the missings values
# * Cabin = Since the percentage of missing values is very high the column is dropped
# * Embarked = Since the percentage of missing values is very low, the higgest frequency value is substitued with the missings values

# In[ ]:


#Age
train_mean = round(ttrain['Age'].mean(),0) 
ttrain['Age'] = ttrain['Age'].fillna(train_mean)

#Cabin
ttrain.drop(['Cabin'],inplace=True,axis=1)
ttrain.head()


# In[ ]:


#Embarked
print(ttrain['Embarked'].value_counts())

value = ttrain['Embarked'].value_counts().index[0]

#Since the frequency of 'S' is the highest we substitute the missing values with 'S'
ttrain['Embarked'] = ttrain['Embarked'].fillna(value) 


# The testing set consists of 418.
# The missing values of the features from the testing set are handled as follows:
# * Age = Since the percentage of missing values is low, the mean of the column is substitued with the missings values
# * Cabin = Since the percentage of missing values is very high the column is dropped
# * Fare = Since the percentage of missing values is very low, the higgest frequency value is substitued with the missings values

# In[ ]:


#Age
test_mean = round(ttest['Age'].mean(),0) 
ttest['Age'] = ttest['Age'].fillna(test_mean)

#Cabin
ttest.drop(['Cabin'],inplace=True,axis=1)
ttest.head()


# In[ ]:


#Fare
print(ttest['Fare'].mode())
test = ttest['Fare'].mode()[0]

#Since the frequency of '7.7500' is the highest we substitute the missing values with '7.7500'
ttest['Fare'] = ttest['Fare'].fillna(test) 


# In[ ]:


print("Missing value in training set")
print(ttrain.isnull().sum())
print("\nMissing value in test set")
print(ttest.isnull().sum())


# All the missing values within the datasets have been dealt with and the datasets are clean. Thus we now perform Exploratory Data Analysis (EDA).

# # **Exploratory Data Analysis (Data Visualization)**

# In[ ]:


ttrain.head()


# In[ ]:


print(ttrain.groupby(['Survived']).count()['PassengerId'])
sns.countplot(x='Survived',data=ttrain)


# The number of survivors are very few.

# In[ ]:


sns.countplot(x='Sex',data=ttrain)


# In[ ]:


print(ttrain.groupby(['Pclass']).mean()['Survived'],"\n")
sns.barplot(x='Pclass',y='Survived',data=ttrain)


# It is observed that there are more survivors from passanger class 1.

# In[ ]:


print(ttrain.groupby(['Pclass','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='Pclass',hue='Survived',data=ttrain)


# Passengers from the 3rd passanger class were not able to survive.

# In[ ]:


print(ttrain.groupby(['Pclass','Sex']).count()['PassengerId'],"\n")
sns.countplot(x='Pclass',hue='Sex',data=ttrain)


# It is observed that there are many male passangers and the maximum number of passangers belonged to the 3rd passanger class.

# In[ ]:


print(ttrain.groupby(['Sex','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='Sex',hue='Survived',data=ttrain)


# There are a large number of female passangers who were able to survive.

# In[ ]:


print(ttrain.groupby(['Pclass','Sex']).mean()['Survived'],"\n")
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=ttrain)


# The above plot shows that many survivors are females among Pclass 1 and 2 .

# In[ ]:


print(ttrain.groupby(['SibSp','Survived']).count()['PassengerId'],"\n")
sns.countplot(x='SibSp',hue='Survived',data=ttrain)


# A large number of passengers who had been travelling alone were not able to survive.

# In[ ]:


sns.stripplot(x='Survived',y='Age',data=ttrain,jitter=True)


# Passengers between the age of 0-65 have survived with an exception of 1 passenger of age 80.

# In[ ]:


tfare = sns.FacetGrid(data=ttrain,hue='Survived')
tfare.map(sns.kdeplot,'Fare')


# The survivors include those who have paid a higher conveyance fare.

# In[ ]:


sns.jointplot(x='Age',y='Fare',data=ttrain,kind='reg')


# There is no relation between Age and Fare paid for conveyance.
# Titanic has many passengers between the age 20-40.
# Most of the tickets purchased costs less comparatively. 

# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=ttrain)


# Passangers that had board the ship from Southampton are unlikely to survive.

# In[ ]:


sns.countplot(x='Parch',hue='Survived',data=ttrain)


# Parch can not help us to determine any pattern of survivors.

# In[ ]:


target = ttrain.groupby(['Embarked','Pclass','Survived'])

plt.figure(figsize=(8,8))
target.count()['PassengerId'].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Grouped Conditions")
plt.ylabel("Total Count")

plt.show()


# The majority survivers include those who board the titanic from Southampton (S) and belonged to the 1-2 class. Unfortunately the passengers from the 3 class who board from the same port did not surive

# In[ ]:


correlate = ttrain.corr()
correlate


# In[ ]:


sns.heatmap(correlate)


# The relationship between each and every feature can be represented by identifying the postive and negative correalation between them and then visualizing them into a heatmap 

# # **Summary of Observation**

# *  Less than half of the total number of passenger had survived the titanic accident.
# *  The number of male passengers were more than the female passengers.
# *  Maximum number of passengers traveled in the 3rd passenger class as the fare of the same was economically feasible.
# *  Many passengers were travelling alone.
# *  There were 3 ports from where the passenger could board.
# *  All the passengers travelling were below the age of 75. Although there was a single exception of a passenger of age 80.
# *  The maximum cost of the ticket for the titanic was above 500$.
# *  The survivors majorly included:
# >     1. Female passangers.
# >     2. Passangers from the 1st passenger class.
# >     3. Passanger with sibling, spouse, parent or children. 

# # **Feature Engineering**

# Before preparing any model it is essential to process the dataset and create a dataset with numeric values.

# In[ ]:


print("Columns of training set")
print(ttrain.columns)
print("\nColumns of testing set")
print(ttest.columns)


# Passengers ID and Ticket are unique values for each individual, we can not use them to build a model or predict any future outcomes. Thus it is desirable to drop these columns from the training and the testing set.

# In[ ]:


data = [ttrain, ttest]
for dataset in data:
    dataset.drop(['PassengerId','Ticket'],axis=1,inplace=True)


# We notice that sibling, spouse, parents and children are relations for a passenger, therefore we can club them into 1 single feature. 

# In[ ]:


for dataset in data:
    dataset['Relation'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['Relation'] > 0, 'Travelled_alone'] = 'No'
    dataset.loc[dataset['Relation'] == 0, 'Travelled_alone'] = 'Yes'


# In[ ]:


print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())


# The training and testing set includes object and floating values. It is a good practice to always feed in numerical (Integer) values into a predictive data model. Hence we know convert the two objects into numerical value. We also notice that we do not have any categorical value for the features Age and Fare, thus we seprate them into categories such that they can be easily used as an input to a model. 

# In[ ]:


for dataset in data:
    dataset['Travelled_alone'] = dataset['Travelled_alone'].map({'No':0,'Yes':1})
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1})
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


def age(num): 
    
    if num <= 11: 
        return 0
  
    elif num > 11 and num <= 18:
        return 1
    
    elif num > 18 and num <= 22:
        return 2
    
    elif num > 22 and num <= 27:
        return 3
    
    elif num > 27 and num <= 33:
        return 4
    
    elif num > 33 and num <= 40:
        return 5
    
    elif num > 40 and num <= 66:
        return 6
    
    else: 
        return 7
    
    
def fare(num): 
    
    if num <= 7.91: 
        return 0
  
    elif num > 7.91 and num <= 33:
        return 1
    
    elif num > 33 and num <= 66:
        return 2
    
    elif num > 66 and num <= 99:
        return 3
    
    elif num > 99 and num <= 250:
        return 4
    
    elif num > 250 and num <= 360:
        return 5
   
    else: 
        return 6


# In[ ]:


for dataset in data:
    dataset['Age'] = dataset['Age'].apply(age)
    dataset['Fare'] = dataset['Fare'].apply(fare)


# In[ ]:


for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())


# In[ ]:


titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)


# In[ ]:


for dataset in data:
    dataset.drop(['Name'],axis=1,inplace=True)


# In[ ]:


print("Information of training set\n")
print(ttrain.info())
print("\nInformation of testing set\n")
print(ttest.info())


# # **Data Modeling**

# For the purpose of Data Modeling we need to split our data into training and test set.
# Once the split is done we can put our data into various models and check each the precision of each model.
# We select the model with the highest precision score.

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


X = ttrain.drop('Survived',axis=1)
y = ttrain['Survived']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# # 1. Logistic Regression

# In[ ]:


#Import Packages 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# In[ ]:


#Object creation and fitting of training set
model1 = LogisticRegression()
model1.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predict1 = model1.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predict1))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predict1))


# In[ ]:


#Accuracy Percentage
predict11 = round((model1.score(X_test, y_test)*100),0)
print("Precision of Logistic Regression is: ",predict11,"%") 


# # 2. K-Nearest Neighbour

# In[ ]:


#Import Packages 
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


#Object creation and fitting of training set
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionsknn = knn.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionsknn))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionsknn))


# In[ ]:


#Accuracy Percentage
knnp = round((knn.score(X_test, y_test)*100),0)
print("Precision of K Nearest Neighbors is: ",knnp,"%") 


# # 3. Decision Tree

# In[ ]:


#Import Packages 
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#Object creation and fitting of training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictiondt = dtree.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiondt))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiondt))


# In[ ]:


#Accuracy Percentage
dtt = round((dtree.score(X_test, y_test)*100),0)
print("Precision of Decision Tree is: ",dtt,"%") 


# # 4. Random Forest

# In[ ]:


#Import Packages 
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Object creation and fitting of training set
randfc = RandomForestClassifier(n_estimators=100)
randfc.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionrf = randfc.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionrf))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionrf))


# In[ ]:


#Accuracy Percentage
random = round((randfc.score(X_test, y_test)*100),0)
print("Precision of Random Forest is: ",random,"%") 


# # 5. Support Vector Machine

# In[ ]:


#Import Packages 
from sklearn.svm import SVC


# In[ ]:


#Object creation and fitting of training set
svcm = SVC()
svcm.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionsvc = svcm.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionsvc))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionsvc))


# In[ ]:


#Accuracy Percentage
svc = round((svcm.score(X_test, y_test)*100),0)
print("Precision of Support Vector Classifier: ",svc,"%")


# # 6. Gaussian Naive Bayes

# In[ ]:


#Import Packages 
from sklearn.naive_bayes import GaussianNB


# In[ ]:


#Object creation and fitting of training set
gaus = GaussianNB()
gaus.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictiongus = gaus.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiongus))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiongus))


# In[ ]:


#Accuracy Percentage
rig = round((gaus.score(X_test, y_test)*100),0)
print("Precision of Gaussian Naive Bayes is: ",rig,"%") 


# # **7. Logistic Regression CV**

# In[ ]:


#Object creation and fitting of training set
lgrcv = LogisticRegressionCV()
lgrcv.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionlgcv = lgrcv.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionlgcv))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionlgcv))


# In[ ]:


#Accuracy Percentage
lgcv = round((lgrcv.score(X_test, y_test)*100),0)
print("Precision of  Logistic Regression CV is: ",lgcv,"%") 


# # **8. Gradient Boosting Classifier**

# In[ ]:


#Import Packages 
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


#Object creation and fitting of training set
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictiongbc = gbc.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictiongbc))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictiongbc))


# In[ ]:


#Accuracy Percentage
gbcp = round((gbc.score(X_test, y_test)*100),0)
print("Precision of Gradient Boosting Classifier is: ",gbcp,"%") 


# # **9. Perceptron**

# In[ ]:


#Import Packages 
from sklearn.linear_model import Perceptron


# In[ ]:


#Object creation and fitting of training set
per = Perceptron(max_iter=6)
per.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predictionper = per.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predictionper))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predictionper))


# In[ ]:


#Accuracy Percentage
perr = round((per.score(X_test, y_test)*100),0)
print("Precision of Perceptron is: ",perr,"%") 


# # 10. Stochastic Gradient Descent (SGD)

# In[ ]:


#Import Packages 
from sklearn.linear_model import SGDClassifier


# In[ ]:


#Object creation and fitting of training set
model2 = SGDClassifier()
model2.fit(X_train,y_train)


# In[ ]:


#Creation of a prediction variable
predict2 = model2.predict(X_test)


# In[ ]:


#Accuracy Matrix
print("\nClassification Matrix")
print(classification_report(y_test,predict2))
print("\nConfusion Matrix")
print(confusion_matrix(y_test,predict2))


# In[ ]:


linr = round((model2.score(X_test, y_test)*100),0)
print("Precision of Stochastic Gradient Descent is: ",linr,"%") 


# # **Conclusion**

# In[ ]:


results = pd.DataFrame({'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'Support Vector Machines',  
                                  'Gausian Naive Baye', 'Logistic Regression CV', 'Stochastic Gradient Decent', 'Perceptron','Stochastic Gradient Descent'],
                        'Score': [predict11, knnp, dtt, random, svc, rig, lgcv, gbcp, perr,linr]
                      })

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(10)


# We observe the with the precision of the Decision Tree algorithm is the highest.Thus we can use Decision Tree for the future analysis of our dataset.   

# In[ ]:




