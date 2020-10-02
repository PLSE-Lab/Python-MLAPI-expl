#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
titatrain = pd.read_csv("../input/train.csv")  #importing the train data
titatest = pd.read_csv("../input/test.csv")   #importing the test data

# Any results you write to the current directory are saved as output.


# In[ ]:


titatrain.head()   #To view the head of the train data set 


# In[ ]:


titatest.head()     #To view the head of the test data set


# In[ ]:


titatrain.info()   #Finding the information(such as number of datas missing) on the training dataset


# In[ ]:


titatest.info()


# In[ ]:


titatrain.describe()   #To see the max, min and other details that will help us


# In[ ]:


titatrain.corr()


# In[ ]:


print(titatrain.keys())
print(titatest.keys())


# **EDA****We are going to explore the data and handle the missing values**

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(pd.isnull(titatrain))   #To see the missing values, the Cabin column has lots of missing values, The Age column might be useful to see what age people survived


# In[ ]:


sns.distplot(titatrain['Age'].dropna(),bins=50)  # ".dropna()" because, in our dataset there are NAN and we will get a Value Error if dropna is not present


# In[ ]:


sns.distplot(titatest['Age'].dropna(),bins=50)


# **Filling the data of Age for train and test dataset**
# We need to fill the Age, we cannot fill one value to each missing value. We use Pclass and Age to find the mean Age of the person in the Class he is present and fill that value.

# In[ ]:


sns.boxplot('Pclass','Age',data=titatrain)


# In[ ]:


PC1 = titatrain[titatrain['Pclass']==1]['Age'].mean()  #Finding the mean age of Pclass 1
PC2 = titatrain[titatrain['Pclass']==2]['Age'].mean()  #Finding the mean age of Pclass 2
PC3 = titatrain[titatrain['Pclass']==3]['Age'].mean()  #Finding the mean age of Pclass 3


# In[ ]:


def mis(cont):                   #Will be using this function for test dataset too.
    Age = cont[0]
    Pclass = cont[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return PC1
        elif Pclass == 2:
            return PC2
        else:
            return PC3
    else:
        return Age


# In[ ]:


titatrain['Age'] = titatrain[['Age','Pclass']].apply(mis,axis=1)


# In[ ]:


titatrain.info()


# In[ ]:


sns.boxplot('Pclass','Age',data=titatest)


# In[ ]:


PC1 = titatest[titatest['Pclass']==1]['Age'].mean()  #Finding the mean age of Pclass 1
PC2 = titatest[titatest['Pclass']==2]['Age'].mean()  #Finding the mean age of Pclass 2
PC3 = titatest[titatest['Pclass']==3]['Age'].mean()  #Finding the mean age of Pclass 3


# In[ ]:


titatest['Age'] = titatest[['Age','Pclass']].apply(mis,axis=1)


# In[ ]:


titatest.info()


# **Visualization of data**

# In[ ]:


plt.figure(figsize=(13,6))
sns.countplot('Survived',data=titatrain,hue='Sex').margins(x=0)
plt.legend( loc = 'upper right')


# We can see that the number of female survived is doubble than number of male survived. Whereas the number of male died is very high compared to female.

# In[ ]:


plt.figure(figsize=(13,6))
sns.countplot('Survived',hue='Pclass',data=titatrain).margins(x=0)


# I'm considering Class 3 to be cheapest (Ticket Price), Class 2 to be moderate (Ticket Price), and Class 3 to be costly (Ticket Price). From the above graph we can understand that the people from Class 3 have died more compared to other classes. Number of people survived is high in Class 1.

# In[ ]:


plt.figure(figsize=(13,6))
sns.countplot('SibSp',data=titatrain).margins(x=0)


# We can see that most of the people have come alone and may be the '1' might be the person with spouce and very few families

# In[ ]:


sns.distplot(titatrain['Age'],kde=False,bins=20).margins(x=0)


# In[ ]:


surages = titatrain[titatrain.Survived == 1]["Age"]
notsurages = titatrain[titatrain.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(surages, kde=False).margins(x=0)
plt.title('Survived')
plt.subplot(1, 2, 2)
sns.distplot(notsurages, kde=False)
plt.title('Not Survived')
plt.subplots_adjust(right=2)


# In[ ]:


sns.barplot(x="Sex", y="Survived", hue="Pclass", data=titatrain)


# We can find that the Female who survived are from Pclass 1 than from other Classes.

# In[ ]:


sns.pairplot(titatrain,hue = 'Survived')


# *** To convert the categorical features***  such as Sex, Embarked,Pclass and drop the First coloum in them to avoid Multicollinearity (If independent columns become dependent on each other then this happenes)

# In[ ]:


Sextrain = pd.get_dummies(titatrain['Sex'],drop_first=True)
Embarkedtrain = pd.get_dummies(titatrain['Embarked'],drop_first=True)
Pclasstrain = pd.get_dummies(titatrain['Pclass'],drop_first=True)

Sextest =pd.get_dummies(titatest['Sex'],drop_first=True)
Embarkedtest = pd.get_dummies(titatest['Embarked'],drop_first=True)
Pclasstest = pd.get_dummies(titatest['Pclass'],drop_first=True)


# In[ ]:


titatrain = pd.concat([titatrain,Sextrain,Embarkedtrain,Pclasstrain],axis=1)
titatest = pd.concat([titatest,Sextest,Embarkedtest,Pclasstest],axis=1)


# In[ ]:


titatrain.drop(['Embarked','Sex','Pclass','PassengerId','Cabin','Name','Ticket'],axis=1,inplace=True)


# In[ ]:


titatrain.head()


# In[ ]:


titatest.drop(['Pclass','Name','Sex','Ticket','Embarked','Cabin'],axis=1,inplace=True)


# In[ ]:


titatest.head()


# **Data Prediction**
# 
# Separate X and y in both Train and Test Data set

# In[ ]:


X_train = titatrain.drop('Survived',axis=1)  #We define the training label set
y_train = titatrain['Survived']   #We define the training label set
titatest.fillna('0',inplace=True)
X_test = titatest.drop('PassengerId',axis=1)     #We define the testing label set
#we don't have y_test, that is what we're trying to predict with our model


# 

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.3,random_state = 0)


# Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.fit_transform(X_test)
X_valid = Sc_X.fit_transform(X_valid)


# Importing Algorithms

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score #To evaluate the model performance
from sklearn.metrics import classification_report,confusion_matrix


# **Logistic Regression**

# In[ ]:


lomo_train = LogisticRegression()
lomo_train.fit(X_train,y_train)


# In[ ]:


lomo_predictions_train = lomo_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,lomo_predictions_train))
print(confusion_matrix(y_valid,lomo_predictions_train))


# In[ ]:


acc_lomo = accuracy_score(y_valid, lomo_predictions_train)
acc_lomo


# **KNN**

# In[ ]:


knn_train = KNeighborsClassifier(n_neighbors=1)
knn_train.fit(X_train,y_train)


# In[ ]:


knn_predictions_train = knn_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))


# In[ ]:


acc_knn1 = accuracy_score(y_valid, knn_predictions_train)
acc_knn1


# This above prediction is for K = 1, now lets find the value of K where the error is minimum 

# In[ ]:


error_rate = []
for i in range(1,50):
    
    knn_train = KNeighborsClassifier(n_neighbors=i)
    knn_train.fit(X_train,y_train)
    knn_predictions_train_i = knn_train.predict(X_valid)
    error_rate.append(np.mean(knn_predictions_train_i != y_valid))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Approximately at K = 22 the error rate is very low, now let's change the value of K to 22

# In[ ]:


knn_train = KNeighborsClassifier(n_neighbors=22)
knn_train.fit(X_train,y_train)
knn_predictions_train = knn_train.predict(X_valid)
print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))


# In[ ]:


acc_knn = accuracy_score(y_valid, knn_predictions_train)
acc_knn


# **Decision Tree**

# In[ ]:


dtree_train = DecisionTreeClassifier()
dtree_train.fit(X_train,y_train)


# In[ ]:


dtree_predictions_train = dtree_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,dtree_predictions_train))
print(confusion_matrix(y_valid,dtree_predictions_train))


# In[ ]:


acc_dtree = accuracy_score(y_valid, dtree_predictions_train)
acc_dtree


# **Random Forest****

# In[ ]:


rfc_train = RandomForestClassifier(n_estimators=100)
rfc_train.fit(X_train, y_train)


# In[ ]:


rfc_predictions_train = rfc_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,rfc_predictions_train))
print(confusion_matrix(y_valid,rfc_predictions_train))


# In[ ]:


acc_rfc = accuracy_score(y_valid, dtree_predictions_train)
acc_rfc


# **SVM(Suport Vector Machine)** SVC - SUPORT VECTOR CLASIFIER

# In[ ]:


svm_train = SVC()
svm_train.fit(X_train, y_train)


# In[ ]:


svm_predictions_train = svm_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,svm_predictions_train))
print(confusion_matrix(y_valid,svm_predictions_train))


# In[ ]:


acc_svm = accuracy_score(y_valid, svm_predictions_train)
acc_svm


# **SVM using GRIDSEARCH**
# 
# 
# Finding the right parameters (like what C or gamma values to use) is a tricky task.The idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch. The CV stands for cross-validation.
# 
# GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values a[](http://)re the settings to be tested.

# In[ ]:


param_grid = {'C': [0.01,0.1,1, 10, 100, 1000,10000], 'gamma': [10,1,0.1,0.01,0.001,0.0001,0.00001], 'kernel': ['rbf']} 


# One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, which behaves exactly the same.We should add (refit=True) and choose verbose to whatever number we want. Higher the number means there will be more verbose (where verbose means the text output describing the whole process).

# In[ ]:


grid_train = GridSearchCV(SVC(),param_grid,refit=True,verbose=10)


# In[ ]:


grid_train.fit(X_train,y_train) 


# The code runs the same loop with cross-validation, to find the best (parameter) combination, once the best combo is found then it runs fit again on all data passed to fit, all this is done to build a single new model using the best combination.

# In[ ]:


grid_train.best_params_   #to inspect the best parameters found by GridSearchCV


# In[ ]:


grid_train.best_estimator_       #to inspect the best estimator found by GridSearchCV


# In[ ]:


grid_predictions_train = grid_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,grid_predictions_train))
print(confusion_matrix(y_valid,grid_predictions_train))


# In[ ]:


acc_grid_svm = accuracy_score(y_valid, grid_predictions_train)
acc_grid_svm


# **Linear SVM**

# In[ ]:


linsvm_train = LinearSVC()
linsvm_train.fit(X_train, y_train)


# In[ ]:


linsvm_predictions_train = linsvm_train.predict(X_valid)


# In[ ]:


print(classification_report(y_valid,linsvm_predictions_train))
print(confusion_matrix(y_valid,linsvm_predictions_train))


# In[ ]:


acc_linsvm = accuracy_score(y_valid, svm_predictions_train)
acc_linsvm


# **XGB Classifier**

# In[ ]:


xg_train = XGBClassifier()
xg_train.fit(X_train,y_train)
xg_predictions_train = xg_train.predict(X_valid)
print(classification_report(y_valid,knn_predictions_train))
print(confusion_matrix(y_valid,knn_predictions_train))


# In[ ]:


acc_xg = accuracy_score(y_valid, xg_predictions_train)
acc_xg


# **CHECKING THE MODEL PERFORMANCE**

# In[ ]:


model_performance = pd.DataFrame({
    "Model": ["SVC", "Linear SVC", "Random Forest", 
              "Logistic Regression", "K Nearest Neighbors",  
              "Decision Tree",'XGB','SVC Grid'],
    "Accuracy": [acc_svm, acc_linsvm, acc_rfc, 
              acc_lomo, acc_knn, acc_dtree,acc_xg,acc_grid_svm]
})

model_performance.sort_values(by="Accuracy", ascending=False)


# So XGB has high accuracy than other models. Now let's submit

# In[ ]:


xg_train.fit(X_train,y_train)


# In[ ]:


submission_of_predictions = xg_train.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": titatest["PassengerId"],
        "Survived": submission_of_predictions
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)

