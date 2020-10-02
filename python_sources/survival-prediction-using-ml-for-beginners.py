#!/usr/bin/env python
# coding: utf-8

# To all Fellow learners, Good Luck.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# loading the training dataset
train_dataset = pd.read_csv('../input/titanic/train.csv')


# In[ ]:


train_dataset.head()


# Inferences: 
# 1. PassengerId is the Identity column and Survived is the target column. 
# 2. The columns Name and Ticket can be removed

# In[ ]:


train_dataset.info()


# The column - Cabin contains only 204 rows. So the column can be removed.

# In[ ]:


# creating a copy of training dataset to work with
work_dataset = train_dataset.copy()


# In[ ]:


work_dataset = work_dataset.set_index('PassengerId')


# In[ ]:


work_dataset = work_dataset.drop(['Name', 'Ticket', 'Cabin'], axis = 1)


# In[ ]:


work_dataset.head()


# We need to convert Sex and Embarked into binary values. There are two missing values in Embarked. They can be dropped or replaced by mean.

# In[ ]:


# we can use label encoder for conversion of numeric data
from sklearn.preprocessing import LabelEncoder
def encoding(feature):
    if (feature.dtype == 'object'):
        return LabelEncoder().fit_transform(feature)
    else:
        return feature


# In[ ]:


# dropping missing values
work_dataset = work_dataset.dropna()


# In[ ]:


work_dataset = work_dataset.apply(encoding)


# In[ ]:


work_dataset.head()


# We can bin the Age and Fare. First we will apply this dataset to classification algorithm and check the accuracy. Then we shall bin and see the accuracy again.

# In[ ]:


import seaborn as sns
sns.pairplot(work_dataset)

There are some outlieres in the feature Fare. The skewness is positively skewed.
# In[ ]:


work_dataset = work_dataset[work_dataset['Fare'] < 200]
sns.distplot(work_dataset['Fare'])


# In[ ]:


work_dataset.shape


# In[ ]:


# splitting the training and testing dataset.
from sklearn.model_selection import train_test_split
X = work_dataset.drop(['Survived'], axis = 1)
y = work_dataset[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)


# In[ ]:


# Importing Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


print(confusion_matrix(predict, y_test))
print(classification_report(predict, y_test))
accuracy_before = accuracy_score(predict, y_test)*100
print(accuracy_before)


# Let us try binning the Age and Fare features and see the accuracy

# In[ ]:


work_dataset.loc[:,['Age', 'Fare']].describe()


# In[ ]:


sns.distplot(work_dataset.loc[:,['Age']])


# Age is normally distributed (approximately).

# In[ ]:


sns.distplot(work_dataset.loc[:,['Fare']])


# In[ ]:


work_dataset['Fare'] = pd.cut(x = work_dataset.Fare, bins = 17, labels = range(17))
work_dataset['Age'] = pd.cut(x = work_dataset.Age, bins = 8, labels = range(8))


# In[ ]:


sns.distplot(work_dataset.Age, kde = False)


# In[ ]:


sns.distplot(work_dataset.Fare, kde = False)


# Now we shall apply the algorithm and check the accuracy

# In[ ]:


X = work_dataset.drop(['Survived'], axis = 1)
y = work_dataset[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 22)


# In[ ]:


classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
predict = classifier.predict(X_test)


# In[ ]:


print(confusion_matrix(predict, y_test))
print(classification_report(predict, y_test))
accuracy_after = accuracy_score(predict, y_test)*100
print(accuracy_after)


# In[ ]:


print(accuracy_before)
print(accuracy_after)


# The accuracy after binning is higher than accuracy before binning. Now we can apply to the unknown dataset and predict the results

# Let us try with different classification algorithms:

# In[ ]:


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
c = DecisionTreeClassifier()
c.fit(X_train, y_train)
predict = c.predict(X_test)
accuracy = accuracy_score(predict, y_test)*100
print(accuracy)


# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
c_knn = KNeighborsClassifier(n_neighbors = 11)
c_knn.fit(X_train, y_train)
predict = c_knn.predict(X_test)
accuracy = accuracy_score(predict, y_test)*100
print(accuracy)


# In[ ]:


# MLP
from sklearn.neural_network import MLPClassifier
c = MLPClassifier(hidden_layer_sizes = (100,))
c.fit(X_train, y_train)
predict = c.predict(X_test)
accuracy = accuracy_score(predict, y_test)*100
print(accuracy)


# In[ ]:


# SVC
from sklearn.svm import SVC
c = SVC()
c.fit(X_train, y_train)
predict = c.predict(X_test)
accuracy = accuracy_score(predict, y_test)*100
print(accuracy)


# KNN is the best classifier for the dataset

# In[ ]:


# importing the test dataset
test_dataset = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_dataset.head()


# In[ ]:


work_test_dataset = test_dataset.copy()
# making the Id as index or we can drop this ID
work_test_dataset = work_test_dataset.set_index('PassengerId')
# removing the columns that we removed from the training dataset
work_test_dataset = work_test_dataset.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
# converting the non numeric into numeric values
work_test_dataset = work_test_dataset.apply(encoding)


# In[ ]:


work_test_dataset.head()


# In[ ]:


# checking for missing values
work_test_dataset.isnull().sum()


# We have to impute the missing values. We can use simple imputer to impute the missing values

# In[ ]:


# importing simple imputer
from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy = 'mean')
final_test_dataset = pd.DataFrame(impute.fit_transform(work_test_dataset), columns = work_test_dataset.columns)


# In[ ]:


final_test_dataset.isnull().sum()


# Now predicting the values with the model

# In[ ]:


# applying the trained KNN classifier
survived = c_knn.predict(final_test_dataset)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test_dataset.PassengerId, 'Survived':survived})


# Generating the CSV file

# In[ ]:


submission.to_csv('submission.csv', index = False)


# Learning never ends for everyone. Your suggestions and other creative ideas are much appreciated. 
