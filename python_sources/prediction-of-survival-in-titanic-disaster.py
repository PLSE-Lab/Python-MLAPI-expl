#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # setting seaborn default for plots

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# ## About Data

# In[ ]:


print("train data shape : ", train_data.shape)
print("test data shape : ", test_data.shape)
train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# ### Fields Description:
# 
# Survived -> 1:Yes, 0:No      
# Pclass(Passenger Class) -> 1:1st class, 2:2nd class, 3:3rd class     
# SibSp(Siblings/Spouse no) -> in range of 0-5    
# Parch (Parents/Children no) -> in range of 0-6    
# Ticket (ticket no)
# Cabin(Cabin no)      
# Embarked(Port of Embarkation) -> C:Cherbourg, Q:Queenstown, S:Southampton      
# 
# ### Train data missing values : 
# Age : 177      
# Cabin : 687      
# Embarked : 2    
# 
# 
# ### Test data missing values : 
# Age : 86     
# Cabin : 327     
# Fare : 1     
# 

# ## Data Exploration and Visualization

# ### Bar Chart for Categorical Features
# * Sex      
# * Pclass        
# * SibSp     
# * Parch     
# * Embarked      
# * Cabin      
# 
# 

# In[ ]:


def bar_chart(feature):
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# It shows that females are more likely to survive than males

# In[ ]:


bar_chart('Pclass')


# Confirms that first class passengers are more likely to survive

# In[ ]:


bar_chart('SibSp')


# Shows that a person with no sibling/ spouse is more likely to survive

# In[ ]:


bar_chart('Parch')


# Similar observation as that of SibSp. People having no parent or child is more likely to survive.

# In[ ]:


bar_chart('Embarked')


# People embarked from S is more likely to survive, followed by C.

# ## Feature Engineering
# 
# ### 1. Name

# In[ ]:


# Combining train and test data to do feature engineering

train_test = [train_data, test_data]

for data in train_test:
    data['Title'] = data['Name'].str.split(', ').str[1].str.split('.').str[0]


# In[ ]:


train_data['Title'].value_counts()


# In[ ]:


# We mainly want the first three titles for our analysis as they are in majority

for dataset in train_test:
    top_3 = [x for x in dataset.Title.value_counts().sort_values(ascending=False).head(3).index]
    for label in top_3:
        dataset[label] = np.where(dataset['Title']==label,1,0)


# In[ ]:


train_data.head()


# In[ ]:


# Function to delete unnecessary feature from dataset

def drop_columns(df, col):
    df.drop(col, axis=1, inplace=True)


# In[ ]:


for dataset in train_test:
    drop_columns(dataset, 'Name')
    drop_columns(dataset, 'Title')


# In[ ]:


train_data.head()


# ### 2. Sex

# In[ ]:


#Converting and concatenating sex to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Sex'], prefix='gender')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Sex'], prefix='gender')],axis=1)


# In[ ]:


train_test = [train_data, test_data]
for dataset in train_test:
    drop_columns(dataset, 'Sex')
    drop_columns(dataset, 'gender_male')


# In[ ]:


test_data.head()


# ### 3. Age

# In[ ]:


# Replacing missing values with the median age grouped by title : 177, 86

train_data['Age'].fillna(train_data.groupby("Mr")["Age"].transform("median"), inplace=True)
train_data['Age'].fillna(train_data.groupby("Mrs")["Age"].transform("median"), inplace=True)
train_data['Age'].fillna(train_data.groupby("Miss")["Age"].transform("median"), inplace=True)

test_data['Age'].fillna(test_data.groupby("Mr")["Age"].transform("median"), inplace=True)
test_data['Age'].fillna(test_data.groupby("Mrs")["Age"].transform("median"), inplace=True)
test_data['Age'].fillna(test_data.groupby("Miss")["Age"].transform("median"), inplace=True)


# ### 3. Embarked

# In[ ]:


Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


# Based on above observation , we can conveniently replace the missing values of embarked with S.

train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[ ]:


#Converting and concatenating embarked to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Embarked'], prefix='em')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Embarked'], prefix='em')],axis=1)

drop_columns(train_data, 'em_Q')
drop_columns(test_data, 'em_Q')
drop_columns(train_data, 'Embarked')
drop_columns(test_data, 'Embarked')


# ### 4. Fare

# In[ ]:


test_data.head()


# In[ ]:


# replacing missing Fare with median fare for each Pclass
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# ### 5. Cabin

# In[ ]:


train_data.Cabin.value_counts()


# In[ ]:


#Getting the first alphabet of each cabin
train_test = [train_data, test_data]
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[ ]:


Pclass1 = train_data[train_data['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


train_data['Cabin'].value_counts()


# In[ ]:


#Dropping cabin. Need to decide how to replace NaN values

drop_columns(train_data, 'Cabin')
drop_columns(test_data, 'Cabin')


# ### 6. Pclass

# In[ ]:


#Converting and concatenating Pclass to binary using one hot encoding

train_data = pd.concat([train_data, pd.get_dummies(train_data['Pclass'], prefix='class')],axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data['Pclass'], prefix='class')],axis=1)


# In[ ]:


drop_columns(train_data, 'Pclass')
drop_columns(test_data, 'Pclass')


# In[ ]:


train_data.head()


# ### 7. Family size

# In[ ]:


#Adding all the parents, children, spouse and siblings to count the no of members in the family on board

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1


# In[ ]:


#Dropping the unnecessary features

features_drop = ['Ticket', 'SibSp', 'Parch']
train_data = train_data.drop(features_drop, axis=1)
test_data = test_data.drop(features_drop, axis=1)
train_data = train_data.drop('PassengerId', axis=1)


# In[ ]:


train_data.head()


# In[ ]:


#Checking corelation matrix

train_data.corr()


# In[ ]:


#Segregating features and label

y = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)


# In[ ]:


train_data.head()


# ## Modelling

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt



# In[ ]:


#Baseline Model : Probability of not surviving which is the majority class

Survival_prob = (y==0).sum() / len(train_data)
Survival_prob


# ### 1. k-fold Cross Validation

# In[ ]:


k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### 2. Naive Bayes

# In[ ]:


model = GaussianNB()
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ### 3. Decision Trees

# In[ ]:


model = tree.DecisionTreeClassifier(random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ### 4. Random Forest

# In[ ]:


model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ### 5. kNN

# In[ ]:


model = KNeighborsClassifier(n_neighbors = 13)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ### 6. SVM

# In[ ]:


model = svm.SVC(kernel='linear', random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ### 7. Logistic Regression

# In[ ]:


model = LogisticRegression(random_state=0)
score = cross_val_score(model, train_data, y, cv=k_fold, n_jobs=1, scoring='accuracy')
print(score)
round(np.mean(score)*100, 2)


# ## Testing

# In[ ]:


print(train_data.shape)
print(test.shape)
print(test_data.shape)


# In[ ]:


model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0)
model.fit(train_data, y)

test = test_data.drop("PassengerId", axis=1).copy()

prediction = model.predict(test)


# In[ ]:


print(test)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:


#Rough

conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)
train_accuracy = accuracy_score(y.to_numpy(), y_pred)
print(train_accuracy)

precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
print(precision)
print(recall)


#average_precision = average_precision_score(y, y_score)
#disp = plot_precision_recall_curve(model, X, y)
#disp.ax_.set_title('Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))


