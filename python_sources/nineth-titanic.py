#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
submission = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


submission.head()


# In[ ]:


train.shape


# In[ ]:


submission.shape


# In[ ]:


train.info()


# 
# The features Age, Cabin and Embarked have some null values.

# In[ ]:


train.describe()


# In[ ]:


#How many atributes does each feature has?
def unique_values(df):
    print("Unique atributes in each feature:")
    for feature in df.columns:
        print(f"{feature} --> {df[feature].nunique()}")


# In[ ]:


unique_values(train)


# PassengerId and the Name are values that identify the person, but our goal is to generalize the information which is what the classification model ask for.

# ### Dropping the features that are too specific to identify the pessenger.

# In[ ]:


#List of columns that I'm going to remove
columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train = train.drop(columns, axis = 1)
train.head()


# ## Exploratory Data Analysis - EDA

# The main goal here is to find which features have more correlation with the target = Survived/ Not Survived

# In[ ]:


train['Survived'].value_counts()


# 
# 0 --> Didn't Survived; 1 --> Survived

# In[ ]:


#Counplot chart
plt.subplots(figsize = (10,7))
plot = sns.countplot(x = 'Survived', data = train, palette = "BuGn")
plot.set_title("Titanic Survivers", fontsize = 15)
plot.set_xlabel(" ", fontsize = 15)
plot.set_ylabel("Number of people", fontsize = 15)
plot.set_xticklabels(['Not Survived', 'Survived'])


# 
# Unfortunately the majority of people didn't survived.

# In[ ]:


# Percentage of people that Survived
total = train['Survived'].value_counts().sum()
survived = train['Survived'].value_counts()[0]
not_survived = train['Survived'].value_counts()[1]
per_survived = survived/total*100
per_not_survived = not_survived/total*100
print(f"The percentage of passengers that survived are: {per_survived:.2f}% and the passengers that didn't survived are:{per_not_survived:.2f}%")


# ### Pclass - Ticket class

# In[ ]:


total_pclass = train['Pclass'].shape[0]
total_pclass


# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


plt.subplots(figsize = (10,7))
plot = sns.countplot(x = 'Pclass', data = train, palette = "PuBuGn")
plot.set_title("Titanic Ticket Classes", fontsize = 15)
plot.set_xlabel(" ", fontsize = 15)
plot.set_ylabel("Number of people", fontsize = 15)
plot.set_xticklabels(['First', 'Second', 'Third'])
for p in plot.patches:
    height = p.get_height()
    plot.text(p.get_x()+ p.get_width()/2., height + 5,'{:.1f}%'.format(100.*height/total_pclass), ha = 'center')


# The third class was the one with more passengers

# #### Pclass - Ticket class X SURVIVED

# In[ ]:


plt.subplots(figsize = (10,7))
plot = sns.countplot(x = 'Survived', data = train, palette = "PuBuGn", hue = 'Pclass')
plot.set_title("Titanic Survivers per Ticket Classes", fontsize = 15)
plot.set_xlabel(" ", fontsize = 15)
plot.set_ylabel("Number of people", fontsize = 15)
plot.set_xticklabels(['Not Survived', 'Survived'])
plt.legend(['First', 'Second', 'Third'])


# Most of the people who died was from the third class.....

# In[ ]:


#Percentage of survivors by group of classes
train[['Pclass', 'Survived']].groupby(['Pclass']).mean()*100


# ## Gender

# In[ ]:


total_gen = train['Sex'].shape[0]
total_gen


# In[ ]:


plt.subplots(figsize = (10,7))
plot = sns.countplot(x = 'Sex', data = train, palette = "PuBuGn")
plot.set_title("Titanic people's gender", fontsize = 15)
plot.set_xlabel(" ", fontsize = 15)
plot.set_ylabel("Number of people", fontsize = 15)
#plot.set_xticklabels(['First', 'Second', 'Third'])
for p in plot.patches:
    height = p.get_height()
    plot.text(p.get_x()+ p.get_width()/2., height + 5,'{:.1f}%'.format(100.*height/total_gen), ha = 'center')


# ### Survivors by gender

# In[ ]:


plt.subplots(figsize = (10,7))
plot = sns.countplot(x = 'Survived', data = train, palette = "PuBuGn", hue = 'Sex')
plot.set_title("Titanic Survivers by Gender", fontsize = 15)
plot.set_xlabel(" ", fontsize = 15)
plot.set_ylabel("Number of people", fontsize = 15)
plot.set_xticklabels(['Not Survived', 'Survived'])


# In[ ]:


#Percentage of survivors by gender
train[['Sex', 'Survived']].groupby(['Sex']).mean()*100


# ## Age

# In[ ]:


train['Age'].describe()


# In[ ]:


#BOXPLOT
plt.subplots(figsize = (5,5))
plot = sns.boxplot(y ='Age', data = train, color = 'orange').set_title('Age')


# In[ ]:


#Null values
train['Age'].isnull().sum()


# In[ ]:


#Filter the values that are not null
train[train['Age'].isnull() == False]


# In[ ]:


plot = sns.distplot(train[train['Age'].isnull() == False]['Age'])


# ## Fare

# In[ ]:


train['Fare'].value_counts()


# ## Feature Engineering

# ### Age substitution

# In[ ]:


median_age = train['Age'].median()
median_age


# In[ ]:


train[train['Sex'] == "female"]['Age'].isnull()


# In[ ]:


#Woman median age
mean_woman_age = train[train['Sex'] == "female"]['Age'].mean()
mean_woman_age


# In[ ]:


#Woman age substitution
train.loc[(train['Sex'] == 'female') & (train['Age'].isnull()),'Age'] = mean_woman_age


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


#Man median age
mean_man_age = train[train['Sex'] == "male"]['Age'].mean()
mean_man_age


# In[ ]:


#Man age substitution
train.loc[(train['Sex'] == 'male') & (train['Age'].isnull()),'Age'] = mean_man_age


# In[ ]:


train['Age'].isnull().sum()


# In[ ]:


#Distplot with the new values
plot = sns.distplot(train.loc[(train['Survived'] == 1) & (train['Age'].isnull() == False),'Age'], color = 'lightcoral', hist = True, label = "Survived")
plot = sns.distplot(train.loc[(train['Survived'] == 0) & (train['Age'].isnull() == False),'Age'], color = 'c', hist =  False, label = "Not Survived")
plot.set_title("Age distribution and survivors", fontsize = 10)


# #### Gender substitution

# In[ ]:


train['Sex'].value_counts()


# In[ ]:


train['Sex'].replace(['male', 'female'], [0,1], inplace = True)


# In[ ]:


train['Sex'].value_counts()


# ### Correlation matrix

# In[ ]:


train.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap("coolwarm"), axis = 1)


# It seems that the 'Sex' feature has high correlation with the 'Survived' target

# ## Training

# #### Separate the the features from the target

# In[ ]:


#I'm going to use just the features that I really looked for
#X --> features used for trainig
x = train[['Age', 'Sex', 'Pclass']]
# Y --> target
y = train['Survived']


# In[ ]:


x.head()


# #### Separate the training data from the test data

# In[ ]:


from sklearn.model_selection import train_test_split
# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# #### Decision tree model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
# Classifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = 5)
# Model training
model.fit(x_train, y_train)


# ### Evaluation

# In[ ]:


y_pred = model.predict(x_test)
y_pred


# #### Accuracy

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# It means that the model make 76% of the predictions correctly

# In[ ]:


#### Confusion matrix


# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools
#code from the scikit-learn website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #plt.ylim(0.5, 0.5)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(1.5, -0.5) 

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Real Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()


# In[ ]:


#Building the matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix


# In[ ]:


plot_confusion_matrix(cnf_matrix, classes = ['Not Survived', 'Survived'])


# ### Prediction

# In[ ]:


submission.head()


# In[ ]:


submission.isnull().sum()


# In[ ]:


#Woman age substitution
submission.loc[(submission['Sex'] == 'female') & (submission['Age'].isnull()),'Age'] = mean_woman_age


# In[ ]:


submission['Age'].isnull().sum()


# In[ ]:


#Man age substitution
submission.loc[(submission['Sex'] == 'male') & (submission['Age'].isnull()),'Age'] = mean_man_age


# In[ ]:


submission['Age'].isnull().sum()


# In[ ]:


submission['Sex'].value_counts()


# In[ ]:


submission['Sex'].replace(['male', 'female'], [0,1], inplace = True)


# In[ ]:


x = submission[['Age', 'Sex', 'Pclass']]
x.head()


# In[ ]:


x['Sex'].value_counts()


# In[ ]:


#prediction
predict = model.predict(x) 
predict


# ### Preparing for the Kaggle submission

# In[ ]:


submission['Survived'] = predict


# In[ ]:


submission


# In[ ]:


sub_id = submission[['PassengerId', 'Survived' ]]
sub_id


# In[ ]:


sub_id.to_csv('titanic_submission.csv', index = False)


# In[ ]:




