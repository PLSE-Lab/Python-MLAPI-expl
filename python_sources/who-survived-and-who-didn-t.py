#!/usr/bin/env python
# coding: utf-8

# ### Kaggle Competition | Titanic Machine Learning from Disaster
# 
# >The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.  This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# >One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.  Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# >In this contest, we ask you to complete the analysis of what sorts of people were likely to survive.  In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# >This Kaggle Getting Started Competition provides an ideal starting place for people who may not have a lot of experience in data science and machine learning."
# 
# From the competition [homepage](http://www.kaggle.com/c/titanic-gettingStarted).
# 
# 
# ### Goal for this Notebook:
# Show a simple example of an analysis of the Titanic disaster in Python. This is aimed for those looking to get into the field or those who are already in the field and looking to see an example of an analysis done with Python.
# 
# #### This Notebook will show basic examples of: 
# #### Data Handling
# *   Importing Data with Pandas
# *   Cleaning Data
# *   Exploring Data through Visualizations with Matplotlib and seaborn
# 
# #### Data Analysis
# *    Supervised Machine learning Techniques:
#     +   K Nearest Neighbour 
#     +   Gaussian Naive Bayes
#     +   Support Vector Machine (SVM)
#     +   Decision Tree Classifier
#     +   Plotting results

# ## Importing the required Libraries

# In[ ]:


# Importing the required libraries
import pandas as pd
import numpy as np
import scipy as sc
import sys
import IPython
from IPython import display 
import sklearn 
import random
import time


# In[ ]:


#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# In[ ]:


#Configure Visualization Defaults
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[ ]:


# Importing the model. 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


# ### Describing and analysing the given data sets for better understanding of the data provided

# In[ ]:


data_raw = pd.read_csv("../input/titanic/train.csv")
data_test = pd.read_csv("../input/titanic/test.csv")

data_train = data_raw.copy(deep = True)
# combining the two datasets for cleaning
data_to_be_cleaned = [data_train, data_test]


# In[ ]:


data_raw.info()
print('*-'*25)
data_test.info()


# In[ ]:


print('Train columns with null values:\n', data_train.isnull().sum())
print("*-"*25)

print('Test/Validation columns with null values:\n', data_test.isnull().sum())
print("*-"*25)


# In[ ]:


data_raw.describe(include = 'all')


# ## Cleaning the dataset

# In[ ]:


###COMPLETING: complete or delete missing values in train and test/validation dataset
for dataset in data_to_be_cleaned:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)


# In[ ]:


data_to_be_cleaned


# In[ ]:


#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data_train.drop(drop_column, axis=1, inplace = True)

print(data_train.isnull().sum())
print("-"*50)
print(data_test.isnull().sum())


# In[ ]:


###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_to_be_cleaned:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    #Continuous variable bins
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #Age Bins/Buckets using cut or value bins
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


# In[ ]:


data_to_be_cleaned[0].info()


# In[ ]:


data_to_be_cleaned[1].info()


# In[ ]:


#cleanup rare title names
print(data_train['Title'].value_counts())
print('*-*'*10)
stat_min = 10 
title_names = (data_train['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code
data_train['Title'] = data_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data_train['Title'].value_counts())
print("-"*50)


# In[ ]:


#preview data again
data_train.info()
data_test.info()
data_train.head(10)


# In[ ]:


data_test.shape


# In[ ]:


data_train.shape


# In[ ]:


#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

#code categorical data
label = LabelEncoder()
for dataset in data_to_be_cleaned:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


# In[ ]:


data_test


# In[ ]:


#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
original_train_data = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
train_data = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','IsAlone', 'FamilySize', 'AgeBin_Code', 'FareBin_Code'] #coded for algorithm calculation
original_dataset =  Target + original_train_data
print('Original X Y: ', original_dataset, '\n')
print('Training X Y: ', train_data, '\n')


# #### Double check Cleaned Data

# In[ ]:


print('Train columns with null values: \n', data_train.isnull().sum())
print("-"*10)
print (data_train.info())
print("-"*10)

print('Test/Validation columns with null values: \n', data_test.isnull().sum())
print("-"*10)
print (data_test.info())
print("-"*10)


# In[ ]:


X = data_train[train_data]


# In[ ]:


Y = data_train[Target]


# ## Split Training and Testing Data

# In[ ]:


#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_train[train_data], data_train[Target],test_size= 0.2, random_state = 0)

print("Data given for training - Shape: {}".format(data_train.shape))
print("Train split of the given training data - Shape: {}".format(x_train.shape))
print("Test split of the tgiven training data - Shape: {}".format(x_test.shape))

original_refined_dataset = data_train[original_train_data]

x_train.head()


# In[ ]:


original_refined_dataset


# ## Perform Exploratory Analysis with Statistics

# In[ ]:


#Discrete Variable Correlation by Survival using
#group by aka pivot table
for x in original_refined_dataset:
    if data_train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data_train[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        

#using crosstabs
print(pd.crosstab(data_train['Title'],data_train[Target[0]]))


# In[ ]:



plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data_train['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data_train['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data_train['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data_train[data_train['Survived']==1]['Fare'], data_train[data_train['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data_train[data_train['Survived']==1]['Age'], data_train[data_train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data_train[data_train['Survived']==1]['FamilySize'], data_train[data_train['Survived']==0]['FamilySize']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


#graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data_train, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data_train, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data_train, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data_train, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data_train, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data_train, ax = saxis[1,2])


# In[ ]:


#graph distribution of qualitative data: Pclass
#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data_train, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data_train, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data_train, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')


# In[ ]:


#graph distribution of qualitative data: Sex
#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1,3,figsize=(14,12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data_train, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data_train, ax  = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data_train, ax  = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')


# In[ ]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Correlation of Features', y=1.05, size=15)

correlation_heatmap(data_train)


# ## Model Data

# In[ ]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
# Importing the model. 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# #### KNN Model

# In[ ]:


knn = KNeighborsClassifier(metric='minkowski', p=2) 
## doing 10 fold staratified-shuffle-split cross validation 
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)


# In[ ]:


## Search for an optimal value of k for KNN.
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train , y_train , cv = cv, scoring = 'accuracy')
    k_scores.append(scores.mean())
print("Accuracy scores are: {}\n".format(k_scores))
print ("Mean accuracy score: {}".format(np.mean(k_scores)))
plt.plot(k_range, k_scores)


# In[ ]:


classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


knn_accy = accuracy_score(y_pred, y_test)
print(knn_accy)


# #### Gaussian Naive Bayes Model

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X,Y)

y_pred = gaussian.predict(x_test)
gaussian_accy = accuracy_score(y_pred, y_test)

print(gaussian_accy)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### SVM Classifier Model

# In[ ]:


svclassifier = SVC(kernel='rbf')

svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)

svclassifier_accy = accuracy_score(y_pred, y_test)
print(svclassifier_accy)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### Decision Tree Model

# In[ ]:


decision_tree_classifier =DecisionTreeClassifier()

decision_tree_classifier.fit(x_train, y_train)
y_pred = decision_tree_classifier.predict(x_test)

decision_tree_classifier_accy = accuracy_score(y_pred, y_test)
print(decision_tree_classifier_accy)


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **The maximum accuracy that was obtained is 83.8% of Decision Tree model. This model will be used to predict the values of the validation test data.**

# In[ ]:


validation_data = data_test


# In[ ]:


validation_data


# In[ ]:


X_val = validation_data[train_data]
val_prediction = decision_tree_classifier.predict(X_val)
passengerid = validation_data["PassengerId"]


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": val_prediction
    })


# In[ ]:


submission


# In[ ]:


submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)


# **This is my first Kaggle Notebook. I hope you all liked it. I would also like to improve my score and make this notebook even better. Any kind of help will be much appreciated**

# If you liked the contents of this notebook, an upvote would be much appreciated. 
