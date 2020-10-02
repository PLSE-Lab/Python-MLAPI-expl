#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # **Introduction**
# 
# This particular notebook stands for Titanic  dataset , as an introductory analysis for data scientist.  If anyone understand this Notebook properly then it will be helpful for future projects also.
# There are several excellent notebooks to study data science competition entries. However many will skip some of the explanation on how the solution is developed as these notebooks are developed by experts for experts. The objective of this notebook is to follow a step-by-step workflow, explaining each step and rationale for every decision we take during solution development.
# 

# Lets import some necessary libraries

# In[1]:


# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import seaborn as sns
pd.options.display.max_rows = 100


# # **Feature Exploration, Engineering and Cleaning**
# 
# A lot of help is taken from [ here.](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier/notebook)
# 
# Now we will proceed much like how most kernels in general are structured, and that is to first explore the data on hand, identify possible feature engineering opportunities as well as numerically encode any categorical features .

# ## **Loading data**
# 
# First let us load the data from the train.csv and test.csv into two dataframes. Then lets check the shape of dataset.

# In[2]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine_data =[ train, test] # To process both set together
print(train.shape)
print(test.shape)


# As you can see there are 891 rows and 12 columns in the train dataframe.The test dataframe has 418 rows and eleven columns

# ## **Overview**
# 
# To get an overview of dataset we normally use head command

# In[3]:


train.head(2) #Default is 5 row. But you can define as your wish by yourself in bracket


# Lets look at the number of columns and their names to get an overview of features

# In[4]:


print(train.columns.values)


# # **Information**
# 
# What are the data types for various features?Which features are mixed data types?Which features may contain errors or typos? Which features contain blank, null or empty values? 
# These are the questions may arrive. Lets look at the following code. We will get an overall idea about the answers of above questions.

# In[5]:


train.info()
print('*'*40)
test.info()


# Following key notes can be taken from above result
# *Seven features are integer or floats. Six in case of test dataset.
# * Five features are strings (object).
# * Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
# * Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# * Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.
# * Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# * Cabin > Age are incomplete in case of test dataset.

# This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting and what will require correcting.
# 

# In[6]:


train.describe(include='all')


# # **Pclass**: Passenger Class
#  
#  There is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.  Let us look at the values and the counts of Pclass. As you can see above the Pclass is catogoerical variables which takes values 1,2 and 3. You can see that most number of passengers are in class 3.
# Let us look at the fraction of people who survived in each class graphically. You can see almost 63% of 1 class survived.47% of 2nd class and 24 % of the third class survived. 
# We will use the same way for describing the other features related to survival rate.

# In[7]:


train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False).plot.bar()


# #  **SibSp**
#  
#  There is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.  Let us look at the values and the counts of SibSp. As you can see above the SibSp is catogoerical variables which takes values 1,2,3 and 4. You can see that most number of passengers are in class 1.
#  
# Lets look at the impact related to survival rate.

# In[8]:


train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False).plot.bar()


# #  **Parch**: Parents & Children
#  
#  There is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.  Let us look at the values and the counts of Parch. As you can see above the Parch is catogoerical variables. You can see that most number of passengers are in class 3.
#  
# Lets look at the impact related to survival rate.

# In[9]:


train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False).plot.bar()


# # **FamilySize from** **SibSp and Parch**
# 
# With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size. Create new feature combining existing features. This will enable us to drop Parch and SibSp from our datasets.

# In[10]:


for dataset in combine_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().plot.bar()


# # **IsAlone**
# 
# It seems has a good effect on our prediction but let's go further and categorize people to check whether they are alone in this ship or not. From the plot its clear that when person is alone has more chance to survive which is understandable.

# In[11]:


for dataset in combine_data:
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['IsAlone'] = dataset['IsAlone'].astype(int)
train[['IsAlone', 'Survived']].groupby(['IsAlone']).mean().plot.bar()


# # **Sex**
# 
#  There is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.  Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=0 and male=1. As you can see above the Sex is categoerical variables which takes values 0 and 1. 
# Let us look at the fraction of people who survived in each class graphically. You can see that most number of passengers are female.
# We will use the same way for describing the other features related to survival rate.
# 

# In[12]:


for dataset in combine_data:
    dataset['Sex'] = dataset['Sex'].map({"male": 1, "female": 0,})
    dataset['Sex'] = dataset['Sex'].astype(int)
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False).plot.bar() 


# # **Title**
# 
# Lets  try to look at the title of different persons. Most of them can be categorised  and those titles are giving sparse result , lets grope then in "rare" catagory.

# In[13]:


for dataset in combine_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()    


# Lets check the title impact on survival rate. Its clear that **Mrs and Miss** title has most survival rate respectively.

# In[14]:


for dataset in combine_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()


# Now it is necessary to mape the titles into values to determine the survival rate as we did for Sex feature.

# In[15]:


title_mapping = {"Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Rare": 5}
for dataset in combine_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
train[['Title', 'Survived']].groupby(['Title']).mean().plot.bar()    


# # **Age**
# 
# We have seen above that the Age variable is missing 177 values. This is a large number ( ~ 13% of the dataset). Simply replacing them with the mean or the median age might not be the best solution since the age may differ by groups and categories of passengers. Let's create a function that fills in the missing age in combined based on these different attributes. 
# There are different way to do this. Below I have mentioned three ways to do it. Among them last method define values with respect to "sex", "Pclass", "Title". Which will minimize the error in replacing actual values

# In[16]:


train.Age.isnull().sum()


# In[17]:


##########
# Easiest way to do
# train['Age']=train.Age.fillna(train.Age.mean())

###########
#To get a good assumption of missing age values
# for dataset in combine_data:
#     age_avg        = dataset['Age'].mean()
#     age_std        = dataset['Age'].std()
#     age_null_count = dataset['Age'].isnull().sum()
    
#     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
#     dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
#     dataset['Age'] = dataset['Age'].astype(int)


##########
#Another way to calculate the missing Age value
for dataset in combine_data:
    dataset["Age"] = dataset.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))


# Lets look at the distribution of age in histogram plot. Plot shows that most people  are in the range of 20-40.

# In[18]:


sns.distplot(train['Age'])


# To get a proper idea about which band of age has more survival rate, lets categorise **Age** in different band and plot them. There are different ways to do it. I have mentioned here all the methods.
# According to plot, people aged less than 16 has more survival rate than others.

# In[19]:


#for dataset in combine_data:
    #dataset['Age']=pd.qcut(dataset['Age'],5,labels=[0,1,2,3,4])
#What I did below is also possible to do with above code.

for dataset in combine_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    dataset['Age'] = dataset['Age'].astype(int)
train[['Age', 'Survived']].groupby(['Age']).mean().plot.bar()


# # **Fare**
# 
# The fare features has only one missing value in test dataset . Lets replace the missing value.
# Now we have categorised the fare is 4 bands. From the plot it is clear that  people who paid fare greater than 31 has more survival rate.

# In[20]:


for dataset in combine_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median()) #Missing value filled
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train[['Survived','Fare']].groupby('Fare').mean().plot.bar()


# # **Embarked**
# 
# The embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' ).
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Embarked feature to a new feature. From the figure it is clear that Embarked "C" has most survival rate.

# In[21]:


#Missing value filled with maximum number of time present element.

train['Embarked'] = train['Embarked'].fillna(max(train['Embarked'].value_counts().keys())) 
#Mapping
for dataset in combine_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

train[['Survived','Embarked']].groupby('Embarked').mean().plot.bar()


# # **Cabin**
# 
# Lets look at the missing values.  it has 677 missing values. Which is more than 50%. So its better to drop this feature.

# In[22]:


train['Cabin'].isnull().sum()
# This column has many null values so will drop it


# ## **Cleaning the data**
# 
# Lets remove those unnecessary features.

# In[24]:


PassengerId_train=train['PassengerId']
PassengerId_test=test['PassengerId']

#Dropping unnecessary column
drop_columns=['Cabin','Name','Ticket','PassengerId','SibSp','Parch']
for dataset in combine_data:
    dataset.drop(drop_columns,axis=1,inplace=True)


# Now we have cleaned and only numerical data. So we can proceed for modeling. Below the overview of data is showed.

# In[25]:


train.head()


# ## **Dummy Variables**
# 
# According to some data scientist this option gives better performance. I have showed below how to do it but didn't used finally. If anyone wants to use then he/she can just remove the single # lines of the below code.

# In[27]:


# Creating dummy variables 
# for dataset in combine_data:
cat_cols = ['Pclass', 'Age', 'Fare', 'Embarked', 'Title','FamilySize']
train= pd.get_dummies(train,columns = cat_cols,drop_first=True)
test= pd.get_dummies(test,columns = cat_cols,drop_first=True)


# # **Model, predict and solve**
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# * LogisticRegression,
# * KNeighborsClassifier,
# * SVC,
# * BernoulliNB,
# * DecisionTreeClassifier,
# * RandomForestClassifier,
# * AdaBoostClassifier,
# * GradientBoostingClassifier,
# * GaussianNB,
# * LinearDiscriminantAnalysis,
# * QuadraticDiscriminantAnalysis,
# * XGBclassifier
# 

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

classifiers = [
    LogisticRegression(C=0.1),
    KNeighborsClassifier(3),
    SVC(probability=True),
    BernoulliNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=100),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier()
     ]


# In[29]:


X= train.drop("Survived", axis=1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=0)


# In[30]:


log_cols = ["Classifier", "Accuracy","F1_Accuracy"]
log      = pd.DataFrame(columns=log_cols)
for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    F1_acc=f1_score(y_test, train_predictions, average='weighted') #average='micro' or average='weighted'
    log_entry = pd.DataFrame([[name, acc, F1_acc]], columns=log_cols)
    log = log.append(log_entry)
log.set_index('Classifier', inplace=True)
log.sort_values(by="Accuracy",ascending=False,inplace=True)  
#log.index=log['Classifier'].values.tolist()
log.plot(kind='bar', stacked=False, figsize=(15,8))
log


# Till above actually our kernel should be finished. The later part is written only for further analysis. Anyone who wish to go forward can just remove the # and use for tune further.
# 
# # **Extra Tuning**
# 
# There are some extra tuning  possible to improve the accuracy. 
# 1. Feature selection through builtin methods of Classifier.
# 2. By doing Grid Search for selecting best parameters.

# In[31]:


#Feature Selection : RandomForestClassifier example is used
#X_train= train.drop("Survived", axis=1)
#y_train = train["Survived"]
#X_test=test
#from sklearn.feature_selection import SelectFromModel
#parameters = {'bootstrap': False, 'max_depth': 6, 'max_features': 'log2', 'min_samples_leaf': 1,
#                                             'min_samples_split': 3, 'n_estimators': 50}
#clf=RandomForestClassifier(**parameters )
#clf.fit(X_train, y_train)
#survived = clf.predict(X_test)
#features = pd.DataFrame()
#features['feature'] = X_train.columns
#features['importance'] = clf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#features.plot(kind='barh', figsize=(20, 20))

#model = SelectFromModel(clf, prefit=True)
#train_reduced = model.transform(X_train)
#test_reduced=model.transform(X_test)

#clf=RandomForestClassifier(**parameters )
#clf.fit(train_reduced, y_train)
#survived = clf.predict(test_reduced)
#submission=pd.DataFrame({'PassengerId':PassengerId_test,'survived':survived})
#submission.to_csv('gender_submission4.csv',index=False )


# Below the Gridsearch is used for parameter tuning. I have made the whole code #taged. Anyone who wants to use this tuning just remove the # tags.

# In[32]:


#parameter_grid = {
#                 'max_depth' : [4, 6, 8],
#                 'n_estimators': [50, 10],
#                 'max_features': ['sqrt', 'auto', 'log2'],
#                 'min_samples_split': [2, 3, 8],
 #                'min_samples_leaf': [1, 3, 10],
#                 'bootstrap': [True, False],
#                 }
#forest = RandomForestClassifier()
#grid_search = GridSearchCV(forest,
#                               scoring='accuracy',
#                               param_grid=parameter_grid,
#                               cv=5)

#grid_search.fit(X_train, y_train)
#model = grid_search
#parameters = grid_search.best_params_

#print('Best score: {}'.format(grid_search.best_score_))
#print('Best parameters: {}'.format(grid_search.best_params_))


# In[33]:



#acc = accuracy_score(y_test, survived)
#F1_acc=f1_score(y_test, survived, average='macro') 


# In[ ]:




