#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>
# <font color="blue">
# TITANIC:MACHINE LEARNING FROM DISASTER
# </font>
# </h1>
# 
# 

# <h2>
# <font color="blue">
# Predict survival on the Titanic
# </font>
# </h2>
# 
# 

# 1. Define the problem statement
# 2. Collecting the data
# 3. EDA
# 4. Feature Engineering
# 5. Feature selection
# 6. Modelling
# 7. Testing
# 

# <h2>
# <font color="blue">
# 1.0 Defining the problemm statement
# </font>
# </h2>
# 
# 

# Analyse people who are likely to survive. Use tools of machine learning to predict which passengers survived the tragedy

# <h2>
# <font color="blue">
# 2.0 Collecting the Data
# </font>
# </h2>
# 
# 

# <h3>
# <font color="blue">
# load train test dataset using pandas
# </font>
# </h3>
# 
# 

# In[ ]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# <h2>
# <font color="blue">
# 3.0 Exploratory data analysis(EDA)
# </font>
# </h2>
# 
# 

# <h3>
# <font color="blue">
# Printing 1st five rows of the data
# </font>
# </h3>
# 
# 

# In[ ]:


train.head()


# In[ ]:


test.head()


# <h3>
# <font color="blue">
# Data dictionary
# </font>
# </h3>
# 
# 

# 1. Survived: 0=No, 1=Yes
# 2. Pclass: Ticket class 1=1st, 2=2nd,3=3rd
# 3. SibSp: # of siblings/spouses aboard the titanic
# 4. Parch: # of Parents/children aboard the titanic
# 5. Ticket: ticket number
# 6. Cabin: cabin number
# 7. Embarked: Port of embarkation C=Cherbourg, Q=Queenstown, S=Southampton

# In[ ]:


train.shape


# In[ ]:


test.shape


# We can see that test data lacks 'Survived' column which is the target variable

# <h4>
# <font color="blue">
# Summary info about the dataset
# </font>
# </h4>
# 
# 

# In[ ]:


train.info()


# This summary shows that there are some missing fields in the Age, Cabin and Embarked column

# In[ ]:


test.info()


# Age, Fare and cabin have missing fields in the test dataset

# <h3>
# <font color="blue">
# Viewing missing values in train and test datasets
# </font>
# </h3>
# 
# 

# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


# Proposed themes: darkgrid, whitegrid, dark, white, and ticks
#sns.set_style("whitegrid")
#sns.boxplot(data=data)
#plt.title("whitegrid")


# <h3>
# <font color="blue">
# Visualizations
# </font>
# </h3>
# 
# 

# <h4>
# <font color="blue">
# Bar charts for categorical features
# </font>
# </h4>
# 
# 

# 1. Pclass
# 2. Sex
# 3. SibSp
# 3. Parch
# 4. Embarked 
# 5. Cabin

# In[ ]:


def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['survived','dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[ ]:


bar_chart('Sex')


# The chart confirms women more likely survived than men

# In[ ]:


bar_chart('Pclass')


# The chart confirms 3rd class more likely died and 1st class more likely survived

# In[ ]:


bar_chart('SibSp')


# The chart confirms that passengers without siblings or spouse more likely died and passengers with siblings or spouse more likely survived

# In[ ]:


bar_chart('Parch')


# The chart shows that passengers withouth parents or children more likely died and 
# passengers with parents or children more likely survived 

# In[ ]:


bar_chart('Embarked')


# 1. The chart shows that passengers aboarded from C slighlty more likely survived.
# 2. The chart shows that passengers aboarded from Q more likely died.
# 3. The chart shows that passengers aboarded from S more likely died.

# <h2>
# <font color="blue">
# 4.0 Feature Engineering
# </font>
# </h2>
# 
# 

# Feature engineering involves creation of new features and transformation of some existing features for machine learning algorithms

# In[ ]:


train.head(3)


# <h3>
# <font color="blue">
# 4.1 How the titanic sank
# </font>
# </h3>
# 
# 

# It first hit the iceberg from its head and started going down to the water therefore 3rd class passengers had more chances to die than the 2nd class and the 1st class. therefore Pclass and the cabin are more informative.

# In[ ]:


from scipy.misc import imread
from pylab import imshow, show

imshow(imread('titanic-pic.jpg'))
show()


# <h3>
# <font color="blue">
# 4.2 Name
# </font>
# </h3>

# The name is not so much informative. However we can extract the title from the name; mrs,mr or miss. they can be informative. the title indicates wether the passenger is a man, woman married or not married. From data analysis we so that more females survived than males, therefore the title is a little informative.

# First we need to combine the two dataset using concat

# In[ ]:


alldata=[train,test] #combining the train and test dataset
for dataset in alldata:
    dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)


# In[ ]:


train.Title.value_counts()


# In[ ]:


test.Title.value_counts()


# <h3>
# <font color="blue">
# Title mapping using only mr.:0, mrs.:1 miss:2 and others:3
# </font>
# </h3>

# In[ ]:


title_mapping={'Mr':0,'Mrs':1,'Miss':2,'Master':3,
               'Dr':3,'Rev':3,'Col':3,'Major':3,'Mlle':3,'Countess':3,'Ms':3,'Lady':3,'Jonkheer':3,
               'Don':3,'Mme':3,'Capt':3,'Sir':3,'Dona':3,} 
for dataset in alldata:
    dataset['Title']=dataset['Title'].map(title_mapping)
                                                                                                                 


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


bar_chart('Title')


# From the barchart mr had high chances of dying 

# <h4>
# <font color="blue">
# Drop name and title
# </font>
# </h4>

# In[ ]:


train.drop('Name',axis=1, inplace=True)
test.drop('Name', axis=1,inplace=True)


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


#  <h3>
# <font color="blue">
# 4.3 Sex
# </font>
# </h3>

# Mapping sex; male:0,female:1

# In[ ]:


sex_mapping={'male':0,'female':1}
for dataset in alldata:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)


# In[ ]:


train.head(2)


# In[ ]:


bar_chart('Sex')


# From the bars more female survived than males.

# <h3>
# <font color="blue">
# 4.4 Age
# </font>
# </h3>

# Check missing values on age

# In[ ]:


train[train.Age.isnull()].shape


# this shows that 177 rows have missing values for age

# <h5>
# <font color="blue">
# fill missing age values with median age for each title
# </font>
# </h5>

# In[ ]:


train.Age.fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test.Age.fillna(test.groupby('Title')['Age'].transform('median'),inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# All the missing values on age have been imputed with median title ages

# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.show()


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(0,20)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(20,35)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(35,40)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(40,60)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(60)


# from the plot we can see that:
# 1. until 16yrs old, there are high chances to survive
# 2. 16-35yrs had high chances to die
# 3. 35-40yrs had high chances to die
# 4. 45-60yrs had high chances to survive
# 5. 60yrs and above had high chances to die

# <h3>
# <font color="blue">
# Binning Age
# </font>
# </h3>

# Binning converts continous data into categorical
# 1. child=0
# 2. young=1
# 3. adult=2
# 4. mid_age=3
# 5. senior=4

# In[ ]:


for dataset in alldata:
    dataset.loc[ dataset['Age']<= 15, 'Age']=0,
    dataset.loc[(dataset['Age']>15) & (dataset['Age']<=35),'Age']=1,
    dataset.loc[(dataset['Age']>35) & (dataset['Age']<=45),'Age']=2,
    dataset.loc[(dataset['Age']>45) & (dataset['Age']<=60),'Age']=3,
    dataset.loc[ dataset['Age']>60, 'Age']=4


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


bar_chart('Age')


# <h3>
# <font color="blue">
# 4.5 Embarked
# </font>
# </h3>

# <h3>
# <font color="blue">
# Filling the missing values for embarked
# </font>
# </h3>
# 

# Embarked: Where did the passengers aboard the titanic, the cities they came from. Embarked is informative because the city this ppl lived can point the Pclas they took. For example ppl from rich cities will take 1st class and vise versa.
# The column embarked has some missing values.

# In[ ]:


#checking the majority of embarkation from various classes
Pclass1=train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2=train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3=train[train['Pclass']==3]['Embarked'].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))


# 1. more than 50% of 1st class are from S embark
# 2. more than 50% of 2nd class are from S embark
# 3. more than 50% of 3rd class are from S embark

# majority of embarkation from all classes was from S. i will therefore fill the missing values with S

# <h4>
# <font color="blue">
# filling missing values
# </font>
# </h4>

# In[ ]:


for dataset in alldata:
    dataset['Embarked']=dataset['Embarked'].fillna('S')


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# <h4>
# <font color="blue">
# mapping embarked
# </font>
# </h4>

# We should transform embarked feature from string to categorical

# In[ ]:


embarked_mapping={'S':0,'C':1,'Q':2}
for dataset in alldata:
    dataset["Embarked"]=dataset['Embarked'].map(embarked_mapping)


# In[ ]:


train.head()


# <h3>
# <font color="blue">
# 4.6 Fare
# </font>
# </h3>
# 

# I will fill the missing fare values with median fare for each class because fare is closely related to the class

# In[ ]:


train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train.Fare.max()))
facet.add_legend()
plt.show()


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()
plt.xlim(0,20)


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()
plt.xlim(20,40)


# passengers with cheap ticket died more while majority with expensive ticket survived

# <h4>
# <font color="blue">
# Convert fare into categorical through binning
# </font>
# </h4>
# 

# In[ ]:


for dataset in alldata:
    dataset.loc[ dataset['Fare']<= 15, 'Fare']=0,
    dataset.loc[(dataset['Fare']>15) & (dataset['Fare']<=30),'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare']=2,
    dataset.loc[(dataset['Fare']>100),'Fare']=3


# In[ ]:


train.head()


# <h3>
# <font color="blue">
# 4.7 Cabin
# </font>
# </h3>
# 

# Cabin is a room in the ship, it is informative; third class cabin had high chances to die while 1st class cabin had high chances to survive

# In[ ]:


train.Cabin.value_counts()


# In[ ]:


for dataset in alldata:
    dataset['Cabin']=dataset['Cabin'].str[:1]


# In[ ]:


#checking the majority of cabin from various classes
Pclass1=train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2=train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3=train[train['Pclass']==3]['Cabin'].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))


# from the barchart it is evident that 1st class had cabin A,B,C,D & E

# In[ ]:


cabin_mapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8}
for dataset in alldata:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
train.head()


# In[ ]:


#fill the missing cabin values with median cabin value of each Pclass
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
test['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
train.head()


# <h3>
# <font color="blue">
# 4.8 Family size
# </font>
# </h3>

# we need to create another feature, called family size: we get the family size by adding every passenger
# to the Parch and Sibsp

# In[ ]:


train['Family_size']=train['SibSp']+train['Parch']+1
test['Family_size']=test['SibSp']+test['Parch']+1
train.head()


# In[ ]:


facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Family_size',shade=True)
facet.set(xlim=(0,train['Family_size'].max()))
facet.add_legend()


# In[ ]:


#changing continous family size variable into categorical values, we use mapping
family_mapping={1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0}
for dataset in alldata:
    dataset['Family_size']=dataset['Family_size'].map(family_mapping)
train.head()


# <h4>
# <font color="blue">
# Dropping unneccesary features
# </font>
# </h4>

# some features are not informative, so we are going to drop some of them

# In[ ]:


#dropping Ticket, SibSp and Parch columns
features_drop=['SibSp','Parch','Ticket']
train=train.drop(features_drop,axis=1)
test=test.drop(features_drop,axis=1)
train=train.drop(['PassengerId'],axis=1)
train.head(2)


# In[ ]:


test.head()


# In[ ]:


train_data=train.drop('Survived',axis=1)
target=train['Survived']


# In[ ]:


train_data.shape


# In[ ]:


target.shape


# <h2>
# <font color="blue">
# 5.0 Modelling
# </font>
# </h2>

# <h3>
# <font color="blue">
# importing classifier modules
# </font>
# </h3>

# In[ ]:


#importing classifier modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# <h3>
# <font color="blue">
# Cross validation
# </font>
# </h3>

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold=KFold(n_splits=10,shuffle=True,random_state=0)


# In[ ]:


clf=KNeighborsClassifier(n_neighbors=13)
scoring= 'accuracy'
score=cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=1,scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100,2)


# <h2>
# <font color="blue">
# 6.0 Testing
# </font>
# </h2>

# In[ ]:


clf=KNeighborsClassifier(n_neighbors=13)
clf.fit(train_data,target)
test_data=test.drop('PassengerId',axis=1).copy()
prediction=clf.predict(test_data)


# In[ ]:


submission=pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':prediction
})
submission.to_csv('submission.csv',index=False)


# In[ ]:


submission=pd.read_csv('gender_submission.csv')


# In[ ]:


submission.head()


# <h1>
# <font color="red">
# references
# </font>
# </h1>

# 1. Titanic: Machine Learning from Disaster
# 2. Kaggle - Titanic Solution [3/3] - Classifier, Cross Validation
# 3. Kaggle - Titanic Solution [2/3] - Feature Engineering
# 4. https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-5:-Feature-Engineering
