#!/usr/bin/env python
# coding: utf-8

# ## Load labraries and data 
# First we beggin by loading the necessary libraries and data to solve the problem.
# ![](http://)Among the libraries needed are [Numpy ](http://https://www.numpy.org/)used commonly for matrix operations, [Pandas](http://https://pandas.pydata.org/) for data analysis, [Matplotlib](http://https://matplotlib.org/) for data visualization and [Sckit-learn](http://https://scikit-learn.org/stable/) for Machine Learning

# In[ ]:


#import the necessary libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier




import warnings
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')


# In[ ]:


#read the train and test csv and transform them into DataFrames
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Data Understanding

# In[ ]:


# get a look at the data
train.head()


# In[ ]:


#Look at the information about the variables
print('*'*10, ' ', 'Train data', ' ', '*'*10)
train.info()


# In[ ]:


#check if any variable in train contain NaN values
print(train.isna().sum())


# In[ ]:


#check if any variable in test contain NaN values
print(test.isna().sum())


# Excluding the Survival variable which is the dependent variable we have to predict and the PassangerId which acts only as an index and can be ignore, we have:
# * **categorical features**: 
#  *  Sex: whether the passanger was male or female
# *  **ordinal features**:
#  * Pclass: Socio-economic status of the passanger's ticket, 1(Upper), 2(Middle), 3(Lower)
#  * Age: age of the passanger in years. Has NaN values
#  * SibSp: number of sibling and spouse aboard the Titanic
#  * Parch: number of parents and children aboard the Titanic
#  * Fare: Passanger Fare, Has one NaN value
#  * Embarked; port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). Has NaN values
# * **alphanumerical** 
#  * Ticket: Ticket number, composed of letters and/or numbers
#  * Cabin: Cabin number, composed of letters and/or numbers. Has NaN values

# ## Data Cleaning

# * We will remove the Ticket variable since it does not look to have any effect
# * We will fill the Fare value with the mean value of the fare corresponding to the Pclass of the register
# * Cabin has NaN in 76% of it's values. We will delete this column since the majority are NaN
# * Embarked has only .22% of it's values as NaN. We can fill the NaN values with the most occured value, however, it does not seem really important for survival prediction
# * Age has 19.865% of its values as NaN.There are multiple ways to address this problem, one can fill these values by generiting random variables using the mean and the standard deviation as shown in [this kernel](https://www.kaggle.com/omarelgabry/a-journey-through-titanic/notebook). However I find it more reasonable to use the titles in the passanger's name to get an estimate of the passanger's age as shown [in this kernel]('https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner') 
# * We will create another feature (Title) derived from name that indicates the age and socioeconomic state of the passanger (This is normally done in Feature Engineering but we need it to fill the missing age values)
# * We will remove the Name variable once we create the title variable since it isn't usefull anymore
# 
# Remember that what you do to the training set you have to do the testing set as well

# In[ ]:


#drop cabin column
del train['Cabin']
del test['Cabin']

#drop ticket column
del train['Ticket']
del test['Ticket']

#get the most used value in Embarked and fill NaN values with it
mode = train['Embarked'].mode() # -> "S"
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = train["Embarked"].fillna("S")


#get the mean and stdev of age in train 
""""age_mean = train["Age"].mean()
age_std = train["Age"].std()
nan_train = train["Age"].isna().sum()

#get the mean and stdev of age in test 
age_mean_test = test["Age"].mean()
age_std_test = test["Age"].std()
nan_test = test["Age"].isna().sum()

# generate random values for train using mean and stdev
rand_train = np.random.randint(age_mean - age_std, age_mean + age_std, size=nan_train)

# generate random values for test using mean and stdev
rand_test = np.random.randint(age_mean_test - age_std_test, age_mean_test + age_std_test, size=nan_test)"""


# In[ ]:


#Fill the fare missing value with the mean value of the fare according to the Pclass
test['Fare'] = test['Fare'].fillna(test.groupby('Pclass').mean()['Fare'][3])


# In[ ]:


#get the titles of the passangers
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# We have 17 different titles, some titles rarely appear and I replace them with more common titles that I believe fall into the same age category.
# I assume that:
# * Capt = Capitan, Col = Colonel, Don, Dr = Doctor, Jonkheer, Major, Reverend, Sir, all fall into the same age category as Mr
# * Coutness, Mme = Madame, Lady, Dona, fall into the same age category as Mrs
# * Mlle = mademoiselle, fall into the same age category as Ms
# * Coutness, Lady, Sir, fall into a new category of people that belong to Royalty

# In[ ]:


#replace the categories as shown above
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev'],'Mr')
    dataset['Title'] = dataset['Title'].replace(['Mme','Dona'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')


# In[ ]:


test.head()


# In[ ]:



#Let's see the age distribution of the categories
gb = train[['Title', 'Age']].groupby(['Title'],as_index=False)
Masterl, Missl, Mrl, Mrsl, Royall= [], [], [], [], []
ages = {'Master': Masterl, 'Miss': Missl, 'Mr': Mrl, 'Mrs':Mrsl, 'Royal':Royall }
for key,value in gb:
    ages[key] = value['Age'].values
for x in ages.keys():
    ages[x] = [a for a in ages[x] if str(a) != 'nan']

plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.hist(ages[list(ages.keys())[i]])
    plt.xlabel(list(ages.keys())[i] + ' Age')
    plt.ylabel('Frecuency')
    plt.title(list(ages.keys())[i] + "'s age distribution")
    
plt.subplots_adjust(wspace=.5, hspace=.5)


# As wee see in the charts above the distribution of the ages is not normal, so using the mean is not the right way to fill in missing values, instead, we use the median

# In[ ]:


median_age = {}
median_age['Master'], median_age['Miss'], median_age['Mr'], median_age['Mrs'], median_age['Royal'] =  train[['Title', 'Age']].groupby(['Title'],as_index=False).median()['Age'].values
for row in range(train.shape[0]):
    if str(train['Age'][row]) == 'nan':
        train['Age'][row] = median_age[train['Title'][row]]

for row in range(test.shape[0]):
    if str(test['Age'][row]) == 'nan':
        test['Age'][row] = median_age[test['Title'][row]]
        
del train['Name']
del test['Name']


# ## Data Visualization
# I got the age visualization from [here](https://www.kaggle.com/omarelgabry/a-journey-through-titanic/notebook)

# In[ ]:


train.head()


# In[ ]:


print('Percentage survived = 1: ',((train['Survived'] == 1).sum()/ train['Survived'].count()) * 100)
print('Percentage survived = 0: ',((train['Survived'] == 0).sum()/ train['Survived'].count()) * 100)


# In[ ]:


# plot Plcass, Sex, Sibspc, Parch, Embarked and Title since they are easier to visualize
plt.figure(figsize=(15,20))
columns = ['Pclass','Sex','SibSp','Parch','Embarked','Title']
i=1
for column in columns:
    plt.subplot(5,2,i)
    sns.barplot(column, 'Survived', data=train)
    i +=1

plt.subplots_adjust(wspace=.5, hspace=.5)


# In[ ]:


# For visualizing fare we will make groups from 50 to 50 
# For visualizing age we will make groups of 5 to 5
for frame in [train,test]:
    frame['bin_age']=np.nan
    frame['bin_age']=np.floor(frame['Age'])//5
    frame['bin_fare']=np.nan
    frame['bin_fare']=np.floor(frame['Fare'])//50


# In[ ]:


x_label = [str((x-1)*5) + '-' + str(x*5) for x in range(1,17)]
average_age_bin = train[["bin_age", "Survived"]].groupby(['bin_age'],as_index=False).mean()

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(10,4))
ax = sns.barplot(x='bin_age', y='Survived', data=average_age_bin)
ax.set(xticklabels=x_label)
ax.set_xlabel('Age')
plt.show()


# In[ ]:


x_label = [str((x-1)*50) + '-' + str(x*50) for x in range(1,8)]
average_fare = train[["bin_fare", "Survived"]].groupby(['bin_fare'],as_index=False).mean()
ax = sns.barplot(x='bin_fare', y='Survived', data=average_fare)
ax.set(xticklabels=x_label)
ax.set_xlabel('Fare')
plt.show()


# **observations**:
# * People with higher PClass have higher probabilities of surviving
# * People that have 0-2 SibSp or 1-3 Parch have higher probability of survival
# * People that belong to Royalty have the higher probability of survival
# * Babies (0-10) and older people (75-80) have higer probabilities of survival. People from 25 to 35 have the lowest probabilities of survival
# * People that paid the most expensive fares had higher probabilities of survivals, likewise, people with the cheapest cabins had lower probabilities of survival
# * Female have much higher probabilities of survival
# 

# In[ ]:


test.head()


# ## Getting the data ready for modeling

# We need to change the categorical string variables into numerical values the model understands, there are different ways of accomplishing this, for example, for the Embarked variable setting, 0=S, 1=C, 2=Q. However, this can imply some ordering that doesn't make sense, like S>C>Q or that 2C = Q. For that reason we will use dummy variables that removes such relationships

# In[ ]:


train_new = pd.get_dummies(train, columns=['Sex','Embarked','Title'], drop_first=True)
test_new = pd.get_dummies(test, columns=['Sex','Embarked','Title'], drop_first=True)


# In[ ]:


test_new['Title royal'] = np.zeros(test_new.shape[0], dtype=int)


# In[ ]:


train_new.head()


# In[ ]:


X_train= train_new.drop(['PassengerId','Survived'], axis=1)
Y_train = train_new['Survived']
X_test = test_new.drop(['PassengerId'],axis=1)


# ## Modelling

# We will be training different models in order to compare the performance. This models are:
# * Naive Bayes
# * KNN
# * SVC
# * Decision Tree
# * Random Forest
# * Gradient Boosting Classifier
# * Random Forest

# First we split the training set into training and validation(10%) to test the accuracy of our models

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split( X_train, Y_train, test_size=0.1, random_state=42)


# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_val)
gaussian_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
gaussian_acc


# In[ ]:


#K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
knn_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
knn_acc


# In[ ]:


#Suport Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_val)
svc_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(svc_acc)


# In[ ]:


#Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
y_pred = decisiontree.predict(X_val)
decision_tree_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
decision_tree_acc


# In[ ]:


#Random Forest
randomforest = RandomForestClassifier(n_estimators=10000)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_val)
random_forest_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
random_forest_acc


# In[ ]:


#Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_val)
gbc_acc = round(accuracy_score(y_pred, y_val) * 100, 2)
gbc_acc


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_val)
logreg_acc= round(accuracy_score(y_pred, y_val) * 100, 2)
logreg_acc


# In[ ]:


models = ['Naive Bayes', 'KNN', 'Support Vector Classifier', 'Decision Tree', 'Random Forest', 'Gradient Boosting Classifier', 'Logicstic Regresion' ]
scores = [gaussian_acc, knn_acc, svc_acc, decision_tree_acc, random_forest_acc, gbc_acc, logreg_acc]
data = {'Models':models, 'Accuracy': scores}
results = pd.DataFrame(data)
results.sort_values(by='Accuracy', ascending=False)


# We get the higher accuracy from Decision Tree and Random Forest, therefore, I will use Random Forest to predict X_test

# ## Submission 

# In[ ]:


passengerIds = test_new['PassengerId']
predictions = randomforest.predict(X_test)
result = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions})
result.to_csv('predictions.csv', index=False)

