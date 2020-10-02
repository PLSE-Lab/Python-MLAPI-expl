#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook we will see the necessary stages required to reach a descent score on the **Titanic: Machine Learning from Disaster** dataset. We wll be going through all the steps namely
# 
# * Problem Definition
# * Hypothesis Generation
# * Data Extraction
# * Data Exploration and Transformation
# * Model Building
# * Deployment
# 
# # Problem Definition:
# The definition of the problem is pretty staright forward, given the attributes we require to build a classification model, which would be able to accurately predict whether some passenger survived the Titanic Disaster or not
# 
# # Hypothesis Generation:
# In this step we would have to be a little creative and try think outside the box, we would have to come up with different hypothesis for the cases of survival of this tragedy. We may have some intuition such as, *females and children is more likely to be prioritized for rescue*, *Survival rate should increase as the fare increases*(Since the higher paying passenger may be an important person) and so on. For more, we would have to dive deep into the dataset itself. There are a lot of kernels with very descriptive insights to the attributes of the dataset. Our task is to validate these hypothesis during the **Data Exploration** phase.
# 
# ## Data Extraction:
# We dont have much to do in this stage, since the data is provided by Kaggle in a CSV format.
# 

# # Imports
# ### For Exploratory Data Analysis

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
sns.set_style('whitegrid')


# **Read the train and test csv files**<br/>
# One thing to remember is if we transform the train set by even a little, then the same transformation would have to be applied to the test dataset as well.

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# Lets check out the head of our training data.

# In[ ]:


train.head(3)


# In[ ]:


train.describe()


# Now lets see the details of our training dataframe using the ```DataFrame.info()``` method.<br/>
# We can already see that there are quite a lot of data missing in the **Cabin** column and **Age** column. We will be handling these missing informations later on. First lets split our Data Exploration stage into some substages.<br>
# 
# Following this checklist it'll be sure that we are not missing out on some important step.
# 
# * **Reading the Data** (*.csv* to pandas DataFrame)
# * **Variable Identification** (Differentiate the **dependent variables** from **independent variables**, Also check which variables are **Continuous** and which are **Categorical**)
# * **Univariate Analysis**
# * **Bivariate Analysis**
# * **Missing Value Treatment**
# * **Outlier Treatment**
# * **Variable Transformation** and **Feature Engineering**
# Now it is clear to us that our Dependent/Target Variable is **Survived**. All the other attributes in the dataset are our Independet/Predictor Variables. We now need to have a clear understanding of what is a Categorical vs. Continuous Variable. A **Categorical variable** is a variable that can take on one of a limited, and usually fixed, number of possible values. Refer to this <a href='https://en.wikipedia.org/wiki/Categorical_variable'>Wikipedia</a> link for more info. Similarly a <a href='https://en.wikipedia.org/wiki/Continuous_or_discrete_variable'>Continuous Variable</a> is one which can take on an uncountable set of values.

# In[ ]:


train.info()


# # Variable Identification

# In[ ]:


# Variable Identification

dtypes = ['int64','float64','object']
for d in dtypes:
    print('\n','*'*20,'{} type Attributes'.format(d.upper()), '*'*20, end='\n\n')
    print(list(train.select_dtypes(d).columns), end='\n\n')
    for field in list(train.select_dtypes(d).columns):
        print('{} ==> {}'.format(field, 'CONTINUOUS' if train[field].value_counts().count() > 10 else 'CATEGORICAL'))


# # Univariate Analysis
# Now we need to perform univariate analysis. In this stage we only perform analysis on one feature at a time. Thus the name 'univariate'

# In[ ]:


# Univariate Analysis

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

sns.countplot(x='Survived', data=train, ax=ax[0])
sns.countplot(x='Sex', data=train, ax=ax[1])

ax[0].set_title('Survived vs. Casualties')
ax[1].set_title('Male vs. Female Count')


# Now in the hypothesis generation stage, we had an intuition that maybe females were prioritized more over males in the rescue process, now lets vvirusalize that using countplot

# In[ ]:


sns.countplot(x='Survived', data=train, hue='Sex')


# So our intuition was correct, when it comes to survived passengers most of them are females. Now lets see these in terms of percentage.

# In[ ]:


# Lets check the percentage of people who survived

percentage_of_survived = 100 * (train['Survived'].value_counts() / train.shape[0])

# Percentage of Males who survivied

survived_male = train[(train['Survived']==1) & (train['Sex']=='male')].shape[0]
total_male = train[train['Sex']=='male'].shape[0]

# Percentage of Females who survivied
survived_female = train[(train['Survived']==1) & (train['Sex']=='female')].shape[0]
total_female = train[train['Sex']=='female'].shape[0]

print('Percentage of passngers in both target classes.')
print(percentage_of_survived)
print('Percentage of male passengers who survived: {:0.2f}'.format(100*(survived_male/total_male)))
print('Percentage of female passengers who survived: {:0.2f}'.format(100*(survived_female/total_female)))


# So it looks like **62%** passengers in our training set didnt make it, and the survival rate of females are higher than male's. Which means that the attribute **Sex** will play a significant part in our final predictions.
# <br/>
# Now lets take a look into the different passenger classes (Pclass).

# In[ ]:


# Lets see the countplot for different passenger classes

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

sns.countplot(x='Pclass', data=train, ax=ax[0])
# Lets see the counplot of passenger class w.r.t
sns.countplot(x='Pclass', data=train, hue='Survived', ax=ax[1])

ax[0].set_title('Passenger Class Counts')
ax[1].set_title('Pclass with respect to Survival')


# ## Distributions of numerical columns and identification of Outliers
# 
# Now lets plot histograms of the continuous numerical variables of our dataset.

# In[ ]:


# Distribution of different ages in the training dataset

sns.distplot(train['Age'], bins=30)


# Note that the distribution of age is pretty much normal, whereas most of our values reside in the range of 20 to 40. Whereas in the following distplot of the column 'Fare' we see that most of the Fares are in between 0 and 100, but we have a lot of outliers, which is why our plot extends to 500 in the x-axis

# In[ ]:


sns.distplot(train['Fare'], bins=30)


# # Bivariate Analysis
# 
# Let us plot the continuous numerical columns w.r.t each other

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
sns.scatterplot(x='Age', y='Fare', data=train, ax=ax[0])
sns.scatterplot(x='Age', y='Fare', data=train, hue='Survived', ax=ax[1])
ax[0].set_title('Age vs. Fare')
ax[1].set_title('Age vs. Fare w.r.t  Survived')


# In[ ]:


# Similarly for Passenger Class and age
plt.figure(figsize=(10,4))
sns.boxplot(x='Pclass', y='Age', data=train)


# This boxplot makes sense, since the it required more money to be a first class passengers, and with age people acquired more money, which is why the median age of Passenger class 1 is higher than thosee of 2 and 3.

# We can look into the pairplot of the numerical columns, but there wont be anything significant there, since most of the numerical values are categorical.

# # Feature Engineering
# 
# Firstly, since most of the rows doesnt have an entry in the **Cabin** column, let us create another field called **cabindata** which signifies, if the passenger has a entry in the Cabin field or not.<br/>
# *cabindata = 0* if the corresponding Cabin field is null, else<br/>
# *cabindata = 1*

# In[ ]:


train['cabindata'] = train['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
# We apply the same modifications to the test dataset as well
test['cabindata'] = test['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)


# Now lets see if this newly created column is any helpful to us.

# In[ ]:


# Passenger with cabin data present in the dataset
total_cabin_passenger = train[train['cabindata']==1].shape[0]
# Passenger with no Cabin data 
total_non_cabin_passenger = train[train['cabindata']==0].shape[0]

# Passengers with cabin data present and survived
total_cabin_survived_passenger = train[(train['cabindata']==1) & (train['Survived']==1)].shape[0]
# Passengers with cabin data abset and survived
total_non_cabin_survived_passenger = train[(train['cabindata']==0) & (train['Survived']==1)].shape[0]

# Percentage of people who had cabin info and survivied
print('Percentage of people who had cabin information and Survived: {:.2f}'.format(100*(total_cabin_survived_passenger/total_cabin_passenger)))

# Percentage of people who didnt have cabin info and survived
print('Percentage of people who didnt have cabin information and Survived: {:.2f}'.format(100*(total_non_cabin_survived_passenger/total_non_cabin_passenger)))


# So it seems like the present of cabindata plays a very important role in the final outcome, which means our cabindata column will be highly correlated with our target variable *Survived*. Now lets take a look at the missing values in the dataset.

# In[ ]:


train.isnull().sum()


# It looks like we have some values missing in the Age column and most of them missing in the Cabin column. Now there are two ways of handling missing data.
# 
# 1. **Imputation**
# 2. **Dropping**
# 
# In imputation, we fill the missing values with appropriate data. Such as for,<br/>
# 
# * Columns which are Continuous
#     * We can take the **Mean** Value of the existing rows
#     * We can take the **Median** Value of the existing rows
#     * We can consider these missing values as a **Regression** problem in itself
# * Columns which are Categorical
#     * We can take the **Mode** Value of the existing rows
#     * Or, we can consider these missing values as a **Classification** problem itself
#     
# In the case of the field Age we can impute the missing rows with the median age values based on Passenger Class.
# One way to visualize the Missing rows in the dataset is by creating a **Heatmap** of the boolean DataFrame returned from ```isnull()``` method.

# In[ ]:


# Another way is by visualizing using a heatmap
plt.figure(figsize=(10,6))
sns.heatmap(train.isnull())


# In[ ]:


# We are going to be performing our imputation operation using this method

def impute_age(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return np.median(train[train['Pclass']==1]['Age'].dropna().values)
        elif pclass == 2:
            return np.median(train[train['Pclass']==2]['Age'].dropna().values)
        else:
            return np.median(train[train['Pclass']==3]['Age'].dropna().values)
    else:
        return age


# Now lets go ahead and impute the missing rows with proper values and apply the same to the test set

# In[ ]:


# Applying the function to fill in the missing values
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis=1)


# Lets take a look at what percentage of Cabin data missing. It that column recoverable by any means?

# In[ ]:


# Percentage of cabin info missing from the dataframe

print('Percentage of rows missing Cabin data:',100 * (train.isnull()['Cabin'].sum() / train.shape[0]))


# It seems like there is no hope, with this field so lets just go ahead and drop it from both train and test datasets

# In[ ]:


# As 77% data is missing we can simply drop the column from both the dataframes
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# In[ ]:


# Now lets check the heatmap of our isnull() boolean dataframe
sns.heatmap(train.isnull())
# Number of missing rows
print('Number of rows missing Embarked:',train.isnull()['Embarked'].sum())


# As only two rows are missing the Embarked column value, lets just drop them.

# In[ ]:


train.dropna(axis=0, inplace=True)


# In[ ]:


# In this stage we will fix all the missing fair values if present in some row
train['Fare'] = train['Fare'].apply(lambda x: np.mean(train['Fare'].values) if pd.isnull(x) else x)
test['Fare'] = test['Fare'].apply(lambda x: np.mean(train['Fare'].values) if pd.isnull(x) else x)


# Now its time for us to create dummy variables from the object type categorical variables using the method ```pandas.get_dummies()```. The columns we would have to alter are the **Embarked** and **Sex** columns. Since the variable **Embarked** can have any of the three values - <br/>
# * C = Cherbourg
# * Q = Queenstown
# * S = Southampton
# 
# <br/>and the variable **Sex** can have any of the two value - <br/>
# * male
# * female

# In[ ]:


# Creating dummy variables from categorical variables

train_se = pd.get_dummies(data=train['Sex'], drop_first=True)
train_em = pd.get_dummies(data=train['Embarked'], drop_first=True)

train = pd.concat([train, train_se, train_em], axis=1)

train.head(3)


# In[ ]:


# Same transformation is being applied to the test set

test_se = pd.get_dummies(data=test['Sex'], drop_first=True)
test_em = pd.get_dummies(data=test['Embarked'], drop_first=True)

test = pd.concat([test, test_se, test_em], axis=1)


# In[ ]:


# Now that we have turned the categorical columns 'Sex' and 'Embarked' into ordinal values we can go ahead and drop these fields

train.drop(['Sex','Embarked'], axis=1, inplace=True)
test.drop(['Sex','Embarked'], axis=1, inplace=True)


# For the sake of simplicity, let us not spend more time in feature engineering. So we are dropping the text columns, **Name** and **Ticket**.<br/>
# ### NOTE: 
# We can extract various features from these fields, I'll leave links to more kernels where feature engineering is more focused on. Please refer to those for an overall sense of what can be obtained from these attributes. We are also skipping the segment for handling outliers, you can very easily drop the rows containing values way out of proportion just by using pandas.

# In[ ]:


# Now we can go ahead and drop the attributes Name and Ticket

train.drop(['Name', 'Ticket'], axis=1, inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)


# ## Correlation
# Now lets see the correlations between the different variable of our explored dataset.

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(train.corr(), annot=True)


# In[ ]:


# How the independent variables are correlated with dependent
plt.figure(figsize=(10,6))
train.corr()['Survived'].sort_values(ascending=False)[1:].plot(kind='bar',fontsize=16)


# As per our prediction as we can see that cabindata is very highly correlated with **Survived** also our dummy column male is highly negatively correlated with our dependent variable which makes sense. Since when male is 1, then as we saw there is a high chance of that passenger not surviving, which is why the negative correlation.

# # Model Building
# 
# Now that we have explored our dataset and turned it into a DataFrame consisting of entirely numerical values, lets apply various Machine Learning algorithms and see which one gives us the best score.

# In[ ]:


# Imports for Machine Learning

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# For the better evaluation of our models let us split the train.csv data into two parts, train and test.

# In[ ]:


# For the sake of testing and evaluation of the models, we will be splitting the data residing in train.csv into two parts
dummy_X = train.drop(['PassengerId', 'Survived'], axis=1)
dummy_y = train['Survived']


# In[ ]:


# Now splitting our train.csv data into two parts

X_train, X_test, y_train, y_test = train_test_split(dummy_X, dummy_y, test_size=.25, random_state=7)


# # Logistic Regression

# In[ ]:


log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
log_pred = log.predict(X_test)
print('Score: ', round(100*(log.score(X_test, y_test)), 3))
print('-'*20,'log model','-'*20)
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))


# # SGD Classifier

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
print('Score: ', round(100*(sgd.score(X_test, y_test)), 3))
print('-'*20,'sgd model','-'*20)
print(confusion_matrix(y_test, sgd_pred))
print(classification_report(y_test, sgd_pred))


# # Random Forest Classifier
# First let to find a good value for ```n_estimators```

# In[ ]:


rfor_scores = []

# Trying to find the best number of estimator value

for estimator in range(1,40):
    rfor = RandomForestClassifier(n_estimators=estimator)
    rfor.fit(X_train, y_train)
    rfor_scores.append(round(rfor.score(X_test, y_test), 2))
    
rfor_scores[:10]


# We just ran 39 RandomForestClassifer instances and trained it with our train data, and we stored the scores of the model in the array ```rfor_scores```. Now lets see which value of ```n_estimators``` gives us the best score.

# In[ ]:


# PLotting the scores

xticks = np.arange(1,41)
plt.figure(figsize=(10,6))
plt.plot(np.arange(1,40), rfor_scores,'red', ls='dashed', marker='o', markersize=10, markerfacecolor='blue')
plt.xticks(ticks=xticks)
plt.xlabel('Estimators in Consideration')
plt.ylabel('Model Score')
plt.tight_layout()


# In[ ]:


rfc = RandomForestClassifier(n_estimators=np.array(rfor_scores).argmax()+1)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print('Score: ', round(100*(rfc.score(X_test, y_test)), 3))
print('-'*20,'rfc model','-'*20)
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


# # Linear SVC
# 
# This is essentially a SVM with linear kernel. Whereas usually normal SVC makes use of ```kernel=rbf``` (Radial basis function kernel).

# In[ ]:


lsvc = LinearSVC(max_iter=1000)
lsvc.fit(X_train, y_train)
lsvc_pred = lsvc.predict(X_test)
print('Score: ', round(100*(lsvc.score(X_test, y_test)), 3))
print('-'*20,'lsvc model','-'*20)
print(confusion_matrix(y_test, lsvc_pred))
print(classification_report(y_test, lsvc_pred))


# # Support Vector Classifier
# In this case we will be making use of ```GridSearchCV``` from ```sklearn.model_selectoin``` which would give us the best svc() object containing the best values for the hyperparameters **C** and **gamma**

# In[ ]:


grid = GridSearchCV(estimator=SVC(), param_grid={'C':[0.001,0.01,0.1,1,10,100],
                                                 'gamma': [0.01,0.1,1,10,100]}, verbose=.5)
grid.fit(X_train, y_train)


# Lets see how our svc model performs

# In[ ]:


svc = grid.best_estimator_
print('Best Parameters:',grid.best_params_)
print('Best Scores:',grid.best_score_)


# In[ ]:


svc_pred = svc.predict(X_test)
print('Score: ', round(100*(svc.score(X_test, y_test)), 3))
print('-'*20,'svc model','-'*20)
print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))


# Well it looks like linear kernel was a better choice. So lets change the kernel type of this model and see if the performance improves

# In[ ]:


# One last attempt with linear kernel

svc.kernel = 'linear'
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print('Score: ', round(100*(svc.score(X_test, y_test)), 3))
print('-'*20,'svc model','-'*20)
print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

# Ok, so its a little better


# Now lets compare the different model's performance.

# In[ ]:


# Comparison of scores

scores_df = pd.DataFrame(data=[
    round(100*(log.score(X_test, y_test)), 3),
    round(100*(sgd.score(X_test, y_test)), 3),
    round(100*(rfc.score(X_test, y_test)), 3),
    round(100*(lsvc.score(X_test, y_test)), 3),
    round(100*(svc.score(X_test, y_test)), 3)
], index = ['LogisticRegression SGDClassifier RandomForestClassifier LinearSVC SVC'.split()],
    columns = ['Scores'])
scores_df


# In[ ]:


plt.figure(figsize=(8,6))
scores_df.sort_values(by='Scores', ascending=False).plot(kind='bar', fontsize=15, legend=False)


# Now that we know see that RandomForestClassifier works the best, lets apply this algorithm to our entire training dataset.

# In[ ]:


# Applying Random Forest on real training set to obtain the predictions

X_train = train.drop(['PassengerId', 'Survived'], axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId'], axis=1)

print('X_train: {}\ny_train: {}\nX_test: {}'.format(X_train.shape, y_train.shape, X_test.shape))


# In[ ]:


# Final model
random_forest = RandomForestClassifier(n_estimators=np.array(rfor_scores).argmax()+1)
random_forest.fit(X_train, y_train)
predictions = random_forest.predict(X_test)


# In[ ]:


print('Score: {}'.format(round(random_forest.score(X_train, y_train))))


# Now lets create a DataFrame containing the PassengerId and our predictions

# In[ ]:


submission = pd.DataFrame(data = {
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# # Conclusion:
# 
# If you want more in depth feature engineering, please refer to the following very nicely written notebooks. Thank you. :)
# 
# 1. Advanced Feature Engineering: https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
# 2. Overall Guide: https://www.kaggle.com/startupsci/titanic-data-science-solutions

# In[ ]:




