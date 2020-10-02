#!/usr/bin/env python
# coding: utf-8

# This is my first attempt at a Machine Learning project after completing Andrew Ng's[ Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) course on Coursera.<br> It is also my first time using Jupyter Notebook.<br><br>
# As this was my first project I set myself some simple aims:
# 
# 1. Put what I had learned on the course into practice
# 2. Develop a workflow through a machine learning project
# 3. Learn from how other people have tackled this challenge
# 4. Come in the top half of the competition.<br>
# 
# I am using Python for this kernel.<br>
# The workflow that I am going to follow is:<br>
# 
# 1. Load the data<br>
# 2. Summarise and explore the data<br>
# 3. Complete missing data<br>
# 4. Create new features<br>
# 5. Evaluate a range of models <br>
# 6. Choose the best model and create a submission file.
# 
# I have read and re-used the content from a number of other people's notebooks and in particular would like to say thank you to [Sina Khorami](https://www.kaggle.com/sinakhorami) for his clearly explained [Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier) and [Poonam Ligade](https://www.kaggle.com/poonaml) for her comprehensive [Titanic Survival Prediction End to End ML Pipeline](https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline). I also referred to [Jason Brownlee's blog](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/), especially for the model evaluation part of the work.
# 
# 

# # 1. Load the data<br>
# First off, we need to import the libraries that we are going to be using in this kernel.<br> 

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBClassifier # <3


# Now, load in the training and test datasets. These have been provided as comma-separated variable(csv) files. We load each csv file as a Pandas DataFrame object. By doing this we can use the [public pandas objects, functions and methods](https://pandas.pydata.org/pandas-docs/stable/api.html)

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # 2. Summarise and explore the data<br>
# Now we make our first use of pandas through its *info()* function. This lists the data columns in the* train* DataFrame, the number of non-null values and the data type for each of them.

# In[ ]:


print(train.info())


# Now we do the same for the *test* DataFrame.

# In[ ]:


print(test.info())


# Train has 12 columns and test has 11. Test is lacking the "Survived" column, which is what we are going to calculate as our answer to the exercise. Let's hope that train has 891 rows and test 418, as a lot of the columns have that number of non-null values.<br><br>
# Before we can start running our models across our data we need all of our columns to be complete and holding numerical values.

# Next, let's take a look at some of the data itself. First, the training set

# In[ ]:


train.head()


# I'll just observe here that Survived is a Boolean field with a 0 standing for the passenger not surviving and a 1 meaning that they did survive.

# In[ ]:


(test.head())


# Taking a look at the data in the train and test DataFrames, and referring back to the [overview of the data](https://www.kaggle.com/c/titanic/data) we can see the following:<br>
# **PassengerId** is an incremental field that is most likely used as a unique identifier for a passenger record or row.<br>
# **Pclass** is the class of the ticket and is a proxy for social class. <br>
# **Name** is the passenger name. It includes the title/honorific of the passenger. There is a cumbersome method for naming women who are married.<br>
# **Sex** is either male or female.<br>
# **Age** is a float value. Babies who are less than one have a fractional age value. An age in the form 'x.0' is a known age whereas an age in the form 'y.5' is an estimated age. Not all missing ages have been estimated.<br>
# **SibSp** is an integer and refers to the number of their siblings plus spouse on board with the passenger.<br>
# **Parch** is an integer and refers to the number of parents and children on board with the passenger.<br>
# **Ticket number** is an alphanumeric field. Where it has been prefixed with characters there is one (row three in the training set) that looks like it indicates the port of embarkation (STON could indicate Southampton, which is the port of embarkation for that passenger). The first digit in the ticket number looks like it indicates the ticket class.<br>
# **Fare** is a float and is the price paid for the ticket.<br>
# **Cabin** is the cabin that the passenger was in. There is a lot of missing data her. Looking at the rows we have printed out the missing data may be weighted by Pclass.<br>
# **Embarked** is the port of embarkation, with C = Cherbourg, Q = Queenstown, S = Southampton

# One hypothesis is that, if age data was added to after the sinking, then ages have been estimated more when people didn't survive. The following lines manipulate the Age values so that the value to the left of the decimal point is stripped and value to the right of the decimal point is multiplied by 10. We then print out the rows in the training set that equal five, ie those values for age in the format 'y.5'

# In[ ]:


x = ((train.Age - train.Age.round())*10)
print(train.loc[x == 5])


# In[ ]:


x = ((test.Age - test.Age.round())*10)
print(test.loc[x > 1])


# So, that hypothesis turns out to be invalid as an estimated age correlates with the passenger being in 3rd class..

# Next, we describe the datasets. For each column there is a count of the non-null values and their mean and standard deviation. Miniumum and maximum values are also given as well as 25%, 50% and 75% percentiles.
# First, we describe the training set

# In[ ]:


print(train.describe())


# Next, we describe the test set.

# In[ ]:


print(test.describe())


# Roughly 38% of passengers in the training set survived. It looks as though there were more people in the lower classes than first class, which seems reasonable. The average age is about 30 with a standard deviation of 14/15 in both sets. Family groupings exist with at least one outlier family of considerable size. There is a large standard deviation in Fare, that is probably attributable to first class passengers paying a lot more than those in third class. 

# We are very interested in correlation between features, not least between individual features and "Survived". Here we plot a heatmap of the correlation between features.

# In[ ]:


corr=train.corr()#["Survived"]
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# The largest positive correlation with Survived is with Fare, while the largest negative correlation is with Pclass. There is a strong negative correlation between Fare and Pclass as well.<br><br>
# I'm reading this as there being a higher fare paid for Class = 1 (First Class) than lower classes and that you were more likely to survive if you were in a "better" class and paid more for your ticket (which are themselves, heavily correlated).<br><br>
# SibSp and Parch are strongly correlated, indicating a significant number of family groups.<br><br>
# Sex is not included in the correlation heatmap as it is not a numerical field. But, the [code of conduct, "women and children first"](https://en.wikipedia.org/wiki/Women_and_children_first) mandated that *"the lives of women and children were to be saved first in a life-threatening situation, typically abandoning ship, when survival resources such as lifeboats were limited"*<br><br>
# To test this, first we can look at the number of Men and Women who survived by passenger class.

# In[ ]:


sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=train, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class');


# That's pretty stark. Being a woman in first class gave you a strong chance of survival whereas a man in third class had less than a 1 in 5 chance of surviving the wreck. This confirms that both Pclass and Sex influence a person's chance of survival.<br>
# Next, we plot a barchart to see the proportion of men and women who survived based on where they embarked.

# In[ ]:


sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Embarked",
                    data=train, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Port of Embarkation');


# This isn't as clear. There are differences in survival rate based on where a person embarked. However, a man had the worst chance if he embarked at Queensbury, whereas for women it was Southampton.<br><br>
# There are possible reasons for this, such as low sample sizes and/or people embarking at certain ports tending to travel in a particular class. I'm going to press on without investigating this, although I will include Embarked in the model.<br><br>
# Next, we look at survival rates based on age.
# 

# In[ ]:


figure = plt.figure(figsize=(15,8))
withAge = (train[train['Age'].notnull()])
plt.hist([withAge[withAge['Survived']==1]['Age'], withAge[withAge['Survived']==0]['Age']], stacked=True, color = ['g','r'],bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# Here it looks as though children under the age of ten had a better survival rate than other passengers. This supports the theory that the "Women and children first" code of conduct was upheld during the sinking.<br><br>
# Now, we look at the null values. For both train and test there are a significant number of null values for Age and Cabin

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# There are a significant number of Age and Cabin values missing in each set. One hypothesis is that age and cabin values are null more when people didn't survive. The following lines separate out rows where an age exists and those where they don't as DataFrames and then compares their survival rate by sex.

# In[ ]:


ageless = (train[train['Age'].isnull()])
withAge = (train[train['Age'].notnull()])
print ("Full training set")
print (train[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
print ("Training set with an age")
print (withAge[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
print ("Training set without an age")
print (ageless[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())


# This looks promising. There is a distinct difference between survival rates depending on  gender and whether an age value is present or not. Our original hypothesis that no age indicates a lower survival rate is true for men, but not for women. <br><br>
# Later, we can create a new column, EstimatedAge, and set this as 1 where an age value exists and 0 where it is NaN.

# In[ ]:


cabinless = (train[train['Cabin'].isnull()])
withCabin = (train[train['Cabin'].notnull()])
print ("Full training set")
print (train[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
print ("Training set with a cabin")
print (withCabin[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
print ("Training set without a cabin")
print (cabinless[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())


# This also looks promising. There is a distinct difference between survival rates depending on gender and whether a cabin value is present or not across all men and women in third class. Later, we can create a new column, EstimatedCabin, and set this as 1 where an age value exists and 0 where it is NaN.

# That is all the work we are going to do on the training and test datasets separately. Now we combine them together as one DataFrame, full_data, that we will complete the data and perform feature engineering on.<br><br> First, separate out and then drop the 'Survived' column from the train Dataframe

# In[ ]:


targets = train.Survived
train.drop('Survived', 1, inplace=True)


# Merge the training and test sets so we can perform feature engineering on them together.<br>
# full_data = [train,test] was initially tried, but didn't create the desired DataFrame.<br>
# 

# In[ ]:


full_data = train.append(test)
full_data.reset_index(inplace=True)
full_data.drop('index', inplace=True, axis=1)


# Check that target and full_data look ok.

# In[ ]:


print(full_data.shape)
print(targets.shape)


# # 3. Complete missing data

# Now that we have combined our training and test sets we can create the EstimatedAge and EstimatedCabin columns. They are populated with 0s and 1s depending on whether there are nulls in the Age and Cabin values for each row.

# In[ ]:


full_data['EstimatedAge']=full_data.Age.isnull().astype(int)
full_data['EstimatedCabin']=full_data.Cabin.isnull().astype(int)


# Now, take a look at the full_data DataFrame for missing values.

# In[ ]:


full_data.isnull().sum()


# So, we are going to need to either estimate the missing values from the Age, Cabin, Embarked and Fare columns or drop the column altogether. Let's look at Fare and Embarked first. They only have a small number of missing values and so we won't spend much time on them.

# In[ ]:


print(full_data[full_data['Fare'].isnull()])


# There is one missing Fare value. It has come from the test DataFrame and it is for a man in third class. We are going to calculate the median value of the fare paid by a male passenger for a 3rd class ticket in the test DataFrame and use that to populate this Fare value.

# In[ ]:


grouped_test = full_data.iloc[891:].groupby(['Sex','Pclass'])
grouped_median_test = grouped_test.median()

print(grouped_median_test)


# The average paid for a male in 3rd class in the test datset was 7.895 so we use that for the one missing value.

# In[ ]:


full_data.Fare.fillna(7.895, inplace=True)
full_data.Fare = full_data.Fare.astype(int)


# ## Embarked<br>
# There are only two rows where embarked is a null value. First we take a look at those rows.

# In[ ]:


print(train[train['Embarked'].isnull()])


# It looks as though they were travelling together, as they share a ticket number and a first class cabin. Both of these passengers were female and they survived. It doesn't matter much where we estimate their port of embarkation because being female in first class is so strongly correlated with survival. We'll estimate that they got on at Cherbourg, 'C' as women who embarked there had the strongest chance of survival.

# In[ ]:


full_data["Embarked"] = full_data["Embarked"].fillna('C')


# Now, we are going to separate out the Embarked column, which has possible values of 'C', 'Q' or 'S', into three columns Embarked_C, Embarked_Q and Embarked_S which have boolean values encoded by 0 and 1. Once that is done we drop the Embarked column.

# In[ ]:


embarked_dummies = pd.get_dummies(full_data['Embarked'],prefix='Embarked')
full_data = pd.concat([full_data,embarked_dummies],axis=1)
full_data = full_data.drop(["Embarked"], axis=1)


# See below how the new columns have been created.

# In[ ]:


full_data.info()


# ### Ages
# I have seen a number of methods for estimating the missing age values<br><br>
# 1. Generate random values based on the mean and standard deviation across the whole of the training and test datasets<br>
# 2. Generate random values based on the mean and standard deviation across segments of the training and test datasets, slicing for Pclass and sex, for instance.<br>
# 3. Use a machine learning model to estimate the missing age values.<br><br>
# 
# I have chosen to use a machine learning model, [based on this kernel](https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline) that I looked at from [Poonam Ligade](https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline).

# Before we start we need to categorise Sex using integer values. This is because we will be using a Random Forest categoriser which requires numerical values.

# In[ ]:


full_data['Sex'] = full_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# We also need to do some work on the Name column. We will extract the title from each name into a new Title column and then use the get_dummies function to separate this out into indicator variables.

# In[ ]:


full_data['Title'] = full_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


# Next we categorise the titles. 'Dr' and 'Rev' are a bit of a fudge here.

# In[ ]:


Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Nobility",
                    "Don":        "Nobility",
                    "Sir" :       "Nobility",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "Countess":   "Nobility",
                    "Dona":       "Nobility",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Nobility"
                    }
full_data['Title'] = full_data.Title.map(Title_Dictionary)


# Now we can drop the Name variable. All we are going to be using is the title. 

# In[ ]:


full_data.drop('Name',axis=1,inplace=True)


# Finally we use get_dummies to separate out the Title column into indicator variables

# In[ ]:


titles_dummies = pd.get_dummies(full_data['Title'],prefix='Title')
full_data = pd.concat([full_data,titles_dummies],axis=1)


# Now we take a look at the first five rows.

# In[ ]:


print(full_data.head())


# Now we have columns for age, sex, title and Pclass all in a number format. This means that we can now estimate the missing age values using the Random Forest Classifier.<br><br>
# First we create an age_train DataFrame consisting of the relevant columns from the training set.
# 

# In[ ]:


age_train = full_data.head(891)[['Age','Sex', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Nobility', 'Title_Officer', 'Pclass']]
print(age_train.info())


# Split this into known and unknown sets according to whether the Age value in the row is null or not.

# In[ ]:


known_train  = age_train.loc[ (age_train.Age.notnull()) ]# known Age values for training set
unknown_train = age_train.loc[ (age_train.Age.isnull()) ]# unknown values for training set


# All age values are stored in y_train. All the other values are stored in the X_train feature array

# In[ ]:


y_train = known_train.values[:, 0]
X_train = known_train.values[:, 1::]


# Create and fit a model. Use the fitted model to predict the missing values

# In[ ]:


rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rtr.fit(X_train, y_train)
predictedAges = rtr.predict(unknown_train.values[:, 1::])


# Assign those predictions to the full data set and check that all of the Age values in the training set are now not null.

# In[ ]:


dup_train = full_data.head(891)
dup_train.loc[ (dup_train.Age.isnull()), 'Age' ] = predictedAges 
print(dup_train[dup_train['Age'].isnull()])


# Repeat the process for the test datset

# In[ ]:


age_test = full_data.iloc[891:][['Age','Sex', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Nobility', 'Title_Officer', 'Pclass']]

# Split sets into known and unknown test
known_test  = age_test.loc[ (age_test.Age.notnull()) ]# known Age values for testing set
unknown_test = age_test.loc[ (age_test.Age.isnull()) ]# unknown values for testing set
   
# All age values are stored in target arrays
y_test = known_test.values[:, 0]
    
# All the other values are stored in the feature array
X_test = known_test.values[:, 1::]
    
# Create and fit a model
rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
rtr.fit(X_test, y_test)
    
# Use the fitted model to predict the missing values
predictedAgesTest = rtr.predict(unknown_test.values[:, 1::])

# Assign those predictions to the full data set
dup_test = full_data.iloc[891:]
dup_test.loc[ (dup_test.Age.isnull()), 'Age' ] = predictedAgesTest 


# Merge the enhanced training and test sets back into our full_data DataFrame.

# In[ ]:


full_data = dup_train.append(dup_test)
full_data.reset_index(inplace=True)
full_data.drop('index', inplace=True, axis=1)
print(full_data.shape)


# Now we can drop the title variable as we won't be needing it any more.

# In[ ]:


full_data.drop('Title',axis=1,inplace=True)


# ## Cabin
# Cabin has a lot of missing values: 1014 in total. This makes it difficult to estimate values for it. It is also hard to see how much additional information it can give us. Let's drop it.

# In[ ]:


full_data.drop('Cabin',axis=1,inplace=True)


# Now, we shouldn't have any null values in our full_data DataFrame. We check that and continue onto the next stage, enhancing the data.

# In[ ]:


full_data.isnull().sum()


# ## 4. Create new features<br>
# We have already seen that Parch and SibSp are strongly correlated. So, we are goinf to create a new feature for total family size and then another one to indicate when a person appears to be travelling alone.<br><br>
# Create a new column for family size. Form this by adding the values from the Parch and SibSp columns together

# In[ ]:


full_data['FamilySize'] = full_data['Parch'] + full_data['SibSp'] + 1


# Create a new column to identify people travelling alone

# In[ ]:


full_data['IsAlone'] = full_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)


# Having created FamilySize and IsAlone we no longer require the SibSp and Parch columns. We are also not going to be making use of the Ticket or PassengerId columns and so we drop all four of these columns now.

# In[ ]:


full_data = full_data.drop(["SibSp"], axis=1)
full_data = full_data.drop(["Parch"], axis=1)
full_data = full_data.drop(["Ticket"], axis=1)


# ## 5. Evaluate a range of models<br><br>
# The following has been taken/adapted from [Jason Brownlee's blog](https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/), which I have found very useful.
# 
# We start with our full_dataset and separate it into the X (train) and test dataframes. Targets is the Survived column from the original training set

# In[ ]:


X = full_data.head(891)
test = full_data.iloc[891:]
Y = targets


# Now we create a validation set, which is to be 20% of our X training dataset and set its test options and evaluation metric.
# 

# In[ ]:


validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
seed = 7
scoring = 'accuracy'


# Now we are going to run our data through a number of models, evaluate their accuracy and standard deviation and output those. The loop goes through each of the classifiers. For each classifier it splits up the available data into ten subsets, referred to as folds, fits the model on nine of those folds and then evaluates it against the remaining tenth fold. It then repeats the process for the other folds in turn. The resulting cross validation score is an average of the ten results.

# In[ ]:


# Spot Check Algorithms
models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('XGBClassifier',  XGBClassifier()))
# evaluate each model in turn
results = []
mnames = []
for mname, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	mnames.append(mname)
	msg = "%s: %f (%f)" % (mname, cv_results.mean(), cv_results.std())
	print(msg)


# Of the models that we have tested, Logistic Regression looks the most promising. We can also evaluate our models using a confusion matrix and its associated classification report.

# In[ ]:


# Make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print("Logistic Regression")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("K Nearest Neighbours")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
predictions = dt.predict(X_validation)
print("Decision Tree Classifier")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print("Gaussain NB")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print("Support Vector Machine")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
predictions = rf.predict(X_validation)
print("Random Forest")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

XGB = XGBClassifier()
XGB.fit(X_train, Y_train)
predictions = XGB.predict(X_validation)
print("XGBoost")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# Logistic Regression hasn't always given the best results here. However, it has given me the best scores when I have made submissions.<br><br>
# On this somewhat less-than-scientific basis I have chosen to submit a file based on the Logistic Regression classifier, as shown below. First, we prepare the training and test sets, before running the Logistic Regression classifier to create Y_pred, a single column file containing our predictions.

# In[ ]:


X_train = X.drop(["PassengerId"], axis=1)
Y_train = Y
X_test  = test.drop(["PassengerId"], axis=1)
print(X_train.shape, Y_train.shape, X_test.shape)

# LR = LogisticRegression()
# LR.fit(X_train, Y_train)
# Y_pred = LR.predict(X_test)
# LR.score(X_train, Y_train)
# acc_LR = round(LR.score(X_train, Y_train) * 100, 2)
# print(acc_LR)

from xgboost import XGBRegressor

my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)
my_model.fit(X_train, Y_train)

Y_pred = my_model.predict(X_test)
# We will look at the predicted survival to ensure we have something sensible.
print(Y_pred)


# Finally, we create our submission file by merging Y_pred with the PassengerId column from our test DataFrame and creating a csv file called submission.csv with it.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
# print(submission)
submission.to_csv('submission.csv', index=False)


# While there are many faults that I can see in the above, I have done the things that I set out to at the start of entering this competition, which were:
# 
# 1. Put what I had learned on Andrew Ng's[ Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) course on Coursera into practice
# 2. Develop a workflow through a machine learning project
# 3. Learn from how other people have tackled this challenge
# 4. Come in the top half of the competition.<br>
# 
# So, overall, I am happy with my progress to date. My best submission was 0.78947 which put me about two thirds of the way up the leaderboard at the time I submitted.
# 
# I have a lot more to learn and welcome comments about where I have gone wrong or improvements that I could have made to my submission. I hope that the way this is presented is clear, especially for people who are starting out, and if you think I can help, please feel free to ask questions.
