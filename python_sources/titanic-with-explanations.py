#!/usr/bin/env python
# coding: utf-8

# ## First competition for beginners(with explanations)
# 
# ### In this competition, we have a data set of different information about passengers onboard the Titanic, and we see if we can use that information to predict whether those people survived or not.
# 
# Each Kaggle competition has two key data files that you will work with - a training set and a testing set.
# 
# The training set contains data we can use to train our model. It has a number of feature columns which contain various descriptive data, as well as a column of the target values we are trying to predict: in this case, Survival.
# 
# The testing set contains all of the same feature columns, but is missing the target value column. Additionally, the testing set usually has fewer observations (rows) than the training set.
# 
# This is useful because we want as much data as we can to train our model on. Once we have trained our model on the training set, we will use that model to make predictions on the data from the testing set, and submit those predictions to Kaggle.
# 
# 

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


# We'll start by using the pandas library to read both files and inspect their size.

# In[ ]:


test = pd.read_csv("../input/titanic/test.csv")
test_shape = test.shape

train = pd.read_csv("../input/titanic/train.csv")
train_shape = train.shape

print(test_shape)
print(train_shape)


# The type of machine learning we will be doing is called classification, because when we make predictions we are classifying each passenger as survived or not. More specifically, we are performing binary classification, which means that there are only two different states we are classifying.
# 
# In any machine learning exercise, thinking about the topic you are predicting is very important. I call this step acquiring domain knowledge, and it's one of the most important determinants for success in machine learning.
# 
# In this case, understanding the Titanic disaster and specifically what variables might affect the outcome of survival is important. Anyone who has watched the movie Titanic would remember that women and children were given preference to lifeboats (as they were in real life). You would also remember the vast class disparity of the passengers.
# 
# This indicates that **Age**, **Sex**, and **PClass** may be good predictors of survival. We'll start by exploring Sex and Pclass by visualizing the data.
# 
# Because the **Survived** column contains **0** if the passenger did not survive and **1** if they did, we can segment our data by sex and calculate the mean of this column. We can use **DataFrame.pivot_table()** to easily do this:

# In[ ]:


import matplotlib.pyplot as plt

sex_pivot = train.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.show()

class_pivot = train.pivot_table(index="Pclass", values="Survived")
class_pivot.plot.bar()
plt.show()


# The **Sex** and **PClass** columns are what we call **categorical** features. That means that the values represented a few separate options (for instance, whether the passenger was **male** or **female**).
# 
# Let's take a look at the **Age** column using **Series.describe()**.

# In[ ]:


print(train["Age"].describe())


# The **Age** column contains numbers ranging from **0.42** to **80.0** (If you look at Kaggle's data page, it informs us that Age is fractional if the passenger is less than one). The other thing to note here is that there are **714** values in this column, fewer than the **891** rows we discovered that the train data set had earlier in this mission which indicates we have some **missing values**.
# 
# All of this means that the **Age** column needs to be treated slightly differently, as this is a continuous numerical column. One way to look at distribution of values in a continuous numerical set is to use histograms. We can create two histograms to compare visually the those that survived vs those who died across different age ranges:

# In[ ]:


survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# The relationship here is not simple, but we can see that in some age ranges more passengers survived - where the red bars are higher than the blue bars.
# 
# In order for this to be useful to our machine learning model, we can separate this continuous feature into a categorical feature by dividing it into ranges. We can use the **pandas.cut()** function to help us out.
# 
# The **pandas.cut()** function has two required parameters - the column we wish to cut, and a list of numbers which define the boundaries of our cuts. We are also going to use the optional parameter **labels**, which takes a list of labels for the resultant bins. This will make it easier for us to understand our results.
# 
# Before we modify this column, we have to be aware of two things. Firstly, any change we make to the **train** data, we also need to make to the **test** data, otherwise we will be unable to use our model to make predictions for our submissions. Secondly, we need to remember to handle the **missing values** we observed above.
# 
# We split the **Age** column into six categories:
# 
# * **Missing**, from -1 to 0
# * **Infant**, from 0 to 5
# * **Child**, from 5 to 12
# * **Teenager**, from 12 to 18
# * **Young Adult**, from 18 to 35
# * **Adult**, from 35 to 60
# * **Senior**, from 60 to 100
# 
# Note that the cut_points list has one more element than the label_names list, since it needs to define the upper boundary for the last segment.
# 

# In[ ]:


def process_age(df, cut_points, label_names):
    df['Age'] = df['Age'].fillna(-0.5)
    df['Age_categories'] = pd.cut(df['Age'], cut_points, labels=label_names)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

age_pivot = train.pivot_table(index='Age_categories', values='Survived')
age_pivot.plot.bar()
plt.show()


# So far we have identified three columns that may be useful for predicting survival:
# 
# * **Sex**
# * **Pclass**
# * **Age**, or more specifically our newly created **Age_categories**
# 
# Before we build our model, we need to prepare these columns for machine learning. Most machine learning algorithms can't understand text labels, so we have to convert our values into numbers.
# 
# Additionally, we need to be careful that we don't imply any numeric relationship where there isn't one. If we think of the values in the **Pclass** column, we know they are **1**, **2**, and **3**.
# 
# While the class of each passenger certainly has some sort of ordered relationship, the relationship between each class is not the same as the relationship between the numbers **1**, **2**, and **3**. For instance, class **2** isn't "worth" double what class **1** is, and class **3** isn't "worth" triple what class **1** is.
# 
# In order to remove this relationship, we can create dummy columns for each unique value in Pclass.
# 
# We can use the **[pandas.get_dummies()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)** function, which will generate dummy columns.
# 

# In[ ]:


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

cols = ['Pclass', 'Sex', 'Age_categories']

for col in cols:
    train = create_dummies(train, col)
    test = create_dummies(test, col)
    


# In[ ]:


train.head(1)


# Now that our data has been prepared, we are ready to train our first model. The first model we will use is called Logistic Regression, which is often the first model you will train when performing classification.
# 
# We will be using the [scikit-learn](http://scikit-learn.org/stable/index.html) library as it has many tools that make performing machine learning easier. The scikit-learn workflow consists of four main steps:
# 
# * Instantiate (or create) the specific machine learning model you want to use
# * Fit the model to the training data
# * Use the model to make predictions
# * Evaluate the accuracy of the predictions
# Each model in scikit-learn is implemented as a separate class and the first step is to identify the class we want to create an instance of. In our case, we want to use the [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) class.
# 
# We'll start by looking at the first two steps. First, we need to import the class:

# In[ ]:


from sklearn.linear_model import LogisticRegression


# Next, we create a **LogisticRegression** object:

# In[ ]:


lr = LogisticRegression()


# Lastly, we use the [LogisticRegression.fit()](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit) method to train our model. The **.fit()** method accepts two arguments: **X** and **y**. **X** must be a two dimensional array (like a dataframe) of the features that we wish to train our model on, and **y** must be a one-dimensional array (like a series) of our target, or the column we wish to predict.

# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr.fit(train[columns], train['Survived'])


# Congratulations, you've trained your first machine learning model! Our next step is to find out how accurate our model is, and to do that, we'll have to make some predictions.
# 
# If you recall from earlier, we do have a test dataframe that we could use to make predictions. We could make predictions on that data set, but because it doesn't have the **Survived** column we would have to submit it to **Kaggle** to find out our accuracy. This would quickly become a pain if we had to submit to find out the accuracy every time we optimized our model.
# 
# We could also fit and predict on our **train** dataframe, however if we do this there is a high likelihood that our model will **overfit**, which means it will perform well because we're testing on the same data we've trained on, but then perform much worse on new, unseen data.
# 
# Instead we can split our train dataframe into two:
# 
# * One part to train our model on (often 80% of the observations)
# * One part to make predictions with and test our model (often 20% of the observations)
# 
# The convention in machine learning is to call these two parts **train** and **test**. This can become confusing, since we already have our **test** dataframe that we will eventually use to make predictions to submit to Kaggle. To avoid confusion, from here on, we're going to call this Kaggle 'test' data **holdout** data, which is the technical name given to this type of data used for final predictions.
# 
# The scikit-learn library has a handy [model_selection.train_test_split()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function that we can use to split our data. **train_test_split()** accepts two parameters, **X** and **y**, which contain all the data we want to train and test on, and returns four objects: **train_X**, **train_y**, **test_X**, **test_y**:

# In[ ]:


holdout = test

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.20, random_state=0)


# Now that we have our data split into train and test sets, we can fit our model again on our training set, and then use that model to make predictions on our test set.
# 
# Once we have fit our model, we can use the [LogisticRegression.predict()](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict) method to make predictions.
# 
# The **predict()** method takes a single parameter **X**, a two dimensional array of features for the observations we wish to predict. **X** must have the exact same features as the array we used to fit our model. The method returns single dimensional array of predictions.
# 
# There are a number of ways to measure the accuracy of machine learning models, but when competing in Kaggle competitions you want to make sure you use the same method that Kaggle uses to calculate accuracy for that specific competition.
# 
# In this case, [the evaluation section for the Titanic competition on Kaggle](https://www.kaggle.com/c/titanic#evaluation) tells us that our score calculated as "the percentage of passengers correctly predicted". This is by far the most common form of accuracy for binary classification.
# 
# Again, scikit-learn has a handy function we can use to calculate accuracy: [metrics.accuracy_score()](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). The function accepts two parameters, **y_true** and **y_pred**, which are the actual values and our predicted values respectively, and returns our accuracy score.

# In[ ]:


from sklearn.metrics import accuracy_score

lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)

accuracy = accuracy_score(test_y, predictions)

print(accuracy)


# Our model has an accuracy score of 81.0% when tested against our 20% test set. Given that this data set is quite small, there is a good chance that our model is overfitting, and will not perform as well on totally unseen data.
# 
# To give us a better understanding of the real performance of our model, we can use a technique called **cross validation** to train and test our model on different splits of our data, and then average the accuracy scores.
# 
# The most common form of cross validation, and the one we will be using, is called **k-fold** cross validation. 'Fold' refers to each different iteration that we train our model on, and 'k' just refers to the number of folds.
# 
# We will use scikit-learn's [model_selection.cross_val_score()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score) function to automate the process.
# 
# The basic syntax for **cross_val_score()** is:
# 
#     cross_val_score(estimator, X, y, cv=None)
#     
# * **estimator** is a scikit-learn estimator object, like the **LogisticRegression()** objects we have been creating.
# * **X** is all features from our data set.
# * **y** is the target variables.
# * **cv** specifies the number of folds.
# 
# The function returns a numpy ndarray of the accuracy scores of each fold.

# In[ ]:


from sklearn.model_selection import cross_val_score

lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

accuracy = np.mean(scores)

print(scores)
print(accuracy)


# From the results of our k-fold validation, you can see that the accuracy number varies with each fold - ranging between 76.4% and 87.6%. This demonstrates why cross validation is important.
# 
# As it happens, our average accuracy score was 80.2%, which is not far from the 81.0% we got from our simple train/test split, however this will not always be the case, and you should always use cross-validation to make sure the error metrics you are getting from your model are accurate.
# 
# We are now ready to use the model we have built to train our final model and then make predictions on our unseen holdout data, or what Kaggle calls the 'test' data set.

# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr = LogisticRegression()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns])


# The last thing we need to do is create a submission file. Each Kaggle competition can have slightly different requirements for the submission file. Here's what is specified on the [Titanic competition evaluation](https://www.kaggle.com/c/titanic#evaluation) page:
# 
# You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.
# 
# The file should have exactly 2 columns:
# 
# * PassengerId (sorted in any order)
# * Survived (contains your binary predictions: 1 for survived, 0 for deceased)
# 
# We will need to create a new dataframe that contains the **holdout_predictions** we created and the **PassengerId** column from the **holdout** dataframe. We don't need to worry about matching the data up, as both of these remain in their original order.
# 

# In[ ]:


holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)


# In[ ]:


import os
os.chdir(r'/kaggle/working')
submission.to_csv("submission.csv",index=False)


# 

# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')


# ### We're going to focus working with the features used in our model

# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
holdout = pd.read_csv('../input/titanic/test.csv')

def process_age(df):
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = process_age(train)
holdout = process_age(holdout)

columns = ['Age_categories', 'Pclass', 'Sex']

for column in columns:
    train = create_dummies(train, column)
    holdout = create_dummies(holdout, column)
    
print(train.columns)
print(holdout.columns)


# The last nine rows of the output are dummy columns we created, but in the first three rows we can see there are a number of features we haven't yet utilized. We can ignore **PassengerId**, since this is just a column Kaggle have added to identify each passenger and calculate scores. We can also ignore **Survived**, as this is what we're predicting, as well as the three columns we've already used.
# 
# Here is a list of the remaining columns (with a brief description), followed by 10 randomly selected passengers from and their data from those columns, so we can refamiliarize ourselves with the data.
# 
# * **SibSp** - The number of siblings or spouses the passenger had aboard the Titanic
# * **Parch** - The number of parents or children the passenger had aboard the Titanic
# * **Ticket** - The passenger's ticket number
# * **Fare** - The fair the passenger paid
# * **Cabin** - The passengers cabin number
# * **Embarked** - The port where the passenger embarked (C=Cherbourg, Q=Queenstown, S=Southampton)

# In[ ]:


remaining_columns = ['SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
print(train[remaining_columns].head())


# At first glance, both the **Name** and **Ticket** columns look to be unique to each passenger. We will come back to these columns later, but for now we'll focus on the other columns.
# 
# We can use the [Dataframe.describe()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html) method to give us some more information on the values within each remaining column.

# In[ ]:


print(train[remaining_columns].describe(include='all',percentiles=[]))


# Of these, **SibSp**, **Parch** and **Fare** look to be standard numeric columns with no missing values. **Cabin** has values for only **204** of the **891** rows, and even then most of the values are unique, so for now we will leave this column also. **Embarked** looks to be a standard categorical column with 3 unique values, much like **PClass** was, except that there are two missing values. We can easily fill these two missing values with the most common value, "**S**" which occurs **644** times.
# 
# Looking at our numeric columns, we can see a big difference between the range of each. **SibSp** has values between 0-8, **Parch** between 0-6, and **Fare** is on a dramatically different scale, with values ranging from 0-512. In order to make sure these values are equally weighted within our model, we'll need to **rescale** the data.
# 
# Rescaling simply stretches or shrinks the data as needed to be on the same scale, in our case between 0 and 1.
# 
# Within **scikit-learn**, the **preprocessing.minmax_scale()** function allows us to quickly and easily rescale our data:
# 
#     from sklearn.preprocessing import minmax_scale
#     columns = ["column one", "column two"]
#     data[columns] = minmax_scale(data[columns])

# In[ ]:


from sklearn.preprocessing import minmax_scale

holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())

train['Embarked'] = train['Embarked'].fillna('S')
holdout['Embarked'] = holdout['Embarked'].fillna('S')

train = create_dummies(train, 'Embarked')
holdout = create_dummies(holdout, 'Embarked')

columns = ['SibSp', 'Parch', 'Fare']

for col in columns:
    train[col + '_scaled'] = minmax_scale(train[col])
    holdout[col + '_scaled'] = minmax_scale(holdout[col])
    
print(len(train.columns))

columns = ['PassengerId', 'Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled', 'Fare_categories_0-12',
       'Fare_categories_12-50', 'Fare_categories_50-100',
       'Fare_categories_100+', 'Title_Master', 'Title_Miss', 'Title_Mr',
       'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'Cabin_type_A',
       'Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D', 'Cabin_type_E',
       'Cabin_type_F', 'Cabin_type_G', 'Cabin_type_Unknown']
modified_holdout = holdout[columns]


# In order to select the best-performing features, we need a way to measure which of our features are relevant to our outcome - in this case, the survival of each passenger. One effective way is by training a logistic regression model using all of our features, and then looking at the coefficients of each feature.
# 
# The scikit-learn [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) class has an attribute in which coefficients are stored after the model is fit, **LogisticRegression.coef_**. We first need to train our model, after which we can access this attribute.
# 
#     lr = LogisticRegression()
#     lr.fit(train_X,train_y)
#     coefficients = lr.coef_
#     
# The **coef()** method returns a NumPy array of coefficients, in the same order as the features that were used to fit the model. To make these easier to interpret, we can convert the coefficients to a pandas series, adding the column names as the index:
# 
#     feature_importance = pd.Series(coefficients[0],
#                                index=train_X.columns)
#                                
#  

# In[ ]:


columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']

lr = LogisticRegression()

lr.fit(train[columns], train['Survived'])

coefficients = lr.coef_

feature_importance = pd.Series(coefficients[0], index=train[columns].columns)
feature_importance.plot.barh()
plt.show()


# The plot we generated showed a range of both positive and negative values. Whether the value is positive or negative isn't as important in this case, relative to the magnitude of the value. If you think about it, this makes sense. A feature that indicates strongly whether a passenger died is just as useful as a feature that indicates strongly that a passenger survived, given they are mutually exclusive outcomes.
# 
# To make things easier to interpret, I'll alter the plot to show all positive values, and have sorted the bars in order of size:

# In[ ]:


ordered_feature_importance = feature_importance.abs().sort_values()
ordered_feature_importance.plot.barh()
plt.show()


# We'll train a new model with the top 8 scores and check our accuracy using cross validation.

# In[ ]:


from sklearn.model_selection import cross_val_score

columns = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']

lr = LogisticRegression()

all_X = train[columns]
all_y = train['Survived']

scores = cross_val_score(lr, all_X, all_y, cv=10)

accuracy = scores.mean()
print(accuracy)

lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[columns])

holdout_ids = holdout['PassengerId']
submission_df = {"PassengerId": holdout_ids,
                "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)

os.chdir(r'/kaggle/working')
submission.to_csv("submission_2.csv",index=False)


# In[ ]:





# A lot of the gains in accuracy in machine learning come from **Feature Engineering**. Feature engineering is the practice of creating new features from your existing data.
# 
# One common way to engineer a feature is using a technique called **binning**. Binning is when you take a continuous feature, like the fare a passenger paid for their ticket, and separate it out into several ranges (or 'bins'), turning it into a categorical variable.

# In[ ]:


def process_fare(df, cut_points, label_names):
    df['Fare_categories'] = pd.cut(df['Fare'], cut_points, labels=label_names)
    return df

cut_points = [0, 12, 50, 100, 1000]
label_names = ['0-12', '12-50', '50-100', '100+']

train = process_fare(train, cut_points, label_names)
holdout = process_fare(holdout, cut_points, label_names)

train = create_dummies(train, "Fare_categories")
holdout = create_dummies(holdout, "Fare_categories")


# While in isolation the cabin number of each passenger will be reasonably unique to each, we can see that the format of the cabin numbers is one letter followed by two numbers. It seems like the letter is representative of the type of cabin, which could be useful data for us.
# 
# Looking at the Name column, There is a title like 'Mr' or 'Mrs' within each, as well as some less common titles, like the 'Countess' from the final row of our table above. By spending some time researching the different titles, we can categorize these into six types:
# 
# * Mr
# * Mrs
# * Master
# * Miss
# * Officer
# * Royalty
# 
# We can use the [Series.str.extract](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.str.extract.html) method and a **regular expression** to extract the title from each name and then use the [Series.map()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html) method and a predefined dictionary to simplify the titles.
# 

# In[ ]:


titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}

extracted_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
train["Title"] = extracted_titles.map(titles)

extracted_titles = holdout['Name'].str.extract('([A-Za-z]+)\.', expand=False)
holdout['Title'] = extracted_titles.map(titles)

train['Cabin_type'] = train['Cabin'].str[0]
train['Cabin_type'] = train['Cabin_type'].fillna('Unknown')

holdout['Cabin_type'] = holdout['Cabin'].str[0]
holdout['Cabin_type'] = holdout['Cabin_type'].fillna('Unknown')

for column in ['Title', 'Cabin_type']:
    train = create_dummies(train, column)
    holdout = create_dummies(holdout, column)
    


# We now have 34 possible feature columns we can use to train our model. One thing to be aware of as you start to add more features is a concept called **collinearity**. Collinearity occurs where more than one feature contains data that are similar.
# 
# The effect of collinearity is that your model will overfit - you may get great results on your test data set, but then the model performs worse on unseen data (like the holdout set).
# 
# One easy way to understand collinearity is with a simple binary variable like the **Sex** column in our dataset. Every passenger in our data is categorized as either male or female, so 'not male' is exactly the same as 'female'.
# 
# As a result, when we created our two dummy columns from the categorical **Sex** column, we've actually created two columns with identical data in them. This will happen whenever we create dummy columns, and is called the dummy variable trap. The easy solution is to choose one column to drop any time you make dummy columns.

# In[ ]:


import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_categories_0-12',
       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',
       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',
       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown']

plot_correlation_heatmap(train[columns])


# We can see that there is a high correlation between **Sex_female**/**Sex_male** and **Title_Miss**/**Title_Mr**/**Title_Mrs**. We will remove the columns **Sex_female** and **Sex_male** since the title data may be more nuanced.
# 
# Apart from that, we should remove one of each of our dummy variables to reduce the collinearity in each. We'll remove:
# 
# * Pclass_2
# * Age_categories_Teenager
# * Fare_categories_12-50
# * Title_Master
# * Cabin_type_A
# 
# In an earlier step, we manually used the logit coefficients to select the most relevant features. An alternate method is to use one of scikit-learn's inbuilt feature selection classes. We will be using the [feature_selection.RFECV](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) class which performs **recursive feature elimination**
# 
# The RFECV class starts by training a model using all of your features and scores it using cross validation. It then uses the logit coefficients to eliminate the least important feature, and trains and scores a new model. At the end, the class looks at all the scores, and selects the set of features which scored highest.
# 
# Like the **LogisticRegression** class, **RFECV** must first be instantiated and then fit. The first parameter when creating the **RFECV** object must be an estimator, and we need to use the **cv** parameter to specific the number of folds for cross-validation.
# 
#     from sklearn.feature_selection import RFECV
#     lr = LogisticRegression()
#     selector = RFECV(lr,cv=10)
#     selector.fit(all_X,all_y)
#     
# Once the **RFECV** object has been fit, we can use the **RFECV.support_** attribute to access a boolean mask of **True** and **False** values which we can use to generate a list of optimized columns:
# 
#     optimized_columns = all_X.columns[selector.support_]
# 

# In[ ]:


from sklearn.feature_selection import RFECV

columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Young Adult',
       'Age_categories_Adult', 'Age_categories_Senior', 'Pclass_1', 'Pclass_3',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_categories_0-12', 'Fare_categories_50-100',
       'Fare_categories_100+', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C',
       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
       'Cabin_type_T', 'Cabin_type_Unknown']

all_X = train[columns]
all_y = train["Survived"]

lr = LogisticRegression()
selector = RFECV(lr, cv=10)
selector.fit(all_X, all_y)

optimized_columns = all_X.columns[selector.support_]


# The **RFECV()** selector returned only four columns:
# 
#     ['SibSp_scaled', 'Title_Mr', 'Title_Officer', 'Cabin_type_Unknown']
#     
# Let's train a model using cross validation using these columns and check the score.

# In[ ]:


all_X = train[optimized_columns]
all_y = train["Survived"]

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
accuracy = scores.mean()
print(accuracy)


# Let's train these columns on the holdout set, save a submission file and see what score we get from Kaggle.

# In[ ]:


lr = LogisticRegression()
lr.fit(all_X, all_y)
holdout_predictions = lr.predict(holdout[optimized_columns])

holdout_ids = holdout['PassengerId']
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

os.chdir(r'/kaggle/working')
submission.to_csv("submission_3.csv", index=False)


# In[ ]:




