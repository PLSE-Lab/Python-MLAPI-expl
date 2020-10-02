#!/usr/bin/env python
# coding: utf-8

# ##Section 1-0 - First Cut

# We will start by processing the training data, after which we will be able to use to 'train' (or 'fit') our model. With the trained model, we apply it to the test data to make the predictions. Finally, we output our predictions into a .csv file to make a submission to Kaggle and see how well they perform.
# 
# It is very common to encounter missing values in a data set. In this section, we will take the simplest (or perhaps, simplistic) approach of ignoring the whole row if any part of it contains an NaN value. We will build on this approach in later sections.

# ###Pandas - Extracting data

# First, we load the training data from a .csv file. This is the similar to the data found on the Kaggle website:
# 
# https://www.kaggle.com/c/titanic-gettingStarted/data
# 

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('../input/train.csv')


# ###Pandas - Cleaning data

# We then review a selection of the data.

# In[ ]:


df.head(10)


# We notice that the columns describe features of the Titanic passengers, such as age, sex, and class. Of particular interest is the column Survived, which describes whether or not the passenger survived. When training our model, what we are essentially doing is assessing how each feature impacts whether or not the passenger survived (or if the feature makes an impact at all).

# **Exercise:**
# 
# * Write the code to review the tail-end section of the data.

# We observe that the columns Name, Ticket and Cabin are, for our current purposes, irrelevant. We proceed to remove them from our data set.

# In[ ]:


df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# Next, we review the type of data in the columns, and their respective counts.

# In[ ]:


df.info()


# We notice that the columns Age and Embarked have NaNs or missing values. As previously discussed, we take the approach of simply removing the rows with missing values.

# In[ ]:


df = df.dropna()


# **Question**
# 
# * If you were to fill in the missing values, with what values would you fill them with? Why?

# Scikit-learn only takes numerical arrays as inputs. As such, we would need to convert the categorical columns Sex and Embarked into numerical ones. We first review the range of values for the column Sex, and create a new column that represents the data as numbers.

# In[ ]:


df['Sex'].unique()


# In[ ]:


df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)


# Similarly for Embarked, we review the range of values and create a new column called Port that represents, as a numerical value, where each passenger embarks from.

# In[ ]:


df['Embarked'].unique()


# In[ ]:


df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)


# **Question**
# 
# * What problems might we encounter by mapping C, S, and Q in the column Embarked to the values 1, 2, and 3? In other words, what does the ordering imply? Does the same problem exist for the column Sex?

# Now that we have numerical columns that encapsulate the information provided by the columns Sex and Embarked, we can proceed to drop them from our data set.

# In[ ]:


df = df.drop(['Sex', 'Embarked'], axis=1)


# We review the columns our final, processed data set.

# In[ ]:


cols = df.columns.tolist()
print(cols)


# For convenience, we move the column Survived to the left-most column. We note that the left-most column is indexed as 0.

# In[ ]:


cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]


# In our final review of our training data, we check that (1) the column Survived is the left-most column (2) there are no NaN values, and (3) all the values are in numerical form.

# In[ ]:


df.head(10)


# In[ ]:


df.info()


# Finally, we convert the processed training data from a Pandas dataframe into a numerical (Numpy) array.

# In[ ]:


train_data = df.values


# ###Scikit-learn - Training the model

# In this section, we'll simply use the model as a black box. We'll review more sophisticated techniques in later sections.

# Here we'll be using the Random Forest model. The intuition is as follows: each feature is reviewed to see how much impact it makes to the outcome. The most prominent feature is segmented into a 'branch'. A collection of branches is a 'tree'. The Random Forest model, broadly speaking, creates a 'forest' of trees and aggregates the results.
# 
# http://en.wikipedia.org/wiki/Random_forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)


# We use the processed training data to 'train' (or 'fit') our model. The column Survived will be our second input, and the set of other features (with the column PassengerId omitted) as the first.

# In[ ]:


model = model.fit(train_data[0:,2:], train_data[0:,0])


# ###Scikit-learn - Making predictions

# We first load the test data.

# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# We then review a selection of the data.

# In[ ]:


df_test.head(10)


# We notice that test data has columns similar to our training data, but not the column Survived. We'll use our trained model to predict values for the column Survived.

# As before, we process the test data in a similar fashion to what we did to the training data.

# In[ ]:


df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test = df_test.dropna()

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values


# We now apply the trained model to the test data (omitting the column PassengerId) to produce an output of predictions.

# In[ ]:


output = model.predict(test_data[:,1:])


# ###Pandas - Preparing for submission

# In[ ]:


result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])


# We briefly review our predictions.

# In[ ]:


df_result.head(10)


# Finally, we output our results to a .csv file.

# In[ ]:


# df_result.to_csv('../results/titanic_1-0.csv', index=False)


# However, it appears that we have a problem. The Kaggle submission website expects "the solution file to have 418 predictions."
# 
# https://www.kaggle.com/c/titanic-gettingStarted/submissions/attach

# We compare this to our result.

# In[ ]:


df_result.shape


# Since we eliminated the rows containing NaNs, we end up with a set of predictions with a smaller number of rows compared to the test data. As Kaggle requires all 418 predictions, we are unable to make a submission.

# In this section, we took the simplest approach of ignoring missing values, but fail to produce a complete set of predictions. We look to build on this approach in Section 1-1.

# ##Section 1-1 - Filling-in Missing Values
# 

# In the previous section, we ended up with a smaller set of predictions because we chose to throw away rows with missing values. We build on this approach in this section by filling in the missing data with an educated guess.

# We will only provide detailed descriptions on new concepts introduced.

# ###Pandas - Extracting data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../input/train.csv')


# ###Pandas - Cleaning data

# In[ ]:


df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# Similar to the previous section, we review the data type and value counts.

# In[ ]:


df.info()


# There are a number of ways that we could fill in the NaN values of the column Age. For simplicity, we'll do so by taking the average, or mean, of values of each column. We'll review as to whether taking the median would be a better choice in a later section.
# 

# In[ ]:


age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)


# **Exercise**
# 
# * Write the code to replace the NaN values by the median, instead of the mean.

# Taking the average does not make sense for the column Embarked, as it is a categorical value. Instead, we shall replace the NaN values by the mode, or most frequently occurring value.

# In[ ]:


mode_embarked = df['Embarked'].mode().values[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)


# In[ ]:


df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]


# We now review details of our training data.

# In[ ]:


df.info()


# Hence have we have preserved all the rows of our data set, and proceed to create a numerical array for Scikit-learn.

# In[ ]:


train_data = df.values


# ###Scikit-learn - Training the model
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])


# ###Scikit-learn - Making predictions

# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# We now review what needs to be cleaned in the test data.

# In[ ]:


df_test.info()


# In[ ]:


df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# As per our previous approach, we fill in the NaN values in the column Age with the mean.

# In[ ]:


df_test['Age'] = df_test['Age'].fillna(age_mean)


# For the column Fare, however, it makes sense to fill in the NaN values with the mean by the column Pclass, or Passenger class.

# In[ ]:


fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')


# In[ ]:


fare_means


# Here we created a pivot table by calculating the mean of the column Fare by each Pclass, which we will use to fill in our NaN values.

# In[ ]:


df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)


# This is one of the more complicated lines of code we'll encounter, so let's unpack this.

# First, we look at each of the pairs (Fare, Pclass) (i.e. lambda x). From this pair, we check if the Fare part is NaN (i.e. if pd.isnull(x['Fare'])). If Fare is NaN, we look at the Pclass value of that pair (i.e. x['PClass']), and replace the NaN value the mean fare of that class (i.e. fare_means[x['Pclass']]). If Fare is not NaN, then we keep it the same (i.e. else x['Fare']).

# In[ ]:


df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values

output = model.predict(test_data[:,1:])


# ###Pandas - Preparing for submission

# In[ ]:


result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

# df_result.to_csv('../results/titanic_1-1.csv', index=False)


# Now we'll discover that our submission has 418 predictions, and can proceed to make our first leaderboard entry.
# 
# https://www.kaggle.com/c/titanic-gettingStarted/submissions/attach

# In[ ]:


df_result.shape


# Congratulations on making your first Kaggle submission!!

# ##Section 1-2 - Creating Dummy Variables
# 

# In previous sections, we replaced the categorical values {C, S, Q} in the column Embarked by the numerical values {1, 2, 3}. The latter, however, has a notion of ordering not present in the former (which is simply arranged in alphabetical order). To get around this problem, we shall introduce the concept of dummy variables.

# ###Pandas - Extracting data

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('../input/train.csv')


# ###Pandas - Cleaning data

# In[ ]:


df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)

mode_embarked = df['Embarked'].mode().values[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)


# As there are only two unique values for the column Sex, we have no problems of ordering.

# In[ ]:


df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)


# For the column Embarked, however, replacing {C, S, Q} by {1, 2, 3} would seem to imply the ordering C < S < Q when in fact they are simply arranged alphabetically.

# To avoid this problem, we create dummy variables. Essentially this involves creating new columns to represent whether the passenger embarked at C with the value 1 if true, 0 otherwise. Pandas has a built-in function to create these columns automatically.

# In[ ]:


pd.get_dummies(df['Embarked'], prefix='Embarked').head(10)


# We now concatenate the columns containing the dummy variables to our main dataframe.

# In[ ]:


df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)


# **Exercise**
# 
# * Write the code to create dummy variables for the column Sex.

# In[ ]:


df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]


# In[ ]:


We review our processed training data.


# In[ ]:


df.head()


# In[ ]:


train_data = df.values


# In[ ]:


###Scikit-learn - Training the model


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])


# In[ ]:


Scikit-learn - Making predictions


# In[ ]:


df_test = pd.read_csv('../input/test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)

fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:
                            fare_means[x['Pclass']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)


# In[ ]:


Similarly we create dummy variables for the test data.


# In[ ]:


df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],
                axis=1)


# In[ ]:


df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values

output = model.predict(test_data[:,1:])


# In[ ]:


###Pandas - Preparing for submission


# In[ ]:


result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
# df_result.to_csv('../results/titanic_1-2.csv', index=False)

