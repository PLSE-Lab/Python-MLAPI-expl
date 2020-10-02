#!/usr/bin/env python
# coding: utf-8

# ### Inspired by [Basic Approach for Top 3% in the Titanic](https://www.kaggle.com/danielv7/basic-approach-for-top-3-in-the-titanic)
# 
# ### Feature Importance Visualization Inspired by [Feature Engineering Tutorial with Titanic](https://www.kaggle.com/gunesevitan/feature-engineering-tutorial-with-titanic)
# 
# #### Feel free to comment below and upvote if you find this solution helpful to you in anyway :)

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # data processing

from sklearn.model_selection import train_test_split # train_test_split
from sklearn.ensemble import RandomForestClassifier # classifier model
from sklearn.metrics import classification_report # model performance analysis
from sklearn.model_selection import cross_val_score # model performance analysis

import seaborn as sns # data visualization


# ## Import Data
# Read input data then merge them

# In[ ]:


training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# merge the two for convenience in data preprocessing
data = pd.concat([training_data, test_data], sort=False)

# Take a quick look at it
data.head()


# ## Preprocess Data
# 1. Remove columns that has no effect on prediction of survival. (e.g. : `PassengerId` and `Ticket`)
# 2. Filling in missing values with statistical approach.
# 3. Convert categorical column using one-hot encoding. (e.g. : convert `male` -> 1 and `female` -> 0 in `Sex` column)
# 4. Normalize variables with skewed distributions.

# In[ ]:


# STEP 1
# Remove unrelated columns
data.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)

# STEP 2
# Before we start, let's take a look at where these missing values are
def check_missing_values(data):
    for col in data:
        num_of_na = data[col].isna().sum()
        # Test data does not have 'Survived' column
        if num_of_na > 0 and col != 'Survived': 
            print('Found', num_of_na, 'missing values in column', col)
            percentage = num_of_na / 1309 * 100    # 1309 instances in total
            print('That\'s', '%.2f' % percentage + '% missing.\n')
            
check_missing_values(data)


# ### Drop `Cabin` since it's over 75% incomplete

# In[ ]:


# Remove column 'Cabin' from data
data.drop('Cabin', axis=1, inplace=True)


# ### Explore correlations before filling missing values

# In[ ]:


correlations = data.copy().corr().abs().unstack().sort_values()
correlations = correlations['Age'].drop('Age')
correlations_chart = correlations.plot(kind='barh', title='Correlation With Other Attributes')


# In[ ]:


# Relationship with passenger class
for pclass in range(1, 4):
    analysis = data[data['Pclass'] == pclass]
    mean = analysis['Age'].mean()
    print('Passengers in class', pclass, 'are on average', round(mean), 'y/o')

# Relationship with number of siblings and spouse
for sibsp in np.sort(data['SibSp'].unique()):
    analysis = data[data['SibSp'] == sibsp]    
    mean = analysis['Age'].mean()
    
    if np.isnan(mean): 
        print('Average age of passengers with', sibsp, 'siblings and spouse are unknown')
    else:
        print('Passengers with', sibsp, 'siblings and spouse are on average', round(mean), 'y/o')


# ### >2 sibling and spouse indicate possible children
# Therefore we fill missing `Age` by the following logic

# In[ ]:


# if number of sibling and spouse > 2:
    # age = 16/9/10/14 accordingly
# else:
    # age = mean(average(pclass) + average(sibsp))
    
# Impute age based on the above log
def impute_age(columns):
    sibsp_age_map = {0.0: 31, 
                     1.0: 31, 
                     2.0: 24, 
                     3.0: 16, 
                     4.0: 9, 
                     5.0: 10,
                     8.0: 14,}
    
    pclass_age_map = {1.0: 39,
                      2.0: 30,
                      3.0: 25}
    age = columns[0]
    pclass = columns[1]
    sibsp = columns[2]
    
    if pd.isna(age):
        if sibsp > 2.0:
            age = sibsp_age_map[sibsp]
        else:
            age = (pclass_age_map[pclass] + sibsp_age_map[sibsp]) / 2
            
    return age

# Now fill in the missing age
data['Age'] = data[['Age', 'Pclass', 'SibSp']].apply(impute_age, axis=1)

# Then fill in missing 'Fare' and 'Embarked'
# Since the amount missing is very small, 
# we simply use 'mode' value (most often seen value)
data['Fare'].fillna(data['Fare'].mode()[0], inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[ ]:


# STEP 3
gender =  pd.get_dummies(data['Sex'], drop_first=True)
embarked_location = pd.get_dummies(data['Embarked'], drop_first=True)
data = pd.concat([data, gender, embarked_location], axis=1, sort=False)

# Now we can safely drop both the 'Sex' column and the 'Embarked' column
data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

# Lastly we have to handle the 'Name' column
# NOTE: 
# We are only interested in the title of the passenger,
# as it may have implications with socio-economic status as well as age of the passenger.
# Title is located between the comma and the period of the name.
titles = [name.split(",")[1].split(".")[0].strip() for name in data["Name"]]
data["Title"] = pd.Series(titles)

# Since some titles are really rare, we group them together into one category.
data["Title"] = data["Title"]                 .replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

# Then we group the equivalent ones
data["Title"] = data["Title"].replace('Mlle','Miss')
data["Title"] = data["Title"].replace('Ms','Miss')
data["Title"] = data["Title"].replace('Mme','Mrs')

# convert using one-hot encoding
one_hot_titles = pd.get_dummies(data['Title'], drop_first=True)
data = pd.concat([data, one_hot_titles], axis=1, sort=False)

# Now we can safely remove both the 'Name' column and the 'Title' column
data.drop(['Name', 'Title'], axis=1, inplace=True)

# Let's take a look at what our training data becomes!
data.head()


# In[ ]:


# STEP 4
# In this case, variable 'Fare' is positively skewed.
# A skewed distribution will affect model's performance.
# We need to fix the skewness by either applying a boxcox transformation or log transformation.

# First let's take a look
before = sns.distplot(data['Fare'])


# ### After log transformation

# In[ ]:


# Apply a log transformation
data["Fare"] = data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# Let's see what it looks like now
after = sns.distplot(data['Fare'])


# ## Feature Engineering
# Sum `SibSp` and `Parch` into `FamilySize`

# In[ ]:


# Create column 'FamilySize' based on number of family members
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# Now we can safely remove 'SibSp' and 'Parch'
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# ## Training Preparation

# In[ ]:


# split the data
training_data = data.copy().iloc[:890]
test_data = data.copy().iloc[891:].drop('Survived', axis=1)

# 'Survived' column contains all the training labels
x = training_data.drop('Survived', axis=1)
y = training_data['Survived']

# take 30% of the training set as validation set
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=1)

# Build our machine learning model
# Feel free to explore hyperparameters
model = RandomForestClassifier(random_state=0,
                               n_estimators=450,
                               criterion='gini',
                               n_jobs=-1,
                               max_depth = 8,
                               min_samples_leaf=1,
                               min_samples_split= 11)


# ## Training and Validation

# In[ ]:


# Training
model.fit(x_train, y_train)
# Make prediction on the validation set
predictions = model.predict(x_val).astype(int)

# Report on how the model performs on the validation set
print(classification_report(y_val, predictions))
accuracies = cross_val_score(estimator=model, X= x_train, y=y_train, cv=10)

# average cross validation accuracy
print('Cross validation average: %.4f' % accuracies.mean())


# ### Feature Importance Visualization
# After training, we examine feature importance to see what features matter the most

# In[ ]:


feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': model.feature_importances_})
feature_importance_chart = feature_importance.sort_values(by='Importance').plot(x='Feature', kind='barh', title='Feature Importance')


# ## Apply on Test Data

# In[ ]:


passenger_id = pd.read_csv('../input/test.csv')['PassengerId'].values.tolist()
predictions = model.predict(test_data).astype(int)
submission = pd.DataFrame({'PassengerId':passenger_id, 
                           'Survived':predictions}).to_csv('prediction.csv', index=False)

