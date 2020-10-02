#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #scratchpad
# 1. 'PassengerId' check if this can be useful
# 2. 'Name' check if this can be useful
# 3. 'Ticket' check if this can be useful
# 4. 'Cabin' check if this can be useful
# 
# 

# Loading the data into dataframes to analyse

# In[ ]:


raw_train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
raw_test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# Lets take a look at both dfs 

# In[ ]:


raw_train_df.head(5)


# In[ ]:


raw_test_df.head(5)


# In[ ]:


raw_train_df.shape


# In[ ]:


raw_test_df.shape


# Its best to merge train and test data together. This eliminates the hassle of handling train and test data seperately for various analysis and transformations.
# We'll then split them later into train and test once all the transformations are complete.
# 

# In[ ]:


merged_df = pd.concat([raw_train_df, raw_test_df], sort = False).reset_index(drop=True)

merged_df.head(5)


# In[ ]:


merged_df.shape


# Lets create a very naive model that we can then use to improve upon. For this naive model I'm simply going to pick some features, ensure they are numerical and have no missing values and so a decision tree.

# In[ ]:


interesting_cols = [# 'PassengerId', # lets not use this for now but make a note of it
                    'Survived',
                    'Pclass', 
                    'Name', # Going to use this to extract title and inpute missing ages. Will make sense later
                    'Sex', 
                    'Age',
                    'SibSp',
                    'Parch', 
                    # 'Ticket', # lets not use this for now but make a note of it
                    'Fare', 
                    # 'Cabin', # lets not use this for now but make a note of it
                    'Embarked'
                   ]


# In[ ]:


selected_df = merged_df.copy()


# In[ ]:


selected_df = selected_df[interesting_cols]


# In[ ]:


selected_df


# In[ ]:


def custom_one_hot_encode(df, column, categories):
    for category in categories:
        df[f'{column}_{category}'] = (df[column] == category) * 1
    del df[column]
            


# In[ ]:


preprocessed_df = selected_df.copy()



# In[ ]:



sns.heatmap(preprocessed_df.isnull(), cbar=False)


# In[ ]:


preprocessed_df.isnull().sum()


# In[ ]:


custom_one_hot_encode(preprocessed_df , 'Embarked', preprocessed_df.Embarked.unique())


# In[ ]:


custom_one_hot_encode(preprocessed_df , 'Sex', ['male', 'female'])


# In[ ]:


preprocessed_df


# In[ ]:


preprocessed_df.isnull().sum()


# there are 263 missing age values. We can probably infer this from the name by taking the average age of the title of each person.
# So for example the average age of miss, Mrs, Mr etc. 
# 

# Here we extract everthing after the first comma and before the period. In other words the title of the person and create a new column called title

# In[ ]:


preprocessed_df['Title'] = preprocessed_df['Name'].apply(lambda st: st[st.find(",")+1:st.find(".")])


# In[ ]:


preprocessed_df


# Lets check is there are any rows with both the Age and Title empty. Hopefully there wont be any. If there was that means something is not right.

# In[ ]:


preprocessed_df[preprocessed_df['Age'].isnull() & preprocessed_df['Title'].isnull()]


# here we are grouping by title and taking the average age. Then this average age is used to fill in the NaN values based on the title.

# In[ ]:


preprocessed_df['Age'] = preprocessed_df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))


# There was also one missing 'Fare'. Lets deal with that

# In[ ]:


preprocessed_df[preprocessed_df['Fare'].isnull()]


# So its a 60.5 year old Male that Embarked from S and has a ticket class of 3. Can we infer his fare somehow?
# The easiest option is to just take the average which is what I'll do now.
# 
# Other complex methods may be to do some group by methods to see if there is any markable difference in mean fare depending on your age and ticket class.

# In[ ]:


preprocessed_df["Fare"].fillna(preprocessed_df["Fare"].mean(), inplace=True) 


# In[ ]:





# Checking to see if any NaN now

# In[ ]:


preprocessed_df.isnull().sum()


# 1. Great there are none. Lets delete the Title and Name now as we dont need them anymore

# In[ ]:


del preprocessed_df['Title']
del preprocessed_df['Name']


# here is what the final preprocessed_df looks like now:

# In[ ]:


preprocessed_df.head(5)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# splitting back into train and test

# In[ ]:


train_df = preprocessed_df.iloc[:891, :]
test_df  = preprocessed_df.iloc[891:, :]


# In[ ]:


train_df


# In[ ]:


test_df


# In[ ]:



del test_df['Survived']

"""Extract data sets as input and output for machine learning models."""
xTrain = train_df.drop(columns = ["Survived"], axis = 1) # Input matrix as pandas dataframe (dim:891*47).
yTrain = train_df['Survived'] # Output vector as pandas series (dim:891*1)

"""Extract test set"""
xTest  = test_df.copy()

"""See the dimensions of input and output data set."""
print(f"Input Matrix Dimension: {xTrain.shape}")
print(f"Output Vector Dimension: {yTrain.shape}")
print(f"Test Data Dimension: {xTest.shape}")


# Lets scale these as the features have different scales

# In[ ]:


# scaled_preprocessed_df = preprocessed_df.copy()
scaler = StandardScaler()

scaled_xTrain = pd.DataFrame(scaler.fit_transform(xTrain), columns=xTrain.columns)
scaled_xTest = pd.DataFrame(scaler.fit_transform(xTest), columns=xTest.columns)


# Here im using GridSearch to run different hyperameters

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nparameters = {'max_depth': [3,5,7,10],\n              'min_samples_split': [1, 5, 10, 20, 50, 100]\n             }\n\n         \n\ndtc = DecisionTreeClassifier()\ngridsearch_cv = GridSearchCV(dtc, parameters, scoring='accuracy')\nresult = gridsearch_cv.fit(scaled_xTrain, yTrain)\nresult")


# Lets view the results sorted by score

# In[ ]:


pd.DataFrame(result.cv_results_).sort_values(by='rank_test_score')


# Although Max depth 5 has a higher mean score I would go with max depth 3 as it has a lower standard deviation with not much difference in the mean.

# The bestr parameters for the decision tree are:

# In[ ]:


gridsearch_cv.best_params_


# Lets make some predictions now:

# In[ ]:


predictions = gridsearch_cv.predict(xTest)


# In[ ]:


predictions.shape


# In[ ]:



submissionDTC = pd.DataFrame({
        "PassengerId": raw_test_df["PassengerId"],
        "Survived": predictions})
submissionDTC.to_csv("Submission.csv", index = False)

