#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

TITANIC_TRAIN_DATA = "../input/train.csv"
TITANIC_TEST_DATA = "../input/test.csv"

def load_titanic_train_data():
	return pd.read_csv(TITANIC_TRAIN_DATA)

def load_titanic_test_data():
	return pd.read_csv(TITANIC_TEST_DATA)

def load_titanic_combined_data():
    frames = [load_titanic_train_data(), load_titanic_test_data()]
    return pd.concat(frames, ignore_index=True)

def get_fixed_features_data():
    titanic = load_titanic_train_data()
    titanic = _fixed_family_features_data(titanic)
    titanic = _cleaned_up_features_data(titanic)
    titanic = _fix_name_data(titanic)
    return titanic

def _fix_name_data(titanic):
    title_list=[
                'Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer'
               ]
    titanic['Title']=titanic['Name'].map(
        lambda x: _substrings_in_string(x, title_list)
    )
    titanic['Name'] = titanic.apply(_replace_titles, axis=1)
    titanic = titanic.drop("Name", axis=1)
    return titanic
    
def _substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substring
    return np.nan

def _replace_titles(x):
    title=x['Name']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def _cleaned_up_features_data(titanic):
    titanic.Fare = titanic.Fare.map(lambda x: np.nan if x==0 else x)
    titanic.Cabin = titanic.Cabin.fillna('Unknown')
    titanic = titanic.drop("Cabin", axis=1)
    titanic = titanic.drop("Embarked", axis=1)
    return titanic

def _fixed_family_features_data(titanic):
    titanic['Family_Members'] = titanic['SibSp'] + titanic['Parch'] + 1
    titanic = titanic.drop("SibSp", axis=1)
    titanic = titanic.drop("Parch", axis=1)
    titanic = titanic.drop("Ticket", axis=1)
    return titanic


# In[ ]:


titanic = get_fixed_features_data()
titanic.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
titanic.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[ ]:


train_set, test_set = split_train_test(titanic, 0.2)
print("Train set %d" % len(train_set))
print("Test set %d" % len(test_set))


# In[ ]:


titanic.plot(kind="scatter", x="Family_Members", y="Survived", alpha=0.05)


# In[ ]:


corr_matrix = titanic.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[ ]:


corr_matrix["Family_Members"].sort_values(ascending=False)


# In[ ]:


from pandas.tools.plotting import scatter_matrix

attributes = ["Fare", "Sex", "Age", "Pclass", "Family_Members"]
scatter_matrix(titanic[attributes], figsize=(12, 8))

