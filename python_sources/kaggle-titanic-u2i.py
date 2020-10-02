#!/usr/bin/env python
# coding: utf-8

# # Titanic @ u2i
# ![https://media1.giphy.com/media/OJw4CDbtu0jde/giphy.gif](https://media1.giphy.com/media/OJw4CDbtu0jde/giphy.gif)

# **Imports, loading data**

# In[ ]:


import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# **Exploring the data**

# In[ ]:


data.head()


# In[ ]:


data.info()


# What does it mean?
# 
# https://www.kaggle.com/c/titanic/data
# 

# **Hypothesis**

# Who was most likely to survive?
# ![https://www.shiftcomm.com/wp-content/uploads/2017/08/giphy.gif](https://www.shiftcomm.com/wp-content/uploads/2017/08/giphy.gif)

# ![https://66.media.tumblr.com/37e6f5cd93269829406ccb5bbccf34e0/tumblr_pezn7j41oN1qzs7uio4_r1_250.gif](https://66.media.tumblr.com/37e6f5cd93269829406ccb5bbccf34e0/tumblr_pezn7j41oN1qzs7uio4_r1_250.gif)

# In[ ]:


print(data.groupby(["Pclass", 'Sex'])["Survived"].value_counts(normalize=True))


# **Dummy classifier**
# y = data['Surv

# Implement first model - DummyClassifier. It predict results randomly with respect to the training data values distribution.
# * divide the data into features and target - define X, y
# * create the model - an instance of DummyClassifier
# * fit (train) the model on X (method: *fit*)
# * predict "Survival" values of X (method: *predict*)
# * evaluate predictions - compare them with y. Use *accuracy_score(y, predictions)*

# In[ ]:


#code here
y = data.Survived
X = data.drop('Survived', axis=1)
model = DummyClassifier()
model.fit(X,y)
predictions = model.predict(X)
accuracy_score(y, predictions)


# We can do better. Let's use **Decision trees.**
# 

# ![https://cdn-images-1.medium.com/max/1200/1*7EeUAcoUOPLP6DRDhl5IUA.png](https://cdn-images-1.medium.com/max/1200/1*7EeUAcoUOPLP6DRDhl5IUA.png)

# * Do the same as above, but with *DecisionTreeClassifier*

# In[ ]:


#code here
y = data.Survived
X = data.drop('Survived', axis=1)
model = DecisionTreeClassifier()
model.fit(X,y)
predictions = model.predict(X)
accuracy_score(y, predictions)


# Ooops! We need to transform our features. 

# In[ ]:


data.sample(10)


# **One hot encoding vs factorizing**

# In[ ]:


OHE = pd.get_dummies(data)
OHE.head()


# In[ ]:


factorized_data = data.copy()
factorized_data['Embarked_cat'] = data['Embarked'].factorize()[0]
factorized_data.sample(20)


# **Select only impactful features:**

# In[ ]:


def select_features(df):
    int_features =  df.columns.values
    blacklist = ['Survived','PassengerId', 'Name', 'Ticket', 'Fare']
    features = [feat for feat in int_features if feat not in blacklist]
    return features

df = data[select_features(data)]
df.head()


# ## **Feature engineering**

# **Cabin**

# In[ ]:


# what kind of values can Cabin have?
df['Cabin'].value_counts()

#are there any missing values?
# df['Cabin'].isnull().sum()


# nan
# str(nan) == 'nan'

# In[ ]:


def cabin(df):
    new_df = df.copy()
    # map the Cabin feature to it's first letter. Pay attetntion to missing value. Hint: you can use map() and lambdas.
    new_df['Cabin'] = df['Cabin'].map(lambda x: 'NA' if str(x) == 'nan' else x[0] ) #code here
    return new_df

df = cabin(df)
df.sample(10)

cabin(data).groupby('Cabin')["Survived"].value_counts(normalize=True)


# **Age**

# In[ ]:


df['Age'].value_counts()


# In[ ]:


def age(df):
    new_df = df.copy()
    age_bins = [0, 1, 3, 5, 9, 15, 20, 40, 60, 100]
    new_df["Age"] = pd.cut(new_df["Age"], bins=age_bins).astype(object)
    return new_df

df = age(df)
df.sample(20)


# In[ ]:


age(data).groupby('Age')['Survived'].value_counts(normalize=True)


# In[ ]:


def factorize(data):
#     data['Sex_categorized'] = data['Sex'].factorize()[0]
#     data['Embarked_cat'] = data['Embarked'].factorize()[0]
    return pd.get_dummies(data)

df = factorize(df)
df.sample(10)


# In[ ]:


def make_prediction_on_all(df, model):
    X = df
    y = data['Survived'].values
    model.fit(X,y)
    y_pred = model.predict(X)
    return y_pred, model, accuracy_score(y, y_pred)


# In[ ]:


def decision_tree(df):
    return make_prediction_on_all(df, DecisionTreeClassifier())
    
decision_tree(df)[2]


# ![image.png](attachment:image.png)

# The score is too good to be true. What have we missed?

# **Add train/test split**

# In[ ]:


def make_prediction(df, model):
    X = df
    y = data['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, model, accuracy_score(y_test, y_pred)

def decision_tree(df):
    return make_prediction(df, DecisionTreeClassifier())

decision_tree(df)[2]


# **Cross validacja**
# 

# ![image.png](attachment:image.png)

# In[ ]:


def make_prediction_cross(X, model):
    y = data['Survived'].values
    return cross_validate(model, X, y, scoring='accuracy', cv=5)

def plot_result(model_name, result):
    mean_train = np.round( np.mean(result['train_score']), 2 )
    mean_test = np.round( np.mean(result['test_score']), 2 )
    
    plt.title('{0}: cross validation\nmean-train-acc:{1}\nmean-test-acc:{2}'.format(model_name, mean_train, mean_test))
    plt.plot( result['train_score'], 'r-o', label="train" )
    plt.plot( result['test_score'], 'g-o', label="test" )
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('# of fold')
    plt.show()

result = make_prediction_cross(df, DecisionTreeClassifier())
plot_result("DecisionTree", result)


# **Random forest**

# In[ ]:


def random_forest(df):
    return make_prediction(df, RandomForestClassifier(n_estimators=5))
    
random_forest(df)[2]


# **Submit to Kaggle**

# In[ ]:


test_data = pd.read_csv('../input/test.csv')
df_test = test_data[select_features(test_data)]
df_test = cabin(df_test)
df_test = age(df_test)
df_test = factorize(df_test)

final_train, final_test = df.align(df_test, join='left', axis=1, fill_value=0)
_, model,score = make_prediction_on_all(final_train, RandomForestClassifier(n_estimators=5))
preds = model.predict(final_test)
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':preds})

filename = 'u2i.csv'

submission.to_csv(filename,index=False)


# ## How to submit?
# * Click on "Commit" in the upper right corner. It will run this notebook from top to bottom.
# * When it finishes, click on "Open version". Find "Output" in the left menu, your file should be present. Click "Submit to competition"

# # Next
# * try other classifiers from sklearn
# * try to use other features (Name, Ticket, Fare)
# * extract new features from existing ones (e.g sum family members)
# * play with hyperparameters to reduce overfitting
# * other ideas?
