#!/usr/bin/env python
# coding: utf-8

# # Introduction to Scikit-Learn Pipelines
# 
# While working on a Machine Learning project, a vital step is to transform data before feeding it to heavy algorithms. Transformations can be done on all kinds of data types and helps the algoritm to learn from data quickly and accurately. Following are few examples of data transformations :-
# 1. On numerical data :-<br>
# 1.1. Min-max scaling<br>
# 1.2. Standardisation, etc.<br>
# 2. On categorical data :-<br>
# 2.1. Ordinal Encoding<br>
# 2.2. One-hot encoding, etc.<br>
# 
# Besides these popular engineering functions, you can create your custom transformations in Scikit-Learn which are useful to merge, extract, and modify your data features. 
# 
# So you did some EDA and brainstormed some feature engineering techniques for your dataset, great! But while handling a large dataset with 100s of attributes, keeping track of all these steps can get tedious. Also, after successfully applying all these functions on your training set and training your model, you need to go through the same process for your test set. As the task list gets longer, its easier to use the power of Scikit-Learn pipelines. Not only it keeps track of your transformations but also keeps your code clean, makes it easy to apply same changes on test set, and helps in de-bugging.
# 
# Here, we will use the famous Titanic survival dataset and see how Pipelines can help us make our job easier.

# ## 1. Importing Titanic survival dataset
# Dataset available on Kaggle <a href="https://www.kaggle.com/c/titanic/data?select=test.csv">here</a>.

# In[ ]:


import pandas as pd
import numpy as np

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# ## 2. Feature analysis

# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.head()


# To get a quick summary of the dataset use info()

# In[ ]:


df_train.info()


# ### Observations
# 
# 1. <b>PassengerId</b> is of type int and indetifies each passenger uniquely but won't be useful for our model.
# 
# 2. <b>Survived</b> is the target attribute of type Bool.
# 
# 3. <b>Pclass</b> diversify passengers into classes. Although the value is of type integer, this feature is of categorical nature.
# 
# 4. <b>Name</b> is of type object(string). Most likely it will be unique to each passenger and can be discarded before training.
# 
# 5. <b>Sex</b> is of type object(string). Its of categorical nature.
# 
# 6. <b>Age</b> is of type float. Looking at the dataset summary, we can observe that it contains some missing values.
# 
# 7. <b>SibSp</b> is of type int and describes the number of siblings/spouses of each passenger on the ship.
# 
# 8. <b>Parch</b> is of type int and describes the number of parents/children of each passenger on the ship.
# 
# 9. <b>Ticket</b> is of type object(string). It might have many(or all) unique values.
# 
# 10. <b>Fare</b> is of type float.
# 
# 11. <b>Cabin</b> is of type object(string). Its of categorical nature but contains many missing values.
# 
# 12. <b>Embarked</b> is of type object(string). Its of categorical nature and contains few missing values.

# ### Separating target feature

# In[ ]:


df_train_labels = df_train["Survived"].copy() 
df_train = df_train.drop(["Survived"], axis=1)


# ### Deleting not useful features
# By our observating above, we should discard PassengerId and Name while Ticket feature is still to be explored

# In[ ]:


df_train["Ticket"].nunique()


# It turns out that Ticket also has alot of unique values (80% approx of the entire set). So we should probably discard it for this tutorial.

# In[ ]:


df_train = df_train.drop(["PassengerId", "Name", "Ticket"], axis=1)


# ### Dealing with missing values 
# 
# Ideally we have 3 options to deal with missing values in a dataset :- 
# 1. Delete the attribute which contains missing values
# 2. Delete rows with missing values 
# 3. Fill missing values with something (0, mean, median, etc.)
# 
# In the training set, we have 3 columns with missing data :- 
# 1. <b>Age</b> - It contains 177(891-714) missing values. While its closer to 20% of the entire training set, discarding it can be costly to the model as Age might be an important factor for survival. Also, filling it with 0 is also not suitable. So, its better to fill missing values with mean or median.
# 2. <b>Cabin</b> - It contains 687(891-204) missing values. Choosing option 1 will be ideal in this situation as 204 datapoints might not add much value to our model(but it should be checked before being sure).
# 3. <b>Embarked</b> - It contains 2 missing values. Choosing option 3 will be ideal. We can replace missing values with most frequent values.
# 
# While we only have 3 features with missing values in our training data, we need to account for more features with missing value that might be in our test set.

# #### Filling missing _Age_ values with Median with _SimpleImputer_
# Scikit provides a great class to take care of missing values: _SimpleImputer_
# 
# We will apply this class on all numerical features so as to deal with missing values, if any, on unseen data.

# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_train_num = df_train[["Age","SibSp","Parch","Fare"]]
imputer.fit(df_train_num)
df_train_num_numpy = imputer.transform(df_train_num)
df_train_num = pd.DataFrame(df_train_num_numpy, 
                            columns=df_train_num.columns, 
                            index=df_train_num.index)


# Here, we have imported SimpleImputer, created an imputer with strategy="median", created a dataframe of just numerical features, applied fit() on the numerical dataset, and transformed the dataset. The output of transformed dataset is a NumPy array so we later convereted it to a Pandas Dataframe.

# In[ ]:


df_train_num.head()


# #### Deleting _Cabin_ feature

# In[ ]:


df_train = df_train.drop(["Cabin"], axis=1)


# #### Filling missing _Embarked_ values with Most Frequent with _SimpleImputer_
# We will apply this class on all categorical features so as to deal with missing values, if any, on unseen data.

# In[ ]:


imputer_cat = SimpleImputer(strategy="most_frequent")
df_train_cat = df_train[["Pclass","Sex","Embarked"]]
imputer_cat.fit(df_train_cat)
df_train_cat_numpy = imputer_cat.transform(df_train_cat)
df_train_cat = pd.DataFrame(df_train_cat_numpy, 
                            columns=df_train_cat.columns, 
                            index=df_train_cat.index)


# In[ ]:


df_train_cat.head()


# ## 3. Standardising numerical features
# Here we will standardise basic numerical features using Scikit-Learn _StandardScaler_

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train_num_numpy = scaler.fit_transform(df_train_num)
df_train_num = pd.DataFrame(df_train_num_numpy, 
                            columns=df_train_num.columns, 
                            index=df_train_num.index)


# In[ ]:


df_train_num.head()


# ## 4. Dealing with Categorical features
# We are remaining with only 3 categorical features, which are, Pclass, Sex, and Embarked.
# 
# We will use Sci-kit learn's OneHotEncder class to transform these into one-hot vectors.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

ohencoder = OneHotEncoder()
df_train_cat = ohencoder.fit_transform(df_train_cat)


# df_train_cat is a sparse matrix, to convert it to dense matric use, toarray()

# In[ ]:


df_train_cat = df_train_cat.toarray()


# ## 5. Creating Pipeline
# Finally, we move to the main step, creating a Scikit-Learn pipeline. All the above data transformations that we have applied can be implemented using a simple _Pipeline_ function very easily. 
# First, we will learn to build a custom transformation class.

# ### 5.1. Creating custom transformers 
# Custom transformation is nothing but changing data and features according to insights and observations rather than using traditional transformations. For eg:- Creating a new column from product of 2 columns. In case of this dataset, we could create a new column with sum of SibSp and Parch and discard them if we want to reduce our features. Another can be made with combination of age and fare.
# 
# Possibilities are endless, also, looking back at the current notebook, we have already implemented a transformation, we have deleted few attributes  which were not useful. 
# 
# Here is an example to create a custom data transformation class which work seemlessly with Scikit-Learn's other classes.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DeleteNotUsefulFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, delembarked = False):
        self.delembarked = delembarked
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)
        if self.delembarked:
            X = X.drop(["Embarked"], axis=1)
        return np.c_[X]


# We created a new class to delete unwanted features and which will run smoothly in our pipeline to transform data. 
# Here, BaseEstimator is used to avail two extra methods (get_params() and set_params()) and TransformerMixin is used to avail fit_transform() without explicitly writing a function for it.

# ### 5.2. Building Numerical Pipeline
# We are adding two transformations in the Numerical data pipeline which we implemented earlier, i.e, to deal with missing values and Standardisation.

# In[ ]:


from sklearn.pipeline import Pipeline

pipeline_num = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])


# ### 5.3. Builing Categorical Pipeline
# We are adding three transformations in the Ctaegorical data pipeline. These includes, custom transformation class to delete features, to deal with missing values and One-hot encoder.

# In[ ]:


pipeline_cat = Pipeline([
    ('del_features', DeleteNotUsefulFeatures(delembarked=False)),
    ('imputer_cat', SimpleImputer(strategy="most_frequent")),
    ('ohencoder', OneHotEncoder())
])


# ### 5.4. Getting everything together using _ColumnTransformer_
# 
# Now we will merge the two pipelines to create one full pipeline and transform our complete dataset in a single go.

# In[ ]:


from sklearn.compose import ColumnTransformer

num_features = ["Age", "SibSp", "Parch", "Fare"]
cat_features = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
all_features = num_features + cat_features

full_pipeline = ColumnTransformer([
    ('num_transform', pipeline_num, num_features),
    ('cat_transform', pipeline_cat, cat_features)
])


# In[ ]:


training_data = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


training_data.head()


# In[ ]:


train_transformed = full_pipeline.fit_transform(training_data)
train_transformed


# <b> So we have successfully transformed our entire dataset using Scikit-Learn's Pipeline. This can be very useful in projects and is a great tool to power through the code.
#     
# If you enjoyed the tutorial, please do give it an upvote!
# Comment below for feedback and suggestions. Thank you!
