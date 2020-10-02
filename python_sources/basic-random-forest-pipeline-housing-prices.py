#!/usr/bin/env python
# coding: utf-8

# ## What's in This Notebook
# 
# This is meant as a guide to basic model creation, from importing the data through submitting a competition file using a random forest. The goal here is to get a big picture sense of the ML coding/thought process rather than specific deep dives into any particular topic.
# 
# I've used clean coding practices with an emphasis on pipelines in this notebook, which hopefully makes the code extra easy to read!

# ## Step 1: Import the Data

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# The data below is from the [Kaggle House Prices Competition Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). It's already split into train and test sets, so we can load both directly.

# In[ ]:


df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# ## Step 2: Look at the Data
# 
# It's hard to do data analysis on a dataset if you don't know what the data looks like. Knowing how many columns there are per row and what kind of information is in those columns informs what kinds of features you might train on. Knowing how many rows (datapoints) helps chose what model to use - for example, a neural net is unlikely to work well on a dataset that only has 100 datapoints. 
# 
# While you could spend eons on this step alone, a good first start to looking at your dataset is:
# 
#  - [df.head()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) to see what the first few datapoints look like.
#  - [df.describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) to look at numeric data ranges (count, mean, percentiles, etc.).
#  - [df.info(verbose=True, null_counts=True)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html) to learn the type (float, int, etc.) and number of null values in each column.
#  - [sns.heatmap(df.isnull(),yticklabels=False, cbar=False)](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to get a visual representation of how null values are distributed within the dataset.

# In[ ]:


print ("Num Cols: ", len(df.columns))
print ("Num Rows: ",len(df.index))


# In[ ]:


df.head()


# In[ ]:


df.info(verbose=True, null_counts=True)


# In[ ]:


df.describe()


# In[ ]:


ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
ax.set(xlabel='columns', ylabel='rows (white if null)')
plt.show()


# This heatmap show us where the null values in the dataset are located. Each column in the data is its own column in the heatmap. If the column is mostly black, it has very few null values, but if it's mostly white (like PoolQU or Alley) it contains mostly null values.
# 
# From the heatmap and info() call above, we can see that while most of the columns in this dataset will be useful, we have a few that have so many null datapoints that the signal from those columns may be less useful, and we might consider dropping those columns in the data cleaning stage.
# 
# Spending more time digging into the data can vastly improve feature selection. Given the time, one might look into creating additional features based off of these columns (for example, adding up all of the square foot features to get a total indoor square feet or adding up all of the bathroom columns to get a total bathroom count). For the sake of simplicity, we'll skip that step right now.

# In[ ]:


fig=plt.figure(figsize=(25, 5))
unique = df.select_dtypes(include=['object','category']).nunique().sort_values()
plt.bar(unique.index, unique)
plt.xticks(rotation=90)
plt.show()
print("min: ", unique.min())
print("max: ", unique.max())


# ## Step 3: Clean the Data
# 
# While the data from Kaggle competitions is already pretty clean, it still requires some manipulation before a model will accept the data as input. Specifically, we need to:
# 
#   1. **Select the features** we want to use in the model. This may be columns in their original state or features created using additional logic and/or combinations of column information.
#   2. Decide how to deal with null values in the dataset. This process is called **imputation**. 
#   3. **Encode** data that is in a string or other categorical form into numerical features that can be read by a model. The most common way to encode non-numeric features is called **one-hot encoding**, which we'll use below.
# 
# 

# As a baseline simplistic model, we'll use all columns in the dataset as features. That means we'll want to split the columns into three groups:
#  - **numeric columns**, which will be the easiest to process
#  - **categorical columns**, mostly columns that contain strings like "Neighbourhood" or "Building Style". We'll process these columns slightly differently than the numeric columns.
#  - **the target column**, that we're trying to predict. In this case, we're trying to predict the house price, so we'll keep that column separate from the others.
# 
#  From the df.info() results in step 2, we know that the categorical columns are all of type "object", a fact we can use to filter them from the numeric columns.

# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Define feature column categories by column type
categorical_cols = df.select_dtypes(include=['object','category']).columns.to_list()
numeric_cols = df.select_dtypes(include='number').columns.to_list()
# Remove the target column (SalePrice) from our feature list
numeric_cols.remove('SalePrice')

print ("Categorical columns: ", categorical_cols)
print ("Numeric columns: ", numeric_cols)


# While the numeric columns are almost in the correct format, there are still some NaN values we need to deal with. While there are many ways we could decide to impute (fill in) those missing values, we'll keep it simple here: we'll fill in missing values of a column with the average of all the current values in the column.
# 
# For example, if we had a column that before imputation looked like
# 
# [1, 3, None, 1, 3]
# 
# the average of the non-missing columns is 2, so after imputation the column would look like
# 
# [1, 3, 2, 1, 3].
# 
# To do the imputation for us, we'll use a [simple imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).

# In[ ]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')


# The categorical columns require a bit more work. First, we'll have to impute them as well, this time using the mode rather than the mean since we can't take the mean of a string. This takes the column
# 
# ['a', None, 'b', 'a']
# 
# to
# 
# ['a', 'a', 'b', 'a'].
# 
# Next, we'll need to transform those strings into numeric values using [one-hot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/). That takes a single column with options 'a' and 'b' and turns it into two columns: one that indicates (via 0/1) if a row contains an 'a', and one that indicates if a row contains a 'b'. Thus
# 
# 'a' now becomes [1, 0] and
# 
# 'b' now becomes [0, 1].
# 
# To do these transformations, we'll use both a [simple imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) and a [one-hot encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). The [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) is a way of keeping track of what transformations you do on the data in what order that makes those transformations easily repeatable on new data.

# In[ ]:


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Finally, we combine the numeric and categoric data encoding and imputing into one final preprocessor we can use on our data. Note how this column transformer uses the numeric transforms on our numeric columns and our categoric tranformns on our categoric columns.

# In[ ]:


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numerical_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])


# ## Step 4: Create a Train Test Split
# 
# We're finally ready to get back to our data! While we've already been given both a train set (which contains final house sale prices that we can train on) and a test set (which doesn't contain final house sale prices, we need to predict those with our model!) we still need to further split our train set into a **training set** and a **validation set**.
# 
# We'll use the training set to train various models, and the validation set to test how well those models do, allowing us to tune parameters.
# 
# Luckily, there's a great function called [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) that will split up the data into two groups for us. We'll put 80% of the data in the test set and 20% in the validation set.

# In[ ]:


from sklearn.model_selection import train_test_split

# Grab target as y, remove target from X
train_test = df.copy()
y = train_test.SalePrice
X = train_test.drop(columns=['SalePrice'])

# Split into train, test
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state = 17)


# In[ ]:


train_X.head()


# In[ ]:


train_y.head()


# ## Step 5: Train a Model
# 
# Now that we have training and test sets that are well formatted, we can fit a model! There are two main choices to make here:
#  - What **model type** are we going to use? (Linear regression, decision tree, neural net, etc.)
#  - What **error metric** are we going to use? (Mean absolute error, mean squared error, etc.)
# 
# Once you pick a model type and a metric, you can tune the model's individual parameters to optimize for the metric output. While these choices are critical to model success, for now, we'll keep it simple: We'll train using a [random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), and use [mean absolute error (MAE)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) as our success metric.

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

# Time to tune params!
def display_validation(pipeline):
    # Preprocessing of training data, fit model 
    pipeline.fit(train_X,train_y)
    # Preprocessing of validation data, get predictions
    preds = pipeline.predict(val_X)

    # Evaluate the model
    score = mean_absolute_error(val_y, preds)
    print('MAE:', score)


# As an example of what parameter tuning looks like, here we'll tune the number of estimators by calculating the MAE for each of three n_estimator values. Whichever has the smallest MAE value will likely give the best results!

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import random

for n in [50,100, 500]:
    model = RandomForestRegressor(n_estimators=n, random_state = 17)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
    print("n_estimators: ", n)
    display_validation(pipeline)


# Since the MAE is lowest for n_estimators=100, we'll use that in our submission below. (Normally, you'd do a lot more tuning and validation than this -- this example is just a baseline to get started with!)

# ## Step 6: Submit/Store a Final Model
# 
# To submit predictions in Kaggle, you nead to create a csv file with the correct column names. Here, we do that by running the test data through our same training pipeline to process the data and predict the values, then save that result as a csv.

# In[ ]:


# First, train again on that best n_estimators value
final_model = RandomForestRegressor(n_estimators=100, random_state = 17)
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', final_model)
                             ])

# Preprocessing of validation data, get predictions
final_pipeline.fit(train_X,train_y)
test_data_labels = final_pipeline.predict(test)

# Create predictions to be submitted!
pd.DataFrame({'Id': test.Id, 'SalePrice': test_data_labels}).to_csv('RFC_100.csv', index =False)  
print("Done :D")


# All done, congrats on getting to the end! 
