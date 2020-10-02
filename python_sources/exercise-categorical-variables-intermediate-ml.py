#!/usr/bin/env python
# coding: utf-8

# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# ---
# 

# By encoding **categorical variables**, you'll obtain your best results thus far!
# 
# # Setup
# 
# The questions below will give you feedback on your work. Run the following cell to set up the feedback system.

# In[ ]:


# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex3 import *
print("Setup Complete")


# In this exercise, you will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and validation sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  The test set is loaded in `X_test`.

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)


# Use the next code cell to print the first five rows of the data.

# In[ ]:


X_train


# In[ ]:


X_train.head()


# Notice that the dataset contains both numerical and categorical variables.  You'll need to encode the categorical data before training a model.
# 
# To compare different models, you'll use the same `score_dataset()` function from the tutorial.  This function reports the [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE) from a random forest model.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# # Step 1: Drop columns with categorical data
# 
# You'll get started with the most straightforward approach.  Use the code cell below to preprocess the data in `X_train` and `X_valid` to remove columns with categorical data.  Set the preprocessed DataFrames to `drop_X_train` and `drop_X_valid`, respectively.  

# In[ ]:


# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# Check your answers
step_1.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()


# Run the next code cell to get the MAE for this approach.

# In[ ]:


print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


# # Step 2: Label encoding
# 
# Before jumping into label encoding, we'll investigate the dataset.  Specifically, we'll look at the `'Condition2'` column.  The code cell below prints the unique entries in both the training and validation sets.

# In[ ]:


print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())


# In[ ]:


set(X_train['Condition2'].unique()) -  set(X_valid['Condition2'].unique())


# In[ ]:


set(X_valid['Condition2'].unique()) -  set(X_train['Condition2'].unique())


# In[ ]:


bad_col_count = 0 
for col in X_train.select_dtypes(include=['object']).columns:
    if set(X_valid[col].unique()) != set(X_train[col].unique()):
        print(col)
        bad_col_count += 1 
print(bad_col_count)


# If you now write code to: 
# - fit a label encoder to the training data, and then 
# - use it to transform both the training and validation data, 
# 
# you'll get an error.  Can you see why this is the case?  (_You'll need  to use the above output to answer this question._)

# In[ ]:


#step_2.a.hint()


# In[ ]:


# Check your answer (Run this code cell to receive credit!)
step_2.a.solution()


# This is a common problem that you'll encounter with real-world data, and there are many approaches to fixing this issue.  For instance, you can write a custom label encoder to deal with new categories.  The simplest approach, however, is to drop the problematic categorical columns.  
# 
# Run the code cell below to save the problematic columns to a Python list `bad_label_cols`.  Likewise, columns that can be safely label encoded are stored in `good_label_cols`.

# In[ ]:


# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
print(len(good_label_cols), len(bad_label_cols))


# Use the next code cell to label encode the data in `X_train` and `X_valid`.  Set the preprocessed DataFrames to `label_X_train` and `label_X_valid`, respectively.  
# - We have provided code below to drop the categorical columns in `bad_label_cols` from the dataset. 
# - You should label encode the categorical columns in `good_label_cols`.  

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
#____ # Your code here
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(label_X_train[col]) 
    label_X_valid[col] = label_encoder.transform(label_X_valid[col])
# Check your answer
step_2.b.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_2.b.hint()
#step_2.b.solution()


# Run the next code cell to get the MAE for this approach.

# In[ ]:


print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# # Step 3: Investigating cardinality
# 
# So far, you've tried two different approaches to dealing with categorical variables.  And, you've seen that encoding categorical data yields better results than removing columns from the dataset.
# 
# Soon, you'll try one-hot encoding.  Before then, there's one additional topic we need to cover.  Begin by running the next code cell without changes.  

# In[ ]:


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# The output above shows, for each column with categorical data, the number of unique values in the column.  For instance, the `'Street'` column in the training data has two unique values: `'Grvl'` and `'Pave'`, corresponding to a gravel road and a paved road, respectively.
# 
# We refer to the number of unique entries of a categorical variable as the **cardinality** of that categorical variable.  For instance, the `'Street'` variable has cardinality 2.
# 
# Use the output above to answer the questions below.

# In[ ]:


# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the 
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25

# Check your answers
step_3.a.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_3.a.hint()
#step_3.a.solution()


# For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset.  For this reason, we typically will only one-hot encode columns with relatively low cardinality.  Then, high cardinality columns can either be dropped from the dataset, or we can use label encoding.
# 
# As an example, consider a dataset with 10,000 rows, and containing one categorical column with 100 unique entries.  
# - If this column is replaced with the corresponding one-hot encoding, how many entries are added to the dataset?  
# - If we instead replace the column with the label encoding, how many entries are added?  
# 
# Use your answers to fill in the lines below.

# In[ ]:


# Lines below will give you a hint or solution code
#step_3.b.hint()
#step_3.b.solution()


# In[ ]:


# Fill in the line below: How many entries are added to the dataset by 
# replacing the column with a one-hot encoding?
OH_entries_added = 100 * 10000 - 10000

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with a label encoding?
label_entries_added = 0

# Check your answers
step_3.b.check()


# In[ ]:


X_train.shape


# # Step 4: One-hot encoding
# 
# In this step, you'll experiment with one-hot encoding.  But, instead of encoding all of the categorical variables in the dataset, you'll only create a one-hot encoding for columns with cardinality less than 10.
# 
# Run the code cell below without changes to set `low_cardinality_cols` to a Python list containing the columns that will be one-hot encoded.  Likewise, `high_cardinality_cols` contains a list of categorical columns that will be dropped from the dataset.

# In[ ]:


# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)


# Use the next code cell to one-hot encode the data in `X_train` and `X_valid`.  Set the preprocessed DataFrames to `OH_X_train` and `OH_X_valid`, respectively.  
# - The full list of categorical columns in the dataset can be found in the Python list `object_cols`.
# - You should only one-hot encode the categorical columns in `low_cardinality_cols`.  All other categorical columns should be dropped from the dataset. 

# In[ ]:


X_train[low_cardinality_cols].shape


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!

oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


OH_cols_train = pd.DataFrame(oh_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(oh_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

X_train_num = X_train.drop(object_cols, axis = 1)
X_valid_num = X_valid.drop(object_cols, axis = 1)


print('before concat')
print(OH_cols_train.shape)
print(OH_cols_valid.shape)


# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([X_train_num, OH_cols_train], axis=1)
OH_X_valid = pd.concat([X_valid_num, OH_cols_valid], axis=1)




print('after concat')
print(OH_X_train.shape)
print(OH_X_valid.shape)


# Check your answer
step_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
#step_4.hint()
#step_4.solution()


# Run the next code cell to get the MAE for this approach.

# In[ ]:


print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


# # Step 5: Generate test predictions and submit your results
# 
# After you complete Step 4, if you'd like to use what you've learned to submit your results to the leaderboard, you'll need to preprocess the test data before generating predictions.
# 
# **This step is completely optional, and you do not need to submit results to the leaderboard to successfully complete the exercise.**
# 
# Check out the previous exercise if you need help with remembering how to [join the competition](https://www.kaggle.com/c/home-data-for-ml-course) or save your results to CSV.  Once you have generated a file with your results, follow the instructions below:
# 1. Begin by clicking on the blue **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  
# 2. Ensure that the **Save and Run All** option is selected, and then click on the blue **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the blue **Submit** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
# 

# # steps for the final model:
# 
# 1. build model on oh_train + oh_valid
# 2. handle missing values in X_test  
# 3. apply OH on X_test 
# 4. make predictions 
# 5. submit predictions

# # build model on oh_train + oh_valid

# In[ ]:


# (Optional) Your code here
model = RandomForestRegressor(n_estimators=100)
model.fit(OH_X_train.append(OH_X_valid), y_train.append(y_valid))


# # handle missing values in X_test

# In[ ]:


cols_with_missing_val = [col for col in X_test.columns if X_test[col].isnull().any()]
print(cols_with_missing_val)


# In[ ]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='most_frequent')
my_imputer.fit(X_train)
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_test.columns = X_test.columns


# # apply OH on X_test

# In[ ]:


OH_cols_test = pd.DataFrame(oh_encoder.transform(imputed_X_test[low_cardinality_cols]))
OH_cols_test.index = imputed_X_test.index
print('OH_X_test shape', OH_cols_test.shape)
X_test_num = imputed_X_test.drop(object_cols, axis=1)
OH_X_test = pd.concat([X_test_num, OH_cols_test], axis=1)
print('OH_X_test shape after concat', OH_X_test.shape)


# # make predictions

# In[ ]:


test_preds = model.predict(OH_X_test)


# In[ ]:


output = pd.DataFrame({'Id': X_test.index, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
print('done!')


# # Keep going
# 
# With missing value handling and categorical encoding, your modeling process is getting complex. This complexity gets worse when you want to save your model to use in the future. The key to managing this complexity is something called **pipelines**. 
# 
# **[Learn to use pipelines](https://www.kaggle.com/alexisbcook/pipelines)** to preprocess datasets with categorical variables, missing values and any other messiness your data throws at you.

# ---
# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161289) to chat with other Learners.*