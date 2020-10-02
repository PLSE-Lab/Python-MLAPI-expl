#!/usr/bin/env python
# coding: utf-8

# This notebook has use portion of code and text from: https://www.kaggle.com/dansbecker/submitting-from-a-kernel
# 
# **Introduction**
# 
# Machine learning competitions are a great way to improve your skills and measure your progress as a data scientist. If you are using data from a competition on Kaggle, you can easily submit it from your notebook. Here's how you do it.

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


# **Example**
# We're doing very minimal data set up here so we can focus on how to submit modeling results to competitions. Other tutorials will teach you how build great models. So the model in this example will be fairly simple. We'll start with the code to read data, select predictors, and fit a model.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR

# Read the data
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

svm_reg = LinearSVR(epsilon=1.5)

svm_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])

svm_reg.fit(train_X,train_y)


# In addition to your training data, there will be test data. This is frequently stored in a file with the title test.csv. This data won't include a column with your target (y), because that is what we'll have to predict and submit. Here is sample code to do that.

# In[ ]:


# Read the test data
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = svm_reg.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)


# **Prepare Submission File**
# We make submissions in CSV files. Your submissions usually have two columns: an ID column and a prediction column. The ID field comes from the test data (keeping whatever name the ID field had in that data, which for the housing data is the string 'Id'). The prediction column will use the name of the target field.
# 
# We will create a DataFrame with this data, and then use the dataframe's to_csv method to write our submission file. Explicitly include the argument index=False to prevent pandas from adding another column in our csv file.

# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission-SVR_SukulAdisak-4features.csv', index=False)


# **Make Submission**
# Hit the blue Publish button at the top of your notebook screen. It will take some time for your kernel to run. When it has finished your navigation bar at the top of the screen will have a tab for Output. This only shows up if you have written an output file (like we did in the Prepare Submission File step).

# **Last Steps**
# Click on the Output button. This will bring you to a screen with an option to Submit to Competition. Hit that and you will see how your model performed.
# 
# If you want to go back to improve your model, click the Edit button, which re-opens the kernel. You'll need to re-run all the cells when you re-open the kernel.

# **Conclusion**
# You've completed. Congrats.
# (Don't forget to submit it from the Output tab)
# 
# If you are ready to keep improving your model (and your skills), 
# Keep improving on:
#     your feature selections
#     data cleaning
#     and tuning hyperparameter (c and epsilon)
#     etc..
#     
# It is very possible that you can get a better score than me :)
# 
# //Adisak
# 
# 

# In[ ]:




