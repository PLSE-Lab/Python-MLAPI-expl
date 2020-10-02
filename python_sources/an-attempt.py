#!/usr/bin/env python
# coding: utf-8

# This is my first attempt to train and test a model, I would appreciate a lot to have some feedback so I can improve.
# I'm following [this book](http://shop.oreilly.com/product/0636920052289.do).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


raw_data = pd.read_csv('../input/BlackFriday.csv')
raw_data.head()


# I think User_ID and Product_ID are not relevant so I drop them.

# In[ ]:


raw_data.info()
prediction_data = raw_data.drop(["User_ID", "Product_ID"],axis=1)


# Product category 1 and 2 seem to be missing lot of data so I drop them as well.

# In[ ]:


prediction_data = prediction_data.drop("Product_Category_2", axis = 1)
prediction_data = prediction_data.drop("Product_Category_3", axis = 1)
prediction_data.info()


# Gender and age are String categories so I encode them:
# * Gender [F,M] -> [0,1]
# * Age: ['0-17' '18-25' '26-35' '36-45' '46-50' '51-55' '55+'] I use one hot encoding for this one.
# * City Category: ['A' 'B' 'C'] -> [0, 1, 2]
# * Stay in current city years: ['0', '1', '2', '3' , '4+'] -> [0, 1, 2, 3, 4] 
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
toEncode = prediction_data["Gender"]
genderEncoded = encoder.fit_transform(toEncode)
prediction_data["Gender"] = genderEncoded
prediction_data.head()


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
age_1hot = encoder.fit_transform(prediction_data["Age"])
print(encoder.classes_)
print(age_1hot)

# Probably there's a much better way to do this
age_1hot = age_1hot.transpose()
for (cat, ar) in zip(encoder.classes_, age_1hot):
    prediction_data[cat] = ar
print(prediction_data.head())
prediction_data = prediction_data.drop("Age", axis=1)
prediction_data.head()


# In[ ]:





# In[ ]:


encoder = LabelEncoder()
toEncode = prediction_data["City_Category"]
cityEncoded = encoder.fit_transform(toEncode)
prediction_data["City_Category"] = cityEncoded
print(encoder.classes_)
prediction_data.head()


# In[ ]:


encoder = LabelEncoder()
toEncode = prediction_data["Stay_In_Current_City_Years"]
yearsEncoded = encoder.fit_transform(toEncode)
prediction_data["Stay_In_Current_City_Years"] = yearsEncoded
print(prediction_data.info())
prediction_data.head()


# Every feature is numerical now so I explore a bit

# In[ ]:


prediction_data["Gender"].value_counts() / len(raw_data)


# Seems like men love this store, so I use stratified sampling to get a more representative train / test sets of the population. 
# Probably the error wouldn't have been too much either, but the book I mentioned before explains this and I wanted to try.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(prediction_data, prediction_data["Gender"]):
    train_set = prediction_data.loc[train_index]
    test_set = prediction_data.loc[test_index]


# In[ ]:


train_set["Gender"].value_counts()/len(train_set)


# In[ ]:


test_set["Gender"].value_counts()/len(test_set)


# Quick look at the correlation matrix and histograms

# In[ ]:


corr_matrix = prediction_data.corr()
corr_matrix["Purchase"].sort_values(ascending=False)


# In[ ]:


prediction_data.hist(bins=50,figsize=(20,15))


# I drop the labels from the train set to try the first model: Linear Regression

# In[ ]:


blackFriday = train_set.drop("Purchase", axis=1)
blackFridayLabels = train_set["Purchase"].copy()


# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(blackFriday, blackFridayLabels)


# In[ ]:


some_data = blackFriday.iloc[:10]
some_labels = blackFridayLabels.iloc[:10]
print("Predictions:\t",lin_reg.predict(some_data))
print("Labels:\t", list(some_labels))


# In[ ]:


from sklearn.metrics import mean_squared_error
predictions = lin_reg.predict(some_data)
lin_mse = mean_squared_error(predictions, some_labels)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# To see validate the model I use Cross-Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, blackFriday, blackFridayLabels, 
                         scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print("\nScores: ", rmse_scores, "\nMean: ", rmse_scores.mean()
     ,"\nStd Deviation: " , rmse_scores.std())


# I try Random Forest with Cross-Validation, it takes a while to train and test 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10)
scores_forest = cross_val_score (forest_reg, blackFriday, 
                                 blackFridayLabels,
                                 scoring = "neg_mean_squared_error",
                                 cv=10)
rmse_scores_forest = np.sqrt(-scores_forest)


# In[ ]:


print("\nScores: ", rmse_scores_forest, "\nMean: ", rmse_scores_forest.mean()
     ,"\nStd Deviation: " , rmse_scores_forest.std())


# I decided to go with this model, but I know that I should try many of them with many hyperparameters

# In[ ]:


forest_reg.fit(blackFriday, blackFridayLabels)


# I try against the test set, and the final RMSE is: 2957

# In[ ]:


test_x = test_set.drop("Purchase", axis=1)
test_labels = test_set["Purchase"].copy()


# In[ ]:


final_predictions = forest_reg.predict(test_x)
final_mse = mean_squared_error(test_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final RMSE: ", final_rmse)


# 
