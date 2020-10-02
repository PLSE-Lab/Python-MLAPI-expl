#!/usr/bin/env python
# coding: utf-8

# # Housing Price Predictions in Ames, Iowa

# The aim of the project is to build an accurate predictor of house prices in Ames, Iowa.  This can be very helpful for people who are either trying to sell their house and are looking a fair evaluation of it's price or for people looking to buy a house to get some insights into the factors that play a key role in determing the house prices.  We will use advanced regression techniques such as **Gradient Boosting** to make these predictions.  This project is based on the **Kaggle** competition **House Prices: Advanced Regression Techniques**.

# ## Getting the data

# The data for this project was obtained from the **Kaggle** links https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv and https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv for the train set and test set respectively.  Additionally, you can go to https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/205873/data_description.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572954709&Signature=sYEzPiNPFc%2Bq8wiQWT3WpeYQTStONcL%2FIHg0%2FVDmJ4XH3UecbgJB%2Bzr4gj%2FmdD7NOt6OHpsRP9tKWlGzbWzYTQ4QbfVXAtZt%2FMV6NOe2LVFQUcPQwQnM9ZBAnPAWUu6bZRRfOpGxENlPD4oDPA8Xmfod94G5BUOaL1JPOuPU%2Bkcv2UHEeffpAQxUK5%2B%2BcRhJMl68VDtxsuOkjfKj3D98Gl9FvOiioGdcRTE0L0h4i0RJQAp821g2MF5XqznbDpqDjpm6e7HhNDwkIvktBEBdwVYPE73BlpeRH%2B2awizGZIoVaCrliNIOcHyGCki8PqsK15AXr05Tt%2BVdZQKEr0KgxQ%3D%3D to get a description of the features used in the data.

# Lets first import the necessary libraries.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Lets load the data now.  The csv files have been uploaded as train.csv and test.csv.

# In[ ]:


train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col="Id")
train_df.head()


# In[ ]:


test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",index_col="Id")
test_df.head()


# Lets visualize the distribution of the sale prices of the houses.

# In[ ]:


plt.hist(train_df["SalePrice"])
plt.xlabel("House Prices")
plt.ylabel("No. of houses")
plt.title("Distribution of House prices in Ames, Iowa")
plt.show()


# The histogram makes it clear that the majority of houses were sold in the range of $100,000-200,000.

# ## Data Preprocessing

# Lets separate the training dataframe into the features and the targets.

# In[ ]:


train_df.dropna(axis=0,subset=["SalePrice"],inplace=True)  #drop rows with no sale price
y_train=train_df.SalePrice  #define the target variable
X_train=train_df.drop("SalePrice",axis=1)  #drop sale price from predictor df


# Lets separate the columns that deal with numerical and categorical data respectively. 

# In[ ]:


#Columns with numerical data
num_cols=[col for col in X_train.columns if X_train[col].dtype in ["int64","float64"]]
print("There are "+str(len(num_cols))+" numerical columns.")
num_cols


# In[ ]:


#Columns with categorical data
cat_cols=[col for col in X_train.columns if X_train[col].dtype=="object"]
print("There are "+str(len(cat_cols))+" categorical columns.")
cat_cols


# We will have to perform one hot encoding on the numerical columns.  Lets see the number of unique categories in each column.

# In[ ]:


X_train[cat_cols].nunique()


# In[ ]:


cols=num_cols+cat_cols
X_train=X_train[cols]
X_test=test_df[cols]


# We will deal with missing values by imputing the missing values and we will use one hot encoding on the categorical columns.  To make the code cleaner and more compact, we will use pipelines.

# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_transformer=SimpleImputer(strategy="median")

cat_transformer=Pipeline(steps=[
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore",sparse=False))
])

preprocessor=ColumnTransformer(transformers=[
    ("num",num_transformer,num_cols),
    ("cat",cat_transformer,cat_cols)
])


# ## Modeling

# The model we will use in this case is the **XGBoost**.  It stands for extreme gradient boosting and is extremely efficient.

# In[ ]:


from xgboost import XGBRegressor

model=XGBRegressor(n_estimators=1000,learning_rate=0.05,random_state=0)


# Now lets build a pipeline which connects the preprocessor and the model.

# In[ ]:


pipe=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",model)
])


# Let's fit the overall model to the training data.

# In[ ]:


pipe.fit(X_train,y_train)


# ## Cross Validation

# We will use cross validation with 5 folds to evaluate our model's performance.  The metric we will use is the mean absolute error.

# In[ ]:


from sklearn.model_selection import cross_val_score

scores=-cross_val_score(pipe,X_train,y_train,cv=5,scoring="neg_mean_absolute_error") #The negative sign is used because negative mse is calculated
print("The average mean absolute error is "+str(scores.mean()))


# ## Final Predictions

# Let's use our model to predict the sale price of houses in the test set.

# In[ ]:


preds=pipe.predict(X_test)


# Let's convert our results into a dataframe with Id of the houses and their predicted prices.

# In[ ]:


output_df=pd.DataFrame({"Id":X_test.index,"SalePrice":preds})
output_df.head()


# Finally, let's convert the dataframe into a corresponding csv file.

# In[ ]:


output_df.to_csv("test_predictions.csv",index=False)


# ## Conclusion

# The purpose of this project was to build an accurate predictor for house prices in Ames, Idaho.  By using **imputation** and **one-hot encoding**, we were able to preprocess the data so that it can be used in a standard model.  Then we built and trained a model which uses **gradient boosting** and then evaluated the model's performance using **cross validation**.  Finally, we were able to predict house sale prices using the model.
