#!/usr/bin/env python
# coding: utf-8

# # GLOBAL IMPORTS / VARIABLES

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats


# In[ ]:


PATH_DATASET = os.path.join("/kaggle/input/", "random-linear-regression")
TRAINING_DATASET = "train.csv"
TESTING_DATASET = "test.csv"


# # Get the Data

# In[ ]:


def load_dataset(filename, path_dataset=PATH_DATASET):
    print("LOADED: " + filename)
    return pd.read_csv(os.path.join(path_dataset, filename))
    
train = load_dataset(TRAINING_DATASET)
test = load_dataset(TESTING_DATASET)


# # Exploring / Visualizing the data

# In[ ]:


data_points = train.copy()


# In[ ]:


data_points.head()


# In[ ]:


data_points.info()


# > it looks like there is 1 missing value for the y attribute

# In[ ]:


data_points["y"].describe()


# In[ ]:


data_points.plot(kind="scatter", x="x", y="y")


# > x and y seems to be positively correlated (as x increases, y increases as well), 
# > also there's seem to be no outliers in the dataset

# In[ ]:


corr_matrix = data_points.corr()


# In[ ]:


corr_matrix["y"]


# > the corr matrix back up my previous assumption, corr(x,y) = 0.99534

# In[ ]:


data_points["y"].hist()


# > the value of y does not seems to be following any particular distribution

# # Data Cleaning

# In[ ]:


data_points.info()


# In[ ]:


data_points_prepared = data_points.copy()

data_points_prepared["y"].fillna(data_points_prepared["y"].mean(), inplace=True)


# In[ ]:


data_points_prepared.info()


# In[ ]:


data_points_prepared.plot(kind="scatter", x="x", y="y")


# > notice that when I fill in the missing value of y with its mean, it created a situation where outlier exist. From the looks of things, the missing value of y has an x value above 3500, which is and outlier. Therefore, I think it is better to remove it out.

# In[ ]:


data_points_prepared = data_points.dropna()

data_points_prepared.info()


# In[ ]:


data_points_prepared.plot(kind="scatter", x="x", y="y")


# In[ ]:


data_points.plot(kind="scatter", x="x", y="y")


# # Select and Train a Model

# In[ ]:


data_points_prepared_without_y = data_points_prepared.drop(['y'], axis=1)
data_points_prepared_y = data_points_prepared['y'].copy()


# In[ ]:


linear_regression = LinearRegression()
linear_regression.fit(data_points_prepared_without_y, data_points_prepared_y)


# In[ ]:


print (linear_regression.coef_, linear_regression.intercept_)


# # Model Validation

# In[ ]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


lin_scores = cross_val_score(linear_regression, data_points_prepared_without_y, data_points_prepared_y, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[ ]:


pd.Series(lin_rmse_scores).describe()


# > as shown in the above statistics, model perform quite well. It is able to predict with the error rate of 2.810767 on average.

# # Evaluate on Test Set

# In[ ]:


data_points_test_without_y = test.drop("y", axis=1);
data_points_test_y = test["y"].copy();

final_predictions = linear_regression.predict(data_points_test_without_y)

final_mse = mean_squared_error(data_points_test_y, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[ ]:


final_rmse


# In[ ]:


plt.title('Comparison of Y values in test and the Predicted values')
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(data_points_test_without_y["x"], data_points_test_y, label="actual")
plt.scatter(data_points_test_without_y["x"], final_predictions, color="red", label="predicted")
plt.legend(loc='upper left')
plt.show()

