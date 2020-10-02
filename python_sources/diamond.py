#!/usr/bin/env python
# coding: utf-8

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


diamond = pd.read_csv('../input/diamonds.csv')
diamond.head()


# ## Split dataset to the training set and test set

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


index = diamond.index
columuns = diamond.columns
data = diamond.values


# **Drop the ID Column**

# In[ ]:


diamond = diamond.drop('Unnamed: 0', axis='columns')


# In[ ]:


diamond.head()


# In[ ]:


index = diamond.index
columuns = diamond.columns
data = diamond.values


# In[ ]:


diamond_num = diamond.select_dtypes(include=[np.number])


# In[ ]:


train_set, test_set = train_test_split(diamond, test_size=0.2, random_state=42)


# ## Data Analysis

# 1. Analyze the mean and median value for continuous variables
# 2. Analyze the outliers using the box plot (normalize the values using the z-scores)
# 3. Analyze the strength of correlation between the continous variables and labels using correlation matrix and t-test 

# In[ ]:


import matplotlib.pyplot as plt


# #### Analyze the mean, median and mode for the continuos variable

# In[ ]:


def plot_histogram(xlabel, title, values, n_bins):
    plt.hist(values, bins=n_bins)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.axvline(x=values.mean(), linewidth=2, color='r', label="mean")
    plt.axvline(x=values.median(), linewidth=2, color='y',label="median")
    for mode in values.mode():
        plt.axvline(x=mode, linewidth=2, color='k',label="mode")
    plt.legend()


# In[ ]:


diamond_num.head()


# In[ ]:


print("Minimum :", diamond.carat.min())
print("Maximum :", diamond.carat.max())
print("Range : ", diamond.carat.max() - diamond.carat.min())
plot_histogram(diamond.carat.name,"Histogram of "+diamond.carat.name, diamond_num[diamond.carat.name], 20)


# From the histogram for the carat, the data distribution is skewed to the right since the mean is the largest compared to the median and mode

# In[ ]:


print("Minimum :", diamond.depth.min())
print("Maximum :", diamond.depth.max())
print("Range : ", diamond.depth.max() - diamond.depth.min())
plot_histogram(diamond.depth.name,"Histogram of "+diamond.depth.name, diamond_num[diamond.depth.name], 20)


# From the histogram of depth variable, the data is symmetrically distributed since the mean, median and mode are almost at the same point

# In[ ]:


print("Minimum :", diamond.table.min())
print("Maximum :", diamond.table.max())
print("Range : ", diamond.table.max() - diamond.table.min())
plot_histogram(diamond.table.name,"Histogram of "+diamond.table.name, diamond_num[diamond.table.name], 20)


# From the histogram for the table variable, the distribution of table is skewed to the right since the mean is the largest compared to the median and mode although the skewness is not very profound.

# In[ ]:


print("Minimum :", diamond.price.min())
print("Maximum :", diamond.price.max())
print("Range : ", diamond.price.max() - diamond.price.min())
plot_histogram(diamond.price.name,"Histogram of "+diamond.price.name, diamond_num[diamond.price.name], 200)


# Based on the histogram for price, the price is skewed to the right since the mean is the largest of the mode and median

# #### Analyze outliers of the continuous variables using box plot 

# Calculate the z-scores of the continuous variables

# In[ ]:


import scipy as sp


# In[ ]:


diamond_num_z_scores = sp.stats.zscore(diamond_num)


# Plot the Box-Plot

# In[ ]:


plt.boxplot(diamond_num_z_scores)
plt.xticks(np.arange(1,8),diamond_num.columns)
plt.xlabel("Variables")
plt.ylabel("z-scores")
plt.title("Box Plot for Continuous Variables")


# ### Analyzing Qualitative Predictors

# In[ ]:


diamond_quali = diamond.select_dtypes(include=[np.object])


# In[ ]:


diamond_quali.head()


# In[ ]:


diamond_quali.cut.value_counts().plot(kind="bar")


# Based on the categorical plot, there is a case of inbalanced data in the cut predictor. Where there are too few Fair classes.

# In[ ]:


diamond.color.value_counts().plot(kind='bar')


# Based on the categorical plot, there is a case of inbalanced data that can be observed in class H, D, I, J

# In[ ]:


diamond.clarity.value_counts().plot(kind='bar')


# There is a case of biased dataset in the clarity class where there are too few samples of IF and I1 classes

# ## Correlation Analysis

# 1. Construct a correlation matrix over all the continuous variables.
# 2. Select feature when necessary.

# #### Correlation Matrix

# In[ ]:


corr = diamond_num.corr()


# In[ ]:


plt.figure(figsize=(12,9))
plt.imshow(corr, cmap='hot')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=20)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Matrix for Continuous Variables")


# All continuos variables have high correlation with price except for depth and table. Depth and table have very low correlation to each other. In fact, from the correlation matrix, depth and table is a weak predictor to other variables.  

# ## Feature Engineering

# To conduct feature engineering the following approach is conducted
# 1. Drop missing values from dataset.
# 2. Normalize all the numerical predictors in the training set.
# 3. Encode all the categorical variables using an encoding scheme.
# 4. Convert all the dataframe into a numpy array.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
mean = 0
std = 0

class DataframeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,test=False, label_column='price'):
        self.test = test
        self.label_column = label_column
        
    def seperate_quantitative_and_qualitative(self):
        quantitative = self.dataframe.select_dtypes(include=[np.number])
        qualitative = self.dataframe.select_dtypes(include=[np.object])
        return quantitative, qualitative
    
    def normalize_quantitative_values(self, data):
        ##if it is for training, calculate the mean and standard deviation and save it.
        if not self.test:
            global mean
            mean = data.mean()
            global std
            std = data.std()
            scaler = StandardScaler()
            data_normalize = scaler.fit_transform(data)
        elif self.test: #if test, use the same mean and standard deviation obtained from training data
            if mean.empty or std.empty:
                raise Exception('Training data had not been created')
            else:
                data_normalize = (data-mean)/std    
        return data_normalize
    
    def encode_text_attributes(self,data):
        cat = np.empty((data.shape[0],0))
        encoder = LabelBinarizer()
        for column in data.columns:
            cat = np.c_[cat, encoder.fit_transform(data[column])]
        return cat
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X = X.dropna()
        ##get the label
        self.label = X[self.label_column]
        self.dataframe = X.drop(self.label_column, axis='columns')     
        
        ##seperate the quantitative values and  qualitative values
        quantitative_data, qualitative_data = self.seperate_quantitative_and_qualitative()
        normalized_data = self.normalize_quantitative_values(quantitative_data)
        encoded_text_data = self.encode_text_attributes(qualitative_data)
        prepared_data = np.c_[normalized_data, encoded_text_data]
        return prepared_data, self.label    


# In[ ]:


datatransformer = DataframeTransformer()
training_prepared, label = datatransformer.fit_transform(train_set)


# In[ ]:


training_prepared.shape


# ## Training

# 1. Train using linear regression
# 2. Train using neural networks
# 3. Use cross validation
# 4. Use grid search for parameter search

# ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(training_prepared, label)


# #### Use Cross Validation to Evaluate the Model

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(lin_reg, training_prepared, label, 
                         scoring="neg_mean_squared_error", cv=10)


# In[ ]:


lin_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores(scores):
    print("Scores : ",scores)
    print("Mean : ", scores.mean())
    print("Standard Deviation : ", scores.std())


# In[ ]:


display_scores(lin_rmse_scores)


# This model underfits the data. And the standard deviation between the scores are too high. Attempt to use more complex model such as Polynomial Regression to fit the data better

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# Two degreee polynomial

# In[ ]:


poly_2_deg = PolynomialFeatures(degree=2, include_bias=False) 
full_diamond_features_poly_deg_2 = poly_2_deg.fit_transform(training_prepared)


# In[ ]:


polynomial_2_deg_model = lin_reg.fit(training_prepared, label)


# Use Cross Validation

# In[ ]:


scores = cross_val_score(polynomial_2_deg_model, training_prepared, label, 
                         scoring="neg_mean_squared_error", cv=5)
deg_two_poly_rmse_scores = np.sqrt(-scores)


# In[ ]:


display_scores(deg_two_poly_rmse_scores)


# Degree two model fits worse than the linear model. Try degree 3 polynomial just in case

# Degree three polynomial performs much worse than degree two polynomial. Stop increasing polynomial degrees. Try using regularization in the linear model

# #### Ridge Regularization

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = [{
    'alpha':[0.1, 0.01, 0.001, 0.0001, 0.000001, 0]
}]


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge = Ridge()


# In[ ]:


grid_search_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')


# In[ ]:


grid_search_ridge.fit(training_prepared, label)


# In[ ]:


grid_search_ridge.best_params_


# In[ ]:


ridge_model = Ridge(alpha=1e-06)


# In[ ]:


ridge_model.fit(training_prepared, label)


# In[ ]:


scores = cross_val_score(ridge_model, training_prepared, label, 
                         scoring="neg_mean_squared_error", cv=10)
ridge_model_rmse_scores = np.sqrt(-scores)


# In[ ]:


display_scores(ridge_model_rmse_scores)


# Only marginally improve the fit of the model

# #### Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


param_grid = [{
    'alpha':[1, 2, 5, 10, 12]
}]


# In[ ]:


lasso_reg = Lasso()


# In[ ]:


grid_search_lasso = GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error')


# In[ ]:


grid_search_lasso.fit(training_prepared, label)


# In[ ]:


grid_search_lasso.best_params_


# In[ ]:


lasso_model = grid_search_lasso.estimator


# In[ ]:


scores = cross_val_score(lasso_model, training_prepared, label, 
                         scoring="neg_mean_squared_error", cv=10)
lasso_model_rmse_scores = np.sqrt(-scores)


# In[ ]:


display_scores(lasso_model_rmse_scores)


# Lasso best model is when alpha=1. Produces almost the same performance as the Ridge Regression approach

# ### Neural Network Approach

# 1. Construct a 4 layer neural network
# 2. One input for the input with 26 neurons
# 3. 16, 8, and 1 Neurons respectively for the consequetive layers
# 4. One node for the output
# 5. Use dropout for regularization
# 6. Relu for the activation function

# In[ ]:


from tensorflow.keras import models
from tensorflow.keras import layers


# Define the model

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(26, activation='relu', input_shape=(26,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='relu'))


# In[ ]:


from tensorflow.keras import optimizers, losses, metrics


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.01),
             loss=losses.MSE,
             metrics=[metrics.MSE])


# In[ ]:


X_val = training_prepared[:10000]
y_val = label[:10000]
X_train = training_prepared[10000:]
y_train = label[10000:]


# In[ ]:


history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_val, y_val))


# In[ ]:


history_dict = history.history
loss_values = np.sqrt(history_dict['loss'])
val_loss_values = np.sqrt(history_dict['val_loss'])


# In[ ]:


epochs = range(1, len(history_dict['loss'])+1)


# In[ ]:


plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# In[ ]:


display_scores(val_loss_values)


# As aspected, the validation loss increases as the training loss decreases. An obvious case of overfitting. Therefore, it is better to apply K-fold Cross Validation to validate this model further

# ### Training Neural Network with K-Fold Cross Validation

# In[ ]:


def nn_k_fold_cv(model, n_folds=4):
    k = 4
    num_val_samples = len(training_prepared) // k
    num_epochs = 100
    all_scores = []
    for i in range(n_folds):
        print("Processing Fold # ", i)
        val_data = training_prepared[i * num_val_samples: (i + 1) * num_val_samples]
        val_target = label[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate(
            [training_prepared[: i * num_val_samples],
            training_prepared[(i + 1) * num_val_samples:]
        ], 
        axis=0)
    
        partial_train_targets = np.concatenate([
            label[:i * num_val_samples],
            label[(i + 1) * num_val_samples:]
        ], axis=0)
    
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=512, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0) 
        all_scores.append(val_mse)
    return all_scores


# In[ ]:


scores = nn_k_fold_cv(model)


# In[ ]:


display_scores(np.sqrt(scores))


# The neural network model extremely overfits the data. Attempt to reduce the neural network size by removing the second layer that contains 16 neurons.

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(26, activation='relu', input_shape=(26,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='relu'))


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.01),
             loss=losses.MSE,
             metrics=[metrics.MSE])


# In[ ]:


scores = nn_k_fold_cv(model)


# In[ ]:


display_scores(np.sqrt(scores))


# This model performs well on the cross validation set. Attempt to improve result by reducing the number of neurons on the second layer to 4 

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(26, activation='relu', input_shape=(26,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='relu'))


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.01),
             loss=losses.MSE,
             metrics=[metrics.MSE])


# In[ ]:


scores = nn_k_fold_cv(model)


# In[ ]:


display_scores(np.sqrt(scores))


# This model performs better than the previous model. So we will attempt to improve the model by making the model even smaller 

# In[ ]:


model = models.Sequential()
model.add(layers.Dense(26, activation='relu', input_shape=(26,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1, activation='relu'))


# In[ ]:


model.compile(optimizer=optimizers.RMSprop(lr=0.01),
             loss=losses.MSE,
             metrics=[metrics.MSE])


# In[ ]:


scores = nn_k_fold_cv(model)


# In[ ]:


display_scores(np.sqrt(scores))


# This model performs slightly better than the previous model. So this model will be used
# 
# *Note that sometimes this model performs slightly worse than the previous model due to the randomization of the training data. Hence the result might differ after commit*

# ## Testing

# In[ ]:


testdatatransformer = DataframeTransformer(test=True)


# In[ ]:


test_prepared, test_label = testdatatransformer.fit_transform(test_set)


# In[ ]:


prediction = model.predict(test_prepared)


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


nn_mse = mean_squared_error(prediction, test_label)


# In[ ]:


print("Test RMSE", np.sqrt(nn_mse))


# Based on the test set, this model also performs well on the test set.

# ## Training Using Gradient Boosting via XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


def xgb_k_fold_cv(xgb_model, n_folds=4):
    k = 4
    num_val_samples = len(training_prepared) // k
    all_scores = []
    for i in range(n_folds):
        print("Processing Fold # ", i)
        val_data = training_prepared[i * num_val_samples: (i + 1) * num_val_samples]
        val_target = label[i * num_val_samples: (i + 1) * num_val_samples]
    
        partial_train_data = np.concatenate(
            [training_prepared[: i * num_val_samples],
            training_prepared[(i + 1) * num_val_samples:]
        ], 
        axis=0)
    
        partial_train_targets = np.concatenate([
            label[:i * num_val_samples],
            label[(i + 1) * num_val_samples:]
        ], axis=0)
        
        xgb_model.fit(partial_train_data, partial_train_targets)
        predict = xgb_model.predict(val_data)
        val_mse = mean_squared_error(predict, val_target)
        all_scores.append(val_mse)
    return all_scores


# In[ ]:


diamond_xgb = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.6, 
                               learning_rate=0.1, max_depth=12, alpha=5,
                              n_estimators=100)


# Use cross validation to train and evaluate the model

# In[ ]:


scores = xgb_k_fold_cv(diamond_xgb)


# Best MSE score obtained using Boosted Decision Trees

# In[ ]:


display_scores(np.sqrt(scores))


# Evaluate the model with test data

# In[ ]:


diamond_xgb.fit(training_prepared, label)


# In[ ]:


test_predict = diamond_xgb.predict(test_prepared)


# In[ ]:


print("XGBoost Test RMSE", np.sqrt(mean_squared_error(test_predict, test_label)))


# The model also fits well on the test data and performs better than the neural network model
