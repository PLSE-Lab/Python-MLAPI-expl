#!/usr/bin/env python
# coding: utf-8

# # An approach using different kinds of Regressors

# The idea of this notebook is to show you an approach making use of different regressors which are:
# 
# * XGBoosting
# * Random Forest
# * Gradient Boosting Tree
# * Ada Boost Regressor
# 
# In this notebook we compare the performance of each regressor making a variation in the number of estimator for each regressor.

# In[ ]:


"""Loading libraries and stuff"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Preprocessing and metrics"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""Regressors"""
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Preparing dataset

# For this exercise, we will only load a million of samples. The original dataset is like 2GB of samples which is so much for my computer, so loading only a million is enoguh for this example.

# In[ ]:


data = pd.read_csv('../input/train.csv', nrows=100000)


# Next we will to describe the dataframe to look some interesting things.

# In[ ]:


data.describe()


# Some important things we can notice is that the average fare is something like 11.3. Another important think is that the standard deviation in either pickup and dropoff longitud is similar as well as pickup and dropoff latitude.
# We can see that there are some null values, we will proceed to fix it.

# ## 1.1 Removing null values

# In[ ]:


data.isnull().sum()


# In[ ]:


print('Size with nulls (train): %d' % len(data))
data = data.dropna(how = 'any', axis = 'rows')
print('Size without nulls (train): %d' % len(data))


# ## 1.1 Removing fake fares

# One important thing is analyze if the dataset have noise. In this case we are goint to look if there are some "fake" fares. For this case we will accept as a "fake" value every sample minor than zero from the fares_amount column, this is because it could not be negative values for this column.

# In[ ]:


k = 0
for i in data.fare_amount:
    if i < 0:
        k += 1
print("Number of fake fares: ", k)


# There are 38 values which are under 0, this is not possible due to the fare must be more than zero. So, we will to proceed to remove all these values.

# In[ ]:


print('Length of original data: %d' % len(data))
data = data[data.fare_amount>=0]
print('Lengh of new data: %d' % len(data))


# # 2. Visualization

# Always is important to visualizate data, so in this part we are going to go deeper inside dataset for a better understanding.

# ## 2.1 Histogram of fare amount

# In[ ]:


data[data.fare_amount<70].fare_amount.hist(bins=70, figsize=(14,3))
plt.xlabel('Fare $USD')
plt.title("Histogram of fare's distribution");


# It is obvious that the fare amount are centered around 8usd. Also we can see that there are some extra points with fares so expensive, this is more than 45usd.

# # 3. Adding new features

# The orginal dataset is composed with the next features:
# 
# * fare_amount	
# * pickup_longitude	
# * pickup_latitude	
# * dropoff_longitude	
# * dropoff_latitude	
# * passenger_count
# 
# which are the basic atributes to describe the behavior of the pickup and dropoff process. But, are all of these values important? do we could add some more features to the dataset? which? how?.
# Well, it is important to notice that we could add more features from the dataset, for example the distance between the pickup and dropoff a customer. So, based in this idea we will proceed to add this features to the dataset.

# In[ ]:


data['lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_train  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'lat_change',
               'lon_change']


# In[ ]:


Y = data.fare_amount.as_matrix()
X = data[features_train]


# In[ ]:


Y = np.reshape(Y,(Y.shape[0],1))


# In[ ]:


X.head()


# ## 3.1 Scaling the Y vector

# The target (Y) vector has values bigger than zero, for simplicity we will scale it in a range of 0,1.

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
Y = min_max_scaler.fit_transform(Y)


# ## 3.2 Splitting data

# In[ ]:


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


#np.save('Xtrain.npy',Xtrain)
#np.save('Xtest.npy', Xtest)
#np.save('Ytrain.npy', Ytrain)
#np.save('Ytest.npy', Ytest)


# # 4. Regression modules

# In this section we will present a class called "Regressors" which has defined every algorith we will use in this notebook. 

# In[ ]:


class Regressors:
    def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
        """Initializing"""
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.Ytrain = Ytrain
        self.Ytest = Ytest

    def def_xgboost(self, estimators):
        """Defining the XGBoosting regressor"""
        xgb_ = xgb.XGBRegressor(objective ='reg:linear', learning_rate=0.01, max_depth=3, n_estimators=estimators)
        xgb_.fit(self.Xtrain, self.Ytrain)
        pred = xgb_.predict(self.Xtest)
        
        return pred

    def def_RandomForestRegressor(self, estimators):
        """Defining the Random Forest Regeressor"""
        rfr_ = RandomForestRegressor(n_estimators=estimators, max_depth=3)
        rfr_.fit(self.Xtrain, self.Ytrain)
        pred = rfr_.predict(self.Xtest)

        return pred

    def def_GradientBoostingRegressor(self, estimators):
        """Defining the Gradient Boosting Regressor"""
        gbr_ = GradientBoostingRegressor(n_estimators=estimators, max_depth=3)
        gbr_.fit(self.Xtrain, self.Ytrain)
        pred = gbr_.predict(self.Xtest)

        return pred

    def def_AdaBoostRegressor(self, estimators):
        """Defining Ada Boosting Regressor"""
        abr_ = AdaBoostRegressor(n_estimators=estimators)
        abr_.fit(self.Xtrain, self.Ytrain)
        pred = abr_.predict(self.Xtest)

        return pred


# Now we will define some functions which are "def_metrics" and "plot_performance". 
# The functionality of "def_metrics" is to calculate the mean absolute error and the mean squeare error for the regression algorithms.
# The functionality of "plot_ performance" is to display a graph which help us to visualize in a better way the performance of every regression algorithm.

# In[ ]:


def def_metrics(ypred):
    mae = mean_absolute_error(Ytest, ypred)
    mse = mean_squared_error(Ytest, ypred)

    return mae, mse

def plot_performance(plot_name, loss_mae, loss_mse):
    steps = np.arange(50, 500, 50)
    plt.style.use('ggplot')
    plt.title(plot_name)
    plt.plot(steps, loss_mae, linewidth=3, label="MAE")
    plt.plot(steps, loss_mse, linewidth=3, label="MSE")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Number of estimators")
    plt.show()


# In[ ]:


"""Initializing the class"""
model = Regressors(Xtrain, Xtest, Ytrain, Ytest)


# ## 4.2 XGBoost

# In[ ]:


plot_name="XGBoosting Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)


# ## 4.2 Random Forest Regressor

# In[ ]:


plot_name="Random Forest Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)


# ## 4.3 Gradient Boosting Regressor

# In[ ]:


plot_name="Gradient Boosting Regressor"
loss_mae, loss_mse = [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)


# ## 4.4 Ada Boost Regressor

# In[ ]:


plot_name="Ada Boost Regressor"
loss_mae, loss_mse= [], []
print(plot_name)
for est in range(50,500,50):
    print("Number of estimators: %d" % est)
    mae, mse = def_metrics(model.def_xgboost(estimators = est))
    print("MAE: ", mae)
    print("MSE: ", mse)
    loss_mae.append(mae)
    loss_mse.append(mse)
plot_performance(plot_name, loss_mae, loss_mse)


# # 5. Conclusion

# In this notebook we have shown how the performance is getting improved realted with the number of estimators for each regression algorithm. A future work could be focused on varying the hyperparamters for every regressor and apply cross validation.

# In[ ]:




