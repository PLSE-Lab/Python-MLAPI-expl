#!/usr/bin/env python
# coding: utf-8

# The main purpose of this kernel is to represent a way of converting the categorical features in the feature set of car price prediction (like model, make, and such) into numerical features. This is usually done by labeling or using dummy features. However, a linear labeling method will not work as it will not be able to capture the correlation difference between various categories effectively. For instance, if I'm labeling Ferrari cars as 1 and Mazda cars as 2, it would not make sense for the difference between their lables (2 - 1) to be equal to the difference between Mazda and Toyota assuming Toyota was labeled as 3.
# As for Dummy features, that conversion method is great for categorical data but it falls short when there are many categories to consider. For instance, there are 50 states which means we need at least 49 features just to represent the State. This causes problems because of the curse of dimensionality.
# In order to overcome both problems, we went with a labeling method that isn't linear. Instead, we are processing the entire training set to collect means per category. Since we are predicting the price, we collected the mean of price per car make and used that as a label for all cars of that make. Same is done separately for both Model and State. This overcomes the dimensionality problem caused by the dummy features and also allows our model to better capture the variance of the categorical data in their labeled representaiton.
# We call this method Mean Labeling.

# In[1]:


#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm


# In[ ]:


train = pd.read_csv("../input/car-prices/data.csv") #Load the clean training data


# In[ ]:


print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))


# In[ ]:


#check missing values
train.columns[train.isnull().any()]


# Since there aren't any missing values, we don't have to worry about accounting for nulls in our data.

# In[ ]:


numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))


# In[ ]:


#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')

#correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)
print (corr['Price'].sort_values(ascending=False), '\n')


# By splitting the data into numerical and categorical, we can further process them separately to see their effects on the price.

# In[ ]:


cat_data.describe()
sp_pivot = train.pivot_table(index='Make', values='Price', aggfunc=np.mean).sort_values(by='Price') #Get mean price per make
sp_pivot.plot(kind='bar',color='blue')


# Now we can see that the make does somehow correlate with the price. though the correlation isn't very obvious since the make is categorical and not numerical. However if we somehow can replace the categorical data with a numerical replacement, we can possibly express this correlation in a way that a regressor can identify and gain some knowledge about the data from. The same can be done for State and Model. We are skipping City as it's correlation is not likely to be very beneficial with such scattered data.

# In[ ]:


#GrLivArea variable
sns.jointplot(x=np.log(train['Mileage']), y=np.log(train['Price']))


# From this graph showing Mileage vs Price, we can start looking at outliers that may adversly affect our model and preemptively remove them. After processing, we decided to remove all entries with Mileage below 5000, All Models that have less than 5 entries in the training data and all entries with price over 60000$

# In[ ]:


X = train[['Year', 'Mileage', 'Make', 'State']] #Model emitted as there are thousands of models and XGB will not converge with that many features
Y = train.Price
X = pd.get_dummies(data=X)


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)


# XGBoost Regression with categorical data converted using the one hot encoding technique.

# In[ ]:


#gradient booster with one hot conversion
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='ls', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# As we can see, this training takes a very long time. It also isn't very accurate and the variance score is quite bad. We can definitely do better.

# The data contains categorical features like the City, State, Model, and Make information. While it's possible to use one hot encoding to convert it, this causes dimensionality problems as the number of features increases dramatically. So what we did was to group the training data by each of those categorical data separately and computing average of prices, then replacing the categorical data with the average computed. This maintains the correlation between the categorical data and the price as what we care about the most is the price.

# In[3]:


data = pd.read_csv("../input/carpredictiondata/dataProc.csv") #Load the processed data. Categorical features are converted to numerical ones
                                                              #, and outliers are removed


# In[ ]:


#plotting corr of the data we plan to use
visData = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum', 'Price']] 
corr = visData.corr()
sns.heatmap(corr)
print (corr['Price'].sort_values(ascending=False), '\n')


# In[14]:


X = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum']]
Y = data.Price

#Split into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)
X_test.info


# KNN Regression

# In[11]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))

# Uncomment the next 2 lines to produce a model file that you can use later
#from sklearn.externals import joblib
#joblib.dump(knn, 'model.pkl')


# Decision Tree Regression

# In[9]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)


plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# Linear Regression

# In[10]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

predicted = regr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# XGBoost Regression

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# One thing we notice about the graphs is the format of the error is very cylindrical. We looked for ways to reduce that and decided to look at the graphs of the Price and the negative correlated features (Mileage)

# In[ ]:


sns.distplot(data.Price)
sns.distplot(data.Mileage)


# The graphs show that the data is very skewed. We can simply use the log function to convert it into a more uniform distribution

# In[39]:


X = data[['Year', 'Mileage', 'MakeNum', 'StateNum', 'ModelNum']] #CityNum is state, misnamed in data
Y = np.log(data.Price)
X['Mileage'] = np.log(X['Mileage'])
X['Mileage'] = 0.9 * X['Mileage'] + 0.1

#Split into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 42)


# In[40]:


sns.distplot(Y)
sns.distplot(X['Mileage'])


# KNN Regression after normalization

# In[42]:


from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train, Y_train)

predicted = knn.predict(X_test)
err = Y_test - predicted

p2 = knn.predict(X_train)
err2 = Y_train - p2

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# In[45]:


#Compare error of training data vs error of test data. We need to get the exp of the data since we used log before
xperr = np.exp(Y_test) - np.exp(predicted)
xperr2 = np.exp(Y_train) - np.exp(p2)
sns.distplot(xperr)
sns.distplot(xperr2)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_train, p2))


# XGBoost

# In[48]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
err = Y_test - predicted

plt.scatter(Y_test, err , color ='teal')
plt.xlabel('Actual Price',fontsize=25)
plt.ylabel('Error',fontsize=25)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)

from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))


# As we can see, our score improved after normalization as well. 
