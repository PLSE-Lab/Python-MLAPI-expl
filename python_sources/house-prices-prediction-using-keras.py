#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:


# Read in train data
df_train = pd.read_csv('../input/train.csv', index_col=0)


# In[3]:


df_train.head()


# # Prepare data
#     Investigate what data that has a linear or some kind of relation to the sale price
#     Drop the unimportant features or less unimportant features
#     Drop features which has many NaN values

# In[4]:


#descriptive statistics summary
df_train['SalePrice'].describe()


# In[5]:


#histogram
sns.distplot(df_train['SalePrice']);


# In[6]:


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


#     - Skewness means the top of the iceberg is not in the middle but rather towards left or right.
#     - Kurtosis describe if the gaussian distrubution is very small and narrow or very wide

# Use a heatmap to see which features have strongest correlation with house price

# In[7]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# Here we can detect multicollinearity for example basement area and the area of the first floor so these hold more or less the same kind of data. The same goes for garage variables, for example if you have a big garage you also have more cars in it.
# 
# Some variables are also important for the SalePrice with the biggest one being OverallQual
# 
# Let's plot top 10 most important for correlating with SalePrice

# In[8]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# From this plot we can draw the conclusion that:
#     - OverallQual is important
#     - GrLivArea is also important
#     - TotalBsmtSF is important
#     - GarageCars and GarageArea are two important features but we drop GarageArea since it is more or less the same information as GarageCars
#     - TotalBsmtSF and 1stFlrSF are also more or less the same so we drop 1StFlrSF
#     - TotRmsAbvGrd and GrLivArea are also strongly correlated to let's drop TotRmsAbvGrd
#  
#  Let's scatterplot these important features.

# In[9]:


#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# The basement area and total living area seems to have similarities their saleprice plot looks almost identical, let's drop basement area.
# 
# Maybe also remove year built data since this data can be tricky to use.

# Let's have a  look at the missing data.
# 
# Let's display a % of the data that is missing from some columns.

# In[10]:


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# Some of theese features are of interest for us and they don't show a massive shortage of data so lets create mean data for those values.

# In[11]:


df_train = df_train.fillna(df_train.mean())


# Now let's remove outliers for example data that doesn't match what we expect like an insane price for a house
# 
# To do this we standardize the data so that the mean is 0 and a standard deviation of 1. 

# In[12]:


#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


#     -Values that are similar to each other stay close to 0
#     -Values that are a bit odd get high values such as the 7 values.

# In[13]:


#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# What has been revealed:
# 
# * The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers and delete them.
# * The two observations in the top of the plot are those 7 something observations that we said we should be careful about. They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.

# # Prepare data
# Right now I think we have an idea of what kind of data we are interested in and what data we don't think are useful for us. Let's build a pipeline for removing the data.

# Let's reload the data so we can have a fresh start!

# In[14]:


df_train = pd.read_csv('../input/train.csv')


# Let's not log the data since a neural network is quite good at working with non-linear data. I also tested and verified that the model didn't perform better or worse if I logged the data before hand.

# In[15]:


cols = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
df_train = df_train[cols]
# Create dummy values
df_train = pd.get_dummies(df_train)
#filling NA's with the mean of the column:
df_train = df_train.fillna(df_train.mean())
# Always standard scale the data before using NN
scale = StandardScaler()
X_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']]
X_train = scale.fit_transform(X_train)
# Y is just the 'SalePrice' column
y = df_train['SalePrice'].values
seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33, random_state=seed)


# In[16]:


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mae])
    return model


# In[ ]:


model = create_model()
model.summary()


# In[ ]:


history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=32)


# Let's investigate how well this model did!

# In[ ]:


# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# This result is not very good and gives us a mean absolute error just above 20000 dollars. I beleive this model performs bad due to the fact that we have a quite small data-set becuase a neural network performs the best when having a big dataset. 

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
id_col = df_test['Id'].values.tolist()
df_test['GrLivArea'] = np.log1p(df_test['GrLivArea'])
df_test = pd.get_dummies(df_test)
df_test = df_test.fillna(df_test.mean())
X_test = df_test[cols].values
# Always standard scale the data before using NN
scale = StandardScaler()
X_test = scale.fit_transform(X_test)


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = id_col
submission['SalePrice'] = prediction


# In[ ]:


submission.to_csv('submission.csv', index=False)


# **Sources of information**
# 
# [Comprehensive data exploration with Python
# ](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# 
# *Can recommend this notebook it is a fun and informative read*

# In[ ]:




