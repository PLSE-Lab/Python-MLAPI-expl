#!/usr/bin/env python
# coding: utf-8

# # Airbnb Listings  PRICE ESTIMATOR
# 
# This notebook predicts and estimates airbnb listing price with the giving boston dataset
# 
# 
# Acknowledgement
# This dataset is part of Airbnb Inside, and the original source can be found [here](http://insideairbnb.com/get-the-data.html).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns;
from tensorflow import  keras
plt.style.use("seaborn-colorblind")
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load the Tensorboard notebook extension
get_ipython().run_line_magic('reload_ext', 'tensorboard')

get_ipython().run_line_magic('tensorboard', '--logdir logs')

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")


# Read select features and columns from the datasets using pandas

# In[ ]:


used_features = ['property_type','room_type','bathrooms','bedrooms','beds','bed_type','accommodates','host_total_listings_count'
                ,'number_of_reviews','review_scores_value','neighbourhood_cleansed','cleaning_fee','minimum_nights','security_deposit','host_is_superhost',
                 'instant_bookable', 'price']

boston = pd.read_csv("/kaggle/input/boston/listings.csv", usecols= used_features)

boston.shape
boston.head(10)  


# ### Start Data Cleaning and Preprocessing
# 

# Remove all '$' signs, converts the numbers to floats and fill the missing numeric data with the average or the most frequent for non numeric data.

# In[ ]:


for feature in ["cleaning_fee", "security_deposit", "price"]:
    boston[feature] = boston[feature].map(lambda x: x.replace("$",'').replace(",",''), na_action = 'ignore')
    boston[feature] = boston[feature].astype(float)
    boston[feature].fillna(boston[feature].median(), inplace = True)

for feature in ["bathrooms", "bedrooms", "beds", "review_scores_value"]:
    boston[feature].fillna(boston[feature].median(), inplace = True)

boston['property_type'].fillna('Apartment', inplace = True)    

boston.head(5)  


# Use histogram to find the distribution for the target variable or price

# In[ ]:


boston["price"].plot(kind = 'hist', grid = True)
plt.title("Price histogram before subsetting and log-transformation")

boston['price'].skew()


# Since most of the price ranges fall mainly  below 500, let's select and focus on that range

# In[ ]:


boston = boston[(boston['price'] > 50) & (boston['price'] < 500)]
target = np.log(boston.price)
target.hist()
plt.title("Price distribution after the subsetting and log-transformation")

# select the features-the independent variables- and drop the price- target/dependent variable
features = boston.drop('price', axis = 1)
features.head()


# Find the correlation Coefficients for select variables

# In[ ]:


select_features = features.copy()
select_features = select_features[['accommodates','bathrooms','bedrooms','beds','security_deposit','cleaning_fee','minimum_nights','number_of_reviews','review_scores_value']]
select_features['price'] = boston.price
select_features.corr()


# In[ ]:


select_features.corr()['price'].sort_values()


# Use RegPlot to find how **'accomadates'** correlates with price.

# In[ ]:


sns.regplot(x = "accommodates", y = "price", data = new_features)


# Change the **categorical 'instant_bookable' and 'host_is_superhost'** varibale to numerical binary

# In[ ]:


copied_features = features.copy()
copied_features['instant_bookable'].unique();
copied_features['instant_bookable'].replace(to_replace =['f','t'], value = [0,1], inplace =True);
copied_features['host_is_superhost'].replace(to_replace =['f','t'], value = [0,1], inplace =True);
copied_features.head()


# Convert '**bed_type, neighbourhood_cleansed, property_type,room_type**, etc' to one-hot encoding.

# In[ ]:


print(features['bed_type'].unique())
print(features['neighbourhood_cleansed'].unique())
print(features['property_type'].unique())
print(features['room_type'].unique())

copied_features = pd.get_dummies(copied_features, prefix ='', prefix_sep = '')
copied_features.tail()


# In[ ]:


#Let's quickly create a simple Price Estimator Model model using LinearRegression.
#Assign the copied_features to X
X = copied_features;
Y = target


# View the final values we are going to model.

# In[ ]:


X.head()


# And Y

# In[ ]:


Y[0:5]


# In[ ]:


#Find out if there are any 'NANs' values


# In[ ]:


cols = ['accommodates','bathrooms','bedrooms','beds','security_deposit','cleaning_fee','minimum_nights','number_of_reviews','review_scores_value']
for col in cols:
    print("number of NaN values for the column",col," is ",boston[col].isna().sum())


# #### **First, I will use normal and rudimentary sklearn library before using TensorFlow.**

# In[ ]:


#Import Linear Ligression,PolynomialFeatures as well as Pipeline and StandardScalar from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score;


# Fit, Train and Evaluate the Model Using PolynomialFeatures 

# In[ ]:


# Use PipeLine Input
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input);
pipe.fit(X, Y)
pipe.score(X,Y)


# From the above, one can see the score is ~80%, though the whole and the same dataset is used for both training and testing. I will change this very soon.

# Split the data into train and test sets, 

# In[ ]:


x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state =42)
print("Size of x_train==",x_train.shape," that of y_test ==",y_test.shape)


# Initialize the LinearRegression, Fit/Train the Model, Predict and Evaluate the MSE and R^2

# In[ ]:


#Get an instance of LinearRegression
lr = LinearRegression()
# Let's partition the data

#Fit our model
lr.fit(x_train,y_train)

yhat = lr.predict(x_test)
#Calculate the score
print("r2 score: ",r2_score(y_test, yhat))
print("MSE: ",mean_squared_error(y_test, yhat))
yhat = np.exp(yhat);
y_test = np.exp(y_test);
print("predicted yhat ==",yhat[0:5])
print("real y ==",y_test[0:5])


# From the above Eavluations Results: MSE = 0.093, R^2 = 0.65 ~ 65%

# Compare Graphically-using scatter plot- the Results of the **Predicted** Prices versus the **Actual** Prices

# In[ ]:


fig = plt.figure(figsize =(8, 10))
a = plt.axes(aspect = 'equal')
plt.scatter(y_test,yhat,color='blue')
#plt.plot(x_test,yhat,'r')
plt.xlabel("Real Values [Prices]")
plt.ylabel("Predicted Values [Prices]")
plt.show()


# In[ ]:


np.exp(Y).max()


# **REPEAT THE LAST PROCESS USING TENSOR FLOW INSTEAD OF sklearn**

# Split the the dataset into train and test sets. Then separate the categorical from the numerical variables and merge the two eventually.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size =0.33, random_state = 42)

numeric_columns = ['host_total_listings_count','accommodates','bathrooms','bedrooms','beds',
 'security_deposit','cleaning_fee','minimum_nights','number_of_reviews','review_scores_value']

# Get all the categorical feature names that contains strings
categorical_columns = ['host_is_superhost','neighbourhood_cleansed','property_type','room_type','bed_type','instant_bookable']

numeric_features = [tf.feature_column.numeric_column(key = column) for column in numeric_columns]
#print(numeric_features)

categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(key = column,
                        vocabulary_list = features[column].unique())
                        
                        for column in categorical_columns]

print(categorical_features[3])

linear_features = numeric_features + categorical_features

print(linear_features)


# Define the input functions, train, evaluate and predict.

# In[ ]:


def make_input_fn(data_df,label_df,num_epochs=10, shuffle=True,batch_size = 32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size = 32).repeat(num_epochs) # split dataset into batches of 32 and repeat for the number of epochs
    return ds # return a batch of the dataset
  return input_function # return a function object for use

training_input_fn = make_input_fn( data_df = x_train,
                                        label_df = y_train,
                                        batch_size = 32,
                                        shuffle = True,
                                        num_epochs = None
                                      )
eval_input_fn = make_input_fn(data_df = x_test,
                              label_df  = y_test,
                              batch_size = 32,
                              shuffle = False,
                              num_epochs = 1
                             );

linear_regressor = tf.estimator.LinearRegressor(feature_columns = linear_features, model_dir = "linear_regressor")

linear_regressor.train(input_fn = training_input_fn, steps = 2000)

#Evaluate the model

result = linear_regressor.evaluate(input_fn = eval_input_fn)

print(result)
print("Loss is "+ str(result['loss']))

pred = list(linear_regressor.predict(input_fn = eval_input_fn))

pred = [p['predictions'][0] for p in pred]

prices = np.exp(pred)
print(prices)


# In[ ]:


print("Real Value Max ==", np.exp(y_test).max())
print("Predicted Value Max ==", prices.max())


# Compare Graphically the Results of the **Predicted** Prices versus the **Actual** Prices

# In[ ]:


y_test = np.exp(y_test)
fig = plt.figure(figsize =(8, 10))
a = plt.axes(aspect = 'equal')
plt.scatter(y_test,prices,color='blue')
#plt.plot(x_test,yhat,'r')
plt.xlabel("Real Values [Prices]")
plt.ylabel("Predicted Values [Prices]")
plt.show()


# In[ ]:


boston['price']=np.exp(target)
boston['price'].describe()

