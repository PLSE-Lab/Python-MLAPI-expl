#!/usr/bin/env python
# coding: utf-8

# # This is a basic Starter Kernel for the New York City Taxi Fare Prediction Playground Competition 
# Here we'll use a simple linear model based on the travel vector from the taxi's pickup location to dropoff location which predicts the `fare_amount` of each ride.
# 
# This kernel uses some `pandas` and mostly `numpy` for the critical work.  There are many higher-level libraries you could use instead, for example `sklearn` or `statsmodels`.  

# In[ ]:


# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir('../input'))


# ### Setup training data
# First let's read in our training data.  Kernels do not yet support enough memory to load the whole dataset at once, at least using `pd.read_csv`.  The entire dataset is about 55M rows, so we're skipping a good portion of the data, but it's certainly possible to build a model using all the data.

# In[ ]:


# df = pd.DataFrame(index=pd.DatetimeIndex(start=dt.datetime(2016,1,1,0,0,1),
#     end=dt.datetime(2016,1,2,0,0,1), freq='H'))\
#     .reset_index().rename(columns={'index':'datetime'})

# df.head()
# df.dtypes


# In[ ]:


# df['ts'] = df.datetime.values.astype(np.int64) // 10 ** 9
# df.head()
# df.dtypes
# # t = pd.Timestamp(2017, 1, 1, 12)
# # ts = t.astye


# In[ ]:


train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_00)
train_df.dtypes


# In[ ]:


#Reverse GeoCoding
# import pygeocoder
# from pygeocoder import Geocoder


# Let's create two new features in our training set representing the "travel vector" between the start and end points of the taxi ride, in both longitude and latitude coordinates.  We'll take the absolute value since we're only interested in distance traveled. Use a helper function since we'll want to do the same thing for the test set later.

# In[ ]:


#striping the timezone
def convert_to_datetime(df):
    test_time = df['pickup_datetime'].astype(str).str[:-4]
    df['date_time'] =  pd.to_datetime(test_time, format='%Y%m%d %H:%M:%S')
    return df
    
# converting the object to date time
train_df = convert_to_datetime(train_df)
train_df.head()


# Lets describe the df.

# In[ ]:


train_df.describe()


# [](http://)Min value of the fare_amount is less than zero and min passenger count is zero. We should discard those values.

# In[ ]:


def normalize_fare_passenger(df):
    if 'fare_amount' in df.columns:
        print("old lenght: %d" %len(df))
        df = df[df.fare_amount>0]
    print("length after fare_amount normalization: %d" %len(df))
    df = df[df.passenger_count>0]
    print("length after passenger_count normalization: %d" %len(df))
    return df

train_df = normalize_fare_passenger(train_df)
train_df.head()


# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    return df

train_df = add_travel_vector_features(train_df)


# ### Explore and prune outliers
# First let's see if there are any `NaN`s in the dataset.

# In[ ]:


print(train_df.isnull().sum())


# There are a small amount, so let's remove them from the dataset.

# In[ ]:


print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))
train_df.head()


# Now let's quickly plot a subset of our travel vector features to see its distribution.

# In[ ]:


plot = train_df.iloc[1:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# We expect most of these values to be very small (likely between 0 and 1) since it should all be differences between GPS coordinates within one city.  For reference, one degree of latitude is about 69 miles.  However, we can see the dataset has extreme values which do not make sense.  Let's remove those values from our training set. Based on the scatterplot, it looks like we can safely exclude values above 5 (though remember the scatterplot is only showing the first 2000 rows...)

# In[ ]:


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))


# Lets draw some histogram to get an idea about the fare_amount range

# In[ ]:


train_df[train_df.fare_amount<60].fare_amount.hist(bins=200, figsize=(14,3))
plt.xlabel('fare $USD')
plt.title('Histogram');


# Now fare_amout should be directly related to the distance covered in the trip. Lets try to find out the distance covered in each trip
# 

# In[ ]:


train_df.dtypes


# In[ ]:


#this is a kind of haversine formula to calculate the spherical distance
def haversine_distance_calculation(df):
    lon1, lat1, lon2, lat2 = df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6367 * c
    df['distance'] = distance
    return df
train_df = haversine_distance_calculation(train_df)
train_df.head()


# Now lets try to plot the relation between distance and fare_amoutn

# In[ ]:


train_df.plot(x='distance', y='fare_amount', style = 'o')


# Now we are seeing more outliers. Like distance more than 40km but fare_amount is very less and also distance very less but fare_amount is high. We should prune thoes also. So we will round up the distance within 50km and fare within 100 USD

# In[ ]:


def distance_fare_normalization(df):
    print("old lenght with distance greated than 50km: %d" %len(df))
    df = df[df.distance<50]
    print("length after distance normalization: %d" %len(df))
    if 'fare_amount' in df.columns:
        df = df[df.fare_amount<100]
        print("length after fare_amount normalization: %d" %len(df))
    return df

train_df = distance_fare_normalization(train_df)
train_df.head()


# Now lets check the graph again

# In[ ]:


train_df.plot(x='distance', y='fare_amount', style = 'o')


# Now we can try to find any relation between time and fare_amount. We can find the day of year and day of week from date_time and correlate them with. Lets try.

# In[ ]:


def day_converter(df):
    day_of_year = df['date_time'].dt.dayofyear
    day_of_week = df['date_time'].dt.dayofweek
    hour_of_day = df['date_time'].dt.hour
    df['day_of_year'] = day_of_year
    df['day_of_week'] = day_of_week
    df['hour_of_day'] = hour_of_day
    return df

    
train_df = day_converter(train_df)
train_df.plot(x='day_of_year', y='fare_amount', style = 'o')
train_df.head()


# ### Train our model
# Our model will take the form $X \cdot w = y$ where $X$ is a matrix of input features, and $y$ is a column of the target variable, `fare_amount`, for each row. The weight column $w$ is what we will "learn".
# 
# First let's setup our input matrix $X$ and target column $y$ from our training set.  The matrix $X$ should consist of the two GPS coordinate differences, plus a third term of 1 to allow the model to learn a constant bias term.  The column $y$ should consist of the target `fare_amount` values.

# In[ ]:


# Construct and return an Nx3 input matrix for our linear model
# using the travel vector, plus a 1.0 for a constant bias term.
##MH:adding pickup_datetime in train_df
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, df.distance, df.day_of_year, df.day_of_week,df.hour_of_day, np.ones(len(df))))

train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)


# Now let's use `numpy`'s `lstsq` library function to find the optimal weight column $w$.

# In[ ]:


# The lstsq function returns several things, and we only care about the actual weight vector w.
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)


# These weights pass a quick sanity check, since we'd expect the first two values -- the weights for the absolute longitude and latitude differences -- to be positive, as more distance should imply a higher fare, and we'd expect the bias term to loosely represent the cost of a very short ride.
# 
# Sidenote:  we can actually calculate the weight column $w$ directly using the [Ordinary Least Squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) method:
# $w = (X^T \cdot X)^{-1} \cdot X^T \cdot y$

# In[ ]:


w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)


# * Using a neural network with tensorflowframework

# In[ ]:


import tensorflow as tf
import math
import matplotlib.pyplot as plt
import h5py


#Creating placeholders
x = tf.placeholder(tf.float32, [7, None])
y = tf.placeholder(tf.float32, [1, None])

#defining hyper params
learning_rate = 0.0001
num_epochs = 2000
minibatch_size = 1024

#defining trainabl variables
W1 = tf.Variable(tf.random_normal([10, 7], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([10,1]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([6, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([6,1]), name='b2')

W3 = tf.Variable(tf.random_normal([4, 6], stddev=0.03), name='W3')
b3 = tf.Variable(tf.random_normal([4,1]), name='b3')

parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3
             }

#transposing to adjust size
train_X = np.transpose(train_X)
train_y = np.transpose(train_y)


print(np.shape(train_X))
print(np.shape(train_y))


# In[ ]:



def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
#     print(m)
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
#     print(np.shape(permutation))
    #print(permutation)
    shuffled_X = X[:, permutation]
#     print(np.shape(shuffled_X))
    shuffled_Y = Y[permutation].reshape((1,m))
#     print(np.shape(shuffled_Y))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
#     print(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k*mini_batch_size):((k+1)*mini_batch_size)]
        mini_batch_Y = shuffled_Y[:, (k*mini_batch_size):((k+1)*mini_batch_size)]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,(num_complete_minibatches*mini_batch_size):m]
        mini_batch_Y = shuffled_Y[:,(num_complete_minibatches*mini_batch_size):m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# minibatches = random_mini_batches(train_X,train_y,minibatch_size)
# for minibatch in minibatches:
#     (minibatch_X, minibatch_Y) = minibatch
#     print(np.shape(minibatch_X))
#     print(np.shape(minibatch_Y))


# In[ ]:


# now declare the weights connecting the input to the hidden layer

# calculate the output of the hidden layer
def forward_propagation(X, W1,b1,W2,b2, W3, b3):
    Z1 = tf.add(tf.matmul(W1, X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  
#     print(np.shape(A1))
#     print(np.shape(W2))
    Z2 = tf.add(tf.matmul(W2, A1),b2) 
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    return Z3

def compute_cost(Y_h, Y):
    logits = tf.transpose(Y_h)
    labels = tf.transpose(Y)
#     logits = Y_h
#     labels = Y
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels = labels))
    ### END CODE HERE ###    
    return cost



# finally setup the initialisation operator
# init_op = tf.global_variables_initializer()


# In[ ]:


# start the session
costs =[]
Z3 = forward_propagation(x, W1,b1,W2,b2, W3, b3)
S = (tf.nn.l2_loss(W1) +tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))*0.01
cost = compute_cost(Z3, y) + S
m = train_X.shape[1]
# m = train_X.shape[1]
#Defining optimizer
optimizer =tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
init = tf.global_variables_initializer()
print_cost = True

# print(minibatches)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(train_X, train_y, minibatch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch                
                _ , minibatch_cost = sess.run([optimizer, cost],feed_dict={x:minibatch_X,y:minibatch_Y}) #here x,y are the defined placeholders
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 500 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
            
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # lets save the parameters in a variable
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")


# In[ ]:





# ### Make predictions on the test set
# Now let's load up our test inputs and predict the `fare_amount`s for them using our learned weights!

# 

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.dtypes


# In[ ]:


def predict(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z3 = forward_propagate(X,W1,b1,W2,b2,W3,b3)
    out = tf.nn.sigmoid(Z3)
    return out
    


# In[ ]:


# Reuse the above helper functions to add our features and generate the input matrix.
add_travel_vector_features(test_df)
#
##converting to date time
test_df = convert_to_datetime(test_df)
##Normalizing fare_amount>0 & passenger_count>0
test_df = normalize_fare_passenger(test_df)
## dropping null values
test_df = test_df.dropna(how = 'any', axis = 'rows')
test_df =haversine_distance_calculation(test_df)
##normalization distance>50km and fare>100usd
test_df = distance_fare_normalization(test_df)
#converting date time to day vactor
test_df = day_converter(test_df)

test_X = get_input_matrix(test_df)
# Predict fare_amount on the test set using our model (w) trained on the training set.
# test_y_predictions = np.matmul(test_X, w).round(decimals = 2)
test_y_predictions = predict(test_X, parameters).round(decimals = 2)

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))


# ## Ideas for Improvement
# The output here will score an RMSE of $5.74, but you can do better than that!  Here are some suggestions:
# 
# * Use more columns from the input data.  Here we're only using the start/end GPS points from columns `[pickup|dropoff]_[latitude|longitude]`.  Try to see if the other columns -- `pickup_datetime` and `passenger_count` -- can help improve your results.
# * Use absolute location data rather than relative.  Here we're only looking at the difference between the start and end points, but maybe the actual values -- indicating where in NYC the taxi is traveling -- would be useful.
# * Use a non-linear model to capture more intricacies within the data.
# * Try to find more outliers to prune, or construct useful feature crosses.
# * Use the entire dataset -- here we're only using about 20% of the training data!

# Special thanks to Dan Becker, Will Cukierski, and Julia Elliot for reviewing this Kernel and providing suggestions!
