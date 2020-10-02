#!/usr/bin/env python
# coding: utf-8

# # Project - Simple Avocado Price Prediction! 
# 
# In this project, I will create a __linear regression model__ and a __one hidden layer NN__ on estimating the function that represents the price of Avocados based on historical data. This should be done using only numpy and basic python - i.e not using higher-level packages. Basic machine learning consideration when preprocessing and handling data need to be taken in consideration. Lines of code should be commented thoroughly to show understanding.
# 
# 
# 

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Load and prepare the data

# In[14]:


df = pd.read_csv('../input/avocado_prices/avocado.csv')


# In[13]:


data_path = '../input/avocado_prices/avocado.csv'

avocado = pd.read_csv(data_path)


# In[ ]:


avocado.head()


# In[ ]:


### sort the date format
avocado['Date'] = pd.to_datetime(avocado['Date'])


# In[ ]:


''' sort the date in ascending order '''
avocado = avocado.sort_values(by='Date')


# In[ ]:


avocado.dtypes # check the data types


# In[ ]:


avocado[avocado.isnull().any(axis=1)] # clean the null values


# That is great, no null values, I will skip the process of detect the outliers, it will take too long. But it is easy to implement with anomalous detection algorithms (like gaussian kernel). 

# ### Statistics of Data
# We show the statistics of the data below: 

# In[ ]:


avocado.describe()


# In[ ]:


import plotly
import plotly.offline as py
import plotly.graph_objs as go

Type=avocado.groupby('type')['Total Volume'].agg('sum')

values=[Type['conventional'],Type['organic']]
labels=['conventional','organic']

trace=go.Pie(labels=labels,values=values)
py.iplot([trace])


# In[ ]:


import seaborn as sns
sns.set(font_scale=1.5) 
from scipy.stats import norm
fig, ax = plt.subplots(figsize=(15, 9))
sns.distplot(a=avocado.AveragePrice, kde=False, fit=norm)


# We can now see that our price data is almost like a skewed normal distribution. 

# ### Categorical values:
# We are now going to dealing with the categorical values, one hot encoding or other techniques up to our choice. 

# In[ ]:


pd.get_dummies(avocado, columns=["type", "region"], prefix=["type", "region"]).head()


# In[ ]:


pd.get_dummies(avocado, columns=["type"], prefix=["type"]).tail()


# In[ ]:


avocado["type"].value_counts()


# There are only two types, great! We can make a binary feature, or we can choose which type to run our model on.

# In[ ]:


#avocado["type_binary"] = np.where(avocado["type"].str.contains("conventional"), 1, 0)


# In[ ]:


#avocado.head()


# In[ ]:


avocado["region"].value_counts() # We have too many regions, one-hot encoding may be expensive here


# Because all the regions are within US, it is reasonable to drop the `TotalUS` region which is obsolete. 

# In[ ]:


avocado = avocado[avocado.region != 'TotalUS']


# In[ ]:


avocado["Date"].value_counts() # We have too many regions, one-hot encoding may be expensive
# here. In fact, backward difference encoding might works well? 


# In[ ]:


avocado["year"].value_counts() # The year of Avocados


# There are four years. 

# In[ ]:


### We can also choose to predict the values for specific region: 

#avocado['Date'] = pd.to_datetime(avocado['Date'])
#regions = avocado.groupby(avocado.region)
#PREDICTING_FOR = "TotalUS"
#date_price = regions.get_group(PREDICTING_FOR).reset_index(drop=True)
#date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice','Total Volume']].reset_index(drop=True)


# In[ ]:


### Box plot
sns.boxplot(y="type", x="AveragePrice", data=avocado, palette = 'pink')


# From above Figure we can see that the Organic avocado are generally more expensive (mean price) than the conventional avocados. 

# In[ ]:


mask = avocado['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=avocado[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )


# The average price of conventional avocado in Chicago during 2017 and 2018 reaches its peak of all time in our data, and its due to shortage of storage in 2017 US.

# In[ ]:


### Correlation matrix
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
dicts = {}
avocado_copy = avocado
label.fit(avocado.type.drop_duplicates()) 
dicts['type'] = list(label.classes_)
avocado_copy.type = label.transform(avocado_copy.type) 


# In[ ]:


cols = ['AveragePrice','type','year','Total Volume','Total Bags']
cm = np.corrcoef(avocado_copy[cols].values.T)
sns.set(font_scale = 1.7)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)


# From the above matrix, we see strong positive relationship between price and type of avocado. We also observed that strong positive relationship between total bags and total volume of Avocado. Logistic regression can be used to predict the type of avocado and softmax function with DNN for classified the regions of avocado. We might consider to reduce the dimensions of our data if two features are strong correlated. 

# In[ ]:


cols = ['Total Bags','XLarge Bags','Large Bags','Small Bags']
f,ax2=plt.subplots(figsize=(10,10))
sns.heatmap(avocado_copy[cols].corr(),annot=True, linewidth=.5, fmt='.1f',ax=ax2)
plt.show()


# The total bags is most likely affected by large bags. 

# ### Conventional price data: 

# In[ ]:


avocado = avocado[avocado.region != 'TotalUS']


# In[ ]:


avocado["type_binary"] = np.where(avocado["type"].str.contains("conventional"), 1, 0)


# In[ ]:


avocado['Date'] = pd.to_datetime(avocado['Date'])
regions = avocado.groupby(avocado.type)
PREDICTING_FOR = "conventional"
organic_price = regions.get_group(PREDICTING_FOR).reset_index(drop=True)


# In[ ]:


organic_price["type"].value_counts()


# In[ ]:


organic_price.head()


# In[ ]:


# ''' method 1: oridinal region data'''
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# organic_price['region'] = le.fit_transform(organic_price['region'])

# ''' We need to drop all the dummy fields, and one-hot encoding some features'''
# dummy_field = ['year']
# dummies = pd.get_dummies(organic_price, columns=dummy_field, prefix=dummy_field, drop_first=False)
# fields_to_drop = ['type', 'type_binary', 'Date','Unnamed: 0']
# #fields_to_drop = ['region', 'type', 'Date','Unnamed: 0']
# data_short = dummies.drop(fields_to_drop, axis=1)
# data_short.head()


# In[ ]:


''' method 2: We need to drop all the dummy fields, and one-hot encoding some features'''
dummy_field = ['year','region']
dummies = pd.get_dummies(organic_price, columns=dummy_field, prefix=dummy_field, drop_first=False)
#fields_to_drop = ['type', 'type_binary', 'Date','Unnamed: 0']
fields_to_drop = ['type', 'type_binary', 'Date','Unnamed: 0']
data_short = dummies.drop(fields_to_drop, axis=1)
data_short.head()


# ### Total data encoded and numericalized

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
avocado['region'] = le.fit_transform(avocado['region'])
avocado.head()


# In[ ]:


''' We need to drop all the dummy fields, and one-hot encoding some features'''
dummy_field = ['year']
dummies = pd.get_dummies(avocado, columns=dummy_field, prefix=dummy_field, drop_first=False)
fields_to_drop = ['type', 'Date','Unnamed: 0']
#fields_to_drop = ['type', 'Date','Unnamed: 0']
data = dummies.drop(fields_to_drop, axis=1)
data.head()


# Does the region of the Avocados really matters? Yes, it matters. 

# ## Checking out the data
# 
# This dataset has the average prices for avocado sales for quantities from 2015 to 2018. The avocado is classified as `conventional`  and `organic`, it can also be distinguished between different regions within the `US`.  

# In[ ]:


types = avocado.groupby(avocado.type)
PREDICTING_FOR = 'conventional'
prices = types.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)


# In[ ]:


prices[1:500].plot(x='Date', y='AveragePrice', kind="line")


# ### Scaling target variables
# To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
# 
# The scaling factors are saved so we can go backwards when we use the network for predictions.

# In[ ]:


data = data.rename(index=str, columns={"Total Volume": "t_volume", "Total Bags": "t_bags", 
                                "Small Bags": "s_bags", "Large Bags": "l_bags", "XLarge Bags": "xl_bags"});


# In[ ]:


data.head()


# In[ ]:


''' Mean normalization features'''

quant_features = ['AveragePrice', 't_volume','4046', '4225', '4770', 't_bags', 'region', 
                  's_bags', 'l_bags', 'xl_bags']
#quant_features = ['AveragePrice', 't_volume','4046', '4225', '4770', 't_bags', 'region']
#quant_features = ['t_volume','4046', '4225', '4770', 't_bags', 's_bags', 'l_bags',
#                  'xl_bags']

# Store scalings features in a dictionary, in case we need convert them back.
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
print("scaled_features: ", scaled_features)


# In[ ]:


data.head()


# ### Scaling target variable for chosen type of avocado data

# In[ ]:


data_short = data_short.rename(index=str, columns={"Total Volume": "t_volume", "Total Bags": "t_bags", 
                                "Small Bags": "s_bags", "Large Bags": "l_bags", "XLarge Bags": "xl_bags"});


# In[ ]:


''' Mean normalization features'''
#quant_features = ['AveragePrice', 't_volume','t_bags']
# quant_features = ['AveragePrice', 't_volume','4046', '4225', '4770', 't_bags', 's_bags', 'l_bags',
#                  'xl_bags']
quant_features = ['AveragePrice', 't_volume','4046', '4225', '4770', 't_bags', 's_bags', 'l_bags',
                 'xl_bags']

# Store scalings features in a dictionary, in case we need convert them back.
scaled_features = {}
for each in quant_features:
    mean, std = data_short[each].mean(), data_short[each].std()
    scaled_features[each] = [mean, std]
    data_short.loc[:, each] = (data_short[each] - mean)/std
print("scaled_features: ", scaled_features)


# In[ ]:


data_short.head()


# ### Splitting the data into training, testing, and validation sets
# 
# We'll save the data for the last approximately 2 percent of data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.

# In[ ]:


from sklearn.model_selection import train_test_split
shuffle = False
''' devide into train, cross validation and test sets'''
train_set, test_set = train_test_split(data, test_size=0.02,shuffle=shuffle)
# Separate the data into features and targets
target_fields = ['AveragePrice']
features, targets = train_set.drop(target_fields, axis=1), train_set[target_fields]
test_features, test_targets = test_set.drop(target_fields, axis=1), test_set[target_fields]
features.head()
print("number of features =", features.shape[1])

train_features, val_features = train_test_split(features, test_size=0.4,shuffle=shuffle)
train_targets, val_targets = train_test_split(targets, test_size=0.4,shuffle=shuffle)


# In[ ]:


train_set['AveragePrice'].head()


# Well, we will have 76 features in total if we encoded all categorical features, I am thinking about using PCA or other dimenstionality reduction techniques. However, it would took too long to implement them from scratch in Python. 

# In[ ]:


# check the size of examples
print(np.size(train_features))
print(np.size(test_features))
np.size(val_features)


# ### Splitting the data into training, test, and validation sets for type of avocado data
# Get the data set for conventional Avocado training data example

# In[ ]:


from sklearn.model_selection import train_test_split
shuffle = False
''' devide into train, cross validation and test sets'''
train_set, test_set = train_test_split(data_short, test_size=0.1, shuffle=shuffle)
# Separate the data into features and targets
target_fields = ['AveragePrice']
features, targets = train_set.drop(target_fields, axis=1), train_set[target_fields]
test_features, test_targets = test_set.drop(target_fields, axis=1), test_set[target_fields]
features.head()
print("number of features =", features.shape[1])

train_features, val_features = train_test_split(features, test_size=0.15, shuffle=shuffle)
train_targets, val_targets = train_test_split(targets, test_size=0.15, shuffle=shuffle)


# In[ ]:


# check the size of examples
print(np.size(train_features))
print(test_features.shape)
np.size(val_features)


# ## Build the linear regression: 

# ### Unit test: 
# We will test our linear regression solver on a two dimensional data example first, and compare it with results calculated from analytical method such as `normal equation`, (or we can validated its numerical gradient using `finite difference method`) and see how well it performs. 

# In[ ]:


from linearReg import LinearRegression


# In[ ]:


## create our 2D test data set: 
X= np.array([-15.9368,-29.1530,36.1895,37.4922,-48.0588,-8.9415,
             15.3078,-34.7063,1.3892,-44.3838,7.0135,22.7627])
Y=np.array([2.1343,1.1733, 34.3591,36.8380,2.8090,2.1211,14.7103,
            2.6142,3.7402,3.7317,7.6277,22.7524])
theta = np.array([1, 1]);


# In[ ]:


m = int(np.size(X[:]))


# In[ ]:


X_temp = np.concatenate((np.ones((m)), X.transpose()), axis=0).reshape(m,2, order='F');

examples = LinearRegression(X=X_temp, y=Y, theta = theta, lambdaa = 0)

J, theta = LinearRegression.linearRegCostFunction(examples, X=X_temp, y=Y, theta = theta, lambdaa = 0)
print('J=',J,'theta = ',theta)


# The correct returned cost function J at $\theta$ value should be: $303.993192$. The returned gradient of cost function value should be: $[-15.303016; 598.250744]$ according to the analytical solutions. 

# In[ ]:


#x=np.ones((5,1))
#y = np.ones((5,1))
#print(x-y)


# In[ ]:


#  Train linear regression with lambda = 0
lambdaa = 0;
#initial_theta = np.zeros(np.size(X, 2), 1); 
X_temp = np.concatenate((np.ones((m)), X.transpose()), axis=0).reshape(m,2, order='F');

examples = LinearRegression(X=X_temp, y=Y, theta = theta, lambdaa = lambdaa)
iterations = 4000;
alpha = 0.001;


#initial_theta = np.zeros((np.size(X_temp,1), 1)); 
initial_theta = np.array( np.zeros(np.size(X_temp,1)) )
#initial_theta = np.array( np.random.rand(np.size(X_temp,1)) )
#theta = np.array([1, 1]);

## Theta should be 2X2, with column be array of 1s, represents bias. 

#theta_temp = np.concatenate((np.ones((np.size(initial_theta))), initial_theta), axis=1);
theta, J_history = LinearRegression.gradientDescentMulti(examples, X=X_temp, y=Y, theta = initial_theta, 
                                      alpha = alpha,
                                      lambdaa = lambdaa, num_iters=iterations);


# In[ ]:


J_history


# The cost is decreasing. 

# In[ ]:


plt.plot(X,np.matmul(X_temp, theta))
plt.plot(X,Y,'o')


# Instead of gradient descent, we can implment other optimization algorithms such as BFGS, there are many packages we can employed. For problems with more examples than features, we can use analytical method like normal equation to solve the problem. 
# 
# For higher complexity problems with more data, we can try higher order features, (mapping original features to higher order spaces). Alternatively, neural networks will give better performance. 

# ### Analytical method of linear gression:

# In[ ]:


def normalEquation(X, y):

    m = int(np.size(X))

    theta = []

    # Bais of X1 values
    bias_vector = np.ones((m, 1))

    X = np.reshape(X, (m, 1))

    X = np.append(bias_vector, X, axis=1)

    X_transpose = np.transpose(X)

    # Calculating theta
    theta = np.linalg.pinv(np.matmul(X_transpose, X))
    theta = np.matmul(theta, X_transpose)
    theta = np.matmul(theta, y)

    return theta


# In[ ]:


theta = normalEquation(X, Y)


# In[ ]:


plt.plot(X,np.matmul(X_temp, theta))
plt.plot(X,Y,'o')


# ## Avocado example:

# ### Analytical solver:

# In[ ]:


def normalEquation(X, y):
    
    m = int(np.size(X[:, 0]))
    theta = []
    
    # Bais of X1 values
    bias_vector = np.ones((m, 1))

    #X = np.reshape(X, (m, 1))

    X = np.append(bias_vector, X, axis=1)

    X_transpose = np.transpose(X)

    # Calculating theta
    theta = np.linalg.pinv(np.matmul(X_transpose, X))
    theta = np.matmul(theta, X_transpose)
    theta = np.matmul(theta, y)
    
    #theta = (X'*X) \ X'*y; % the Analytical solution of linear regression.
    # theta = pinv(X' * X) * X' * y;

    # Calculating theta
    #theta = np.linalg.pinv(X_transpose.dot(X))
    #theta = theta.dot(X_transpose)
    #theta = theta.dot(y)

    return theta


# In[ ]:


print("number of features n:", train_features.shape[1])
print("number of examples m:", train_features.shape[0])


# In[ ]:


np.size(train_features.values) # the number of entries of tensor X. 


# In[ ]:


### Implement the training data and labels.
X, y = train_features.values, train_targets['AveragePrice'].values[:]
### get the analytical theta weight matrix
theta_analytics = normalEquation(X, y)


# ### Numerical solver: 

# In[ ]:


#  Train linear regression with lambda = 0
X, y = train_features.values, train_targets['AveragePrice'].values[:]
lambdaa = 0.05;
alpha = 0.1;
iterations = 3000;

total_no_feature = X[0,:].size+1;

m = int(np.size(X[:, 0]))
X_temp = np.concatenate((np.ones((m)).reshape(m,1), X), axis=1).reshape(m, total_no_feature, order='F');
y = y.reshape(m,1, order='F');

initial_theta = np.array( np.zeros(np.size(X_temp,1)) ).reshape(total_no_feature, 1, order='F');
avocado_example = LinearRegression(X=X_temp, y=y, theta = initial_theta, lambdaa = lambdaa)



theta, J_history = LinearRegression.gradientDescentMulti(avocado_example, X=X_temp, y=y, theta = initial_theta, 
                                      alpha = alpha,
                                      lambdaa = lambdaa, num_iters=iterations);


# In[ ]:


J_history


# In[ ]:


J, gg = LinearRegression.linearRegCostFunction(avocado_example, X_temp, y, theta, lambdaa)


# In[ ]:


print('Training error for numerical solution of linear regression =', J)


# In[ ]:


def MSE(y, Y):
    return np.mean((y-Y)**2)


# In[ ]:


X_val, y_val = val_features.values, val_targets['AveragePrice'].values[:]
total_no_feature = X_val[0,:].size+1;
m = int(np.size(X_val[:, 0]))
X_val_temp = np.concatenate((np.ones((m)).reshape(m,1), X_val), axis=1).reshape(m, total_no_feature, order='F');
y_val = y_val.reshape(m,1, order='F');
J, gg = LinearRegression.linearRegCostFunction(avocado_example, X_val_temp, y_val, theta, lambdaa)


# In[ ]:


print('Validation error for numerical solution of linear regression =', J)


# The validation error should be larger than training error as expected. 

# In[ ]:


m = int(np.size(X[:, 0]))
if(theta_analytics.shape == (theta_analytics.shape[0],)):
    theta_analytics = theta_analytics[:, np.newaxis]
X_temp = np.concatenate((np.ones((m)).reshape(m,1), X), axis=1).reshape(m, X[0,:].size+1, order='F');
J, gg = LinearRegression.linearRegCostFunction(avocado_example, X_temp, y, theta_analytics, 0)
#J, gg = LinearRegression.linearRegCostFunction(avocado_example, X_val_temp, y_val, theta_analytics.reshape(total_no_feature,1), 0)
print('Validation error for analytical solution (normal equation) =', J)


# In[ ]:


mean, std = scaled_features['AveragePrice']


# In[ ]:


### This is our prediction function for linear regression
def prediction(X, theta):
    h = np.matmul(X, theta);
    return h


# #### Linear regression performs on test data
# We will predict the future average prices for conventional avocado against the actual future prices in our test data set.

# In[ ]:


test_data = data_short[test_set.shape[0]:];
#test_data = data[test_set.shape[0]:];
X_test, y_test = test_features.values, test_targets['AveragePrice'].values[:]
total_no_feature = X_test[0,:].size+1;
m = int(np.size(X_test[:, 0]))
X_test_temp = np.concatenate((np.ones((m)).reshape(m,1), X_test), axis=1).reshape(m, total_no_feature, order='F');
y_test = y_test.reshape(m,1, order='F');

fig, ax = plt.subplots(figsize=(8,4))

predictions = prediction(X_test_temp, theta)*std + mean
y_test_plot = (y_test*std + mean)
predictions=predictions[:800]
y_test_plot=y_test_plot[:800]

ax.plot(predictions, label='Prediction')
ax.plot(y_test_plot, label='Data', LineStyle=':')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(avocado.ix[int(test_data.index[0]):len(y_test_plot)]['Date'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# Overall, concretely, the prediction is not so bad for simplistic linear model, it is somehow underfitted for data with more than 60 features.

# ## Simple one hidden layer DNN for avocado data 
# 
# Below I'll build a DN network from scratch without using pytorch and tensorflow ready to use package. I'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes and batch size. It should be noted that I do not implemented any SOA techniques in this case for simplicity.  
# 
# <img src="assets/neural_network.png" width=300px>
# 
# The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.
# 
# We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.
# 
# > **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation $y = x$. What is the slope of that equation? That is the derivative of $f(x)$.

# In[ ]:


#############
# In my_answers_NN.py file, I implemented the simple DNN. It is originally
# completed by myself as one of the projects with Udacity. I obtained the 
# high accuracy score on the original data example of bike sharing over few years. 
#############

from my_answers_NN import NeuralNetwork


# In[ ]:


''' the cost function, mean square error for regression analysis'''
def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## Unit tests
# 
# Run these unit tests to check the correctness of your network implementation. This will help you be sure your network was implemented correctly befor you starting trying to train it. 

# In[ ]:


import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'avocado.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(avocado, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)


# In[ ]:


import sys

####################
### Set the hyperparameters in you my_answers_NN.py file ###
####################

from my_answers_NN import iterations, learning_rate, hidden_nodes, output_nodes


N_i = train_features.shape[1]

network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=64)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['AveragePrice']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['AveragePrice'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['AveragePrice'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations))                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# In[ ]:


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()


# This is ok for a simple neural network with __only__ one hidden layer. It seems a little bit overfitting because the validation error is higher than the training error as shown. 

# In[ ]:


fig, ax = plt.subplots(figsize=(8,4))

#test_data = data[-int(test_set.index[0]):test_set.index[-1]];
test_data = data_short[int(test_set.index[0]):int(test_set.index[-1])+1];

mean, std = scaled_features['AveragePrice']
predictions = network.run(test_features).T*std + mean

ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['AveragePrice']*std + mean).values, label='Data', LineStyle=':')
ax.set_xlim(right=len(predictions))
ax.legend()

#dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = pd.to_datetime(avocado.ix[int(test_set.index[0]):int(test_set.index[-1])+1]['Date'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::50])
_ = ax.set_xticklabels(dates[12::50], rotation=45)


# In[ ]:


avocado[int(test_set.index[0]):int(test_set.index[-1])+1].tail()


# It is indeed overestimated the period between 03-Dec-17 to 18-Mar-18 due to insufficient data between these periods. 

# ### Further studies: 
# * Logistic regression on predicting the region of avocado based on its other features like Avg  - price and so. 
# * Binary classification on predicting the type of avocado whether it is conventional or organic avocado. 
# * Clustering problem, use K-means or other methods, to predict the region of Avocado.
# * Anomaly detection to detects the outliers in the data, in order to clean the data.
# * PCA can be used to reduced the dimensions (features) of the data, in order to simplify the existing problem. 
# * While more data and more features are better in general, it may leads to so called `curse of dimensionality` in some cases when the training exmaples are not enough. We may use bin-counting scheme or Feature Hashing Scheme to deal with large numbers of categorical features such as IP-addresses. 

# ### Simple possible improvements: 
# * Dropout to deal with overfitting.
# * $L_1$ or $L_{2}$ Regularization.
# * Batch normalization.
# * Complex models, time series forecasting may considering RNN. 
# * Advance optimization methods like BFGS, Adam, earling stopping algorithms.

# In[ ]:





# In[ ]:





# In[ ]:




