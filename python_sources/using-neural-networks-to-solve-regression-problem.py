#!/usr/bin/env python
# coding: utf-8

# # Using Neural Networks to solve regression problem: beginner's guide.
# In this kernel I will discuss how to tackle this regression problem using Artificial Neural Networks. I will do a minimal level of data preprocessing which gave me a 0.114 result with standard Sklearn linear regressors.
# 
# The Neural Network approach leaves me with the question as to why Neural Networks seem to perform much worse than the standard linear regression models. If you have any idea why this is, please let me know in the comments, I am very curious.

# In[ ]:


# Import usual libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew

# Import Neural Network libraries
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Import data processing libraries
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')

# Not the best practice, but ok for now. (surpresses all warnings)
import warnings
warnings.filterwarnings("ignore")

# Read in the data with pandas .read_csv
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# ## Data preprocessing.
# This minimal level of data preprocessing gave me satisfactory results when used with the standard linear regression models such as Lasso, Ridge and xgboost. There are many ways to improve the filling of missing data, discard outliers and changing numerical variables to categorical ones. Many good kernels can be found on this topic and they will improve the end score.

# In[ ]:


df_train.head() # get a general view of the data. The test data has the same columns, only SalePrice is missing.

# Concatenate the test and training data (apart from the SalePrice column) to treat the input columns equally during the preprocessing step.

complete_input = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))


# ## 1. Skewness
# Skewness indicates the tendency of the data to be asymmetrically distributed around its average. Regression models in general perform better when the data resembles a normal distribution. So the first step is to look at whether the data is skewed, both output and input, and if so, transform it so a more normal looking distribution.

# In[ ]:


# Output data skewness

df_train.SalePrice.hist() # raw output (target) data is skewed
plt.figure()
df_train.SalePrice.apply(np.log1p).hist() # log_transformed data is more normal.

df_train.SalePrice = df_train.SalePrice.apply(np.log1p) # transform the output column with a log+1 transformation: x -> log(1+x). NOTE: in the end, to get back to real dollar results, we need to invers transofrm.


# In[ ]:


# Input data skewness. 
# Naturally, skewness is only defined for numerical variables, so we first need to determine which columns are represented by numberical values.

numeric_feats = complete_input.dtypes[complete_input.dtypes != 'object'].index # .dtypes gives a df with cols as index and dtypes as varaible.
                                                                               # .index then gives the col names
    
# calculate skewness of each num feat, discarting NaN's since they don't contribute to skewness and will distort the result.
skewed_feats = complete_input[numeric_feats].apply(lambda x: skew(x.dropna())) # select only the numerical columns
skewed_feats = skewed_feats[skewed_feats > 0.1] # Which level of skewness do you want to correct for? The lower, the more data will be transformed
skewed_feats = skewed_feats.index

print(skewed_feats) # names of each column where the skewness is larger than 0.1, with NaN's discarted.

# Now transform all the columns that have skewed, numerical feats.
complete_input[skewed_feats] = np.log1p(complete_input[skewed_feats])


# ## 2. Missing data
# Missing data are entries that have NaN for one or more of the variables (house indicators). Normally you would want to look into each variable and see what it means when there is a missing value, and figure out what the best aproach would be to fill it (or drop it). However, since this Kernel is focussed around building a Neural Network model, we will just fill every missing value with the mean of the entire set of entries for that variable. 

# In[ ]:


# First get an overview of all the missing data
# NOTE that here we assume that the test dataset is a good representation of the training dataset concerning any parameter.

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'], sort=False)
print(missing_data.head(20))

# how to deal with the missing values can be to drop, set to zero, or to mean, depending on what the variables denotes.

complete_input = complete_input.fillna(0) # for now we set all the missing data to zero.


# ## 3. One-hot encoding (handling categorical data)
# In this step we want to change the type of data for categorical variables. These are often stored as strings denoting their category. However, models often require numerical inputs. This can be achieve by creating dummy variables in a process called one-hot encoding. 
# As an example: A categorical variable that has three categories ('Yes', 'No', 'Unknown') can be transformed to three variables denoted 
# by: [ (1,0,0); (0,1,0); (0,0,1) ]. Sklearn has a built-in function that does this. 

# In[ ]:


# pandas has a single function to do this. Sklearn also has a built in function in the preprocessing package.

complete_input = pd.get_dummies(complete_input)

print(complete_input.info())


# ## 4. Split the complete input set back into train- and test- data.
# Here we split up our complete_input dataset back into training- and test-dataset. We also want to scale our input data because the actual values of the different inputs (house characteristics) might differ a lot in range. This will affect the weight selection of the Neural Network training process. We want them all to be at the same scale to eliminate this bias.

# In[ ]:


# preparing data for sklearn models
print(df_train.shape)
print(df_train.shape[0]) # split the complete_input dataframe at this point since this is the original number of entries of the traning data set

X_train = complete_input[:df_train.shape[0]]
X_test = complete_input[df_train.shape[0]:]
Y_train = df_train.SalePrice

#X_train = preprocessing.StandardScaler(X_train)
#X_test = preprocessing.StandardScaler(X_test)


# ## Implementing the (deep) Neural Network
# The neural network will consist of the following layers:
# 1. The input layer: this layers needs to have as many nodes as our training data has columns (variables, including dummies).
# 1. Three hidden layers that each have 15 nodes and *relu* activation function. (hidden layers make the network 'deep'.
# 1. An output layer that has one output node, which represents the predicted price of the house.
#   
#   
# * The weights in the entire network are updated each cycle by means of *Adam* method, which is an enhanced gradient descent method most often used.  
# * The errors are minimized using the *mean squared error* function as minimizing metric.  
# * The complete data is cycled through the number of *epochs*.  
# * The validation split indicates the ratio between the portion of data we use for training, and for testing.
# * Verbose = 0 means we are not printing any information while training.
# 
# **Note** that the end rmse heavily depends on the structure and parameters of the network. Moreover, it is not the case that ''adding more layers'' gives you a better testing result, due to overfitting on the training data. 
# 

# In[ ]:


num_cols = len(X_train.columns) # get the number of columns as the number of input nodes in our network

model = Sequential() # initiating the model
model.add(Dense(15, input_shape=(num_cols,), activation = 'relu')) # input layer
model.add(Dense(15, activation='relu'))  # hidden layer 1
model.add(Dense(15, activation='relu'))  # hidden layer 2
model.add(Dense(15, activation='relu'))  # hidden layer 3
model.add(Dense(1,))                    # output layer

#Compiles model
model.compile(Adam(lr=0.003), 'mean_squared_error') # optimizing method and error function, LR should be large for large outputs

#Fits model
history = model.fit(X_train, Y_train, epochs = 1000, validation_split = 0.2,verbose = 0)
history_dict=history.history


# ## Looking at the training and test results.
# Now that we trained and fitted out Neural Network model, we can look at a few indicators of how our model did.
# 1. We can plot the evolution, per cycle (epoch), of our training error loss and validation error loss. We see that the training error loss keeps on decreasing ever so slowly, whereas the validation error loss stagnates. Overfitting is characterized by an *increase* of the validation error loss, while the training loss keeps decreasing. If this is the csae, we would want to implement an early stopping mechanism, or just manually decrease the number of epochs.
# 2. We can use our trained Neural Network to predict the prices of our training data, and then compare them with the actual prices (which we have available). We can compute the RMSE between those two. In this case, for this NN structure, we get 0.112, not bad.
# 3. Finally, we predict the prices for our test-data, which will be our final submission.

# In[ ]:


#Plots model's training cost/loss and model's validation split cost/loss
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.figure()
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='val training loss')
plt.legend()

# Test how model holds up for our training data (not that good of an indicator but it gives us an approximate sense)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("The MSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_log_error(Y_train,y_train_pred))))

# Make dataframe of performance model on the training data
y_train_pred_df = pd.DataFrame(y_train_pred, columns=['SalePrice']) # MAKE DIMENSION OUTPUT OK
Compare_df = pd.DataFrame({'TrueValue': np.expm1(Y_train), 'PredValue': np.expm1(y_train_pred_df.SalePrice)})
print(Compare_df)


# ## Submission file
# Now we close off by writing the submission file. Be sure to print the file to make sure it looks as you'd expect it to look. If not, then something went wrong and you can correct it without having to submit the file first.

# In[ ]:


y_test_pred_df = pd.DataFrame(y_test_pred, columns=['SalePrice']) # MAKE DIMENSION OUTPUT OK

# making a submission file
my_submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': np.expm1(y_test_pred_df.SalePrice)})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_NeuralNets.csv', index=False)


print(my_submission)


# ## In conclusion
# This kernel showed how you can easily implement a deep neural network to tackle this regression problem. With minimal data preprocessing we still got an acceptable result. With better, in dept data processing this result could be improved significantly. You can also play around with different Neural Network strucutres to try and improve the result.
# 
# ## QUESTION
# In general, with this level of data preprocessing, it is easy to obtain **better** results with other, simpler linear regression models such as Lasso, Ridge, or xgboost. I am really curious as to why Neural Networks are not as good for this task as these simpler models. If anyone has a suggestion, I would be very happy to hear and discuss.
