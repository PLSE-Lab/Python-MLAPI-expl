#!/usr/bin/env python
# coding: utf-8

# # The Walmart challenge: Modelling weekly sales
# In this notebook, we use data from Walmart to forecast their weekly sales. 

# ## Summary of results and approach
# 
# Work in Progress:
# 
# At writing, our internal competition at Bletchley has ended. Interestingly, the winning group had a different approach then would be expected from an AI/Machine Learning bootcamp. Their forecasts were based simply on a median of the weekly sales grouped by the Type of Store, Store & Department number, Month and Holiday dummy. 
# 
# Therefore, in my next approach, the goal will be to improve their results with the help of Neural Networks and other machine learning methods. In fact, the median will be computed similarly to how the winning group did, and a new variable, the difference to the median, will be computed. This difference will be the new dependent variable and will be estimated based on new holiday dummies, markdowns and info on lagged sales data if available.
# 
# **Final result: MAE dropped from 2200 to 1800.
# **

# ## Understanding the problem and defining a success metric
# 
# The problem is quite straightforward. Data from Walmart stores accross the US is given, and it is up to us to forecast their weekly sales. The data is already split into a training and a test set, and we want to fit a model to the training data that is able to forecast those weeks sales as accurately as possible. In fact, our metric of interest will be the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error). 
# 
# The metric is not very complicated. The further away from the actual outcome our forecast is, the harder it will be punished. Optimally, we exactly predict the weekly sales. This of course is highly unlikely, but we must try to get as close as possible. The base case of our model will be a simple linear regression baseline, which gave a MSE of 
# 
# 

# ## Load and explore data
# Before we do anything, lets import some packages.

# In[ ]:


#Really need these
import pandas as pd 
import numpy as np
from numpy import *


#Handy for debugging
import gc
import time
import warnings
import os

#Date stuff
from datetime import datetime
from datetime import timedelta

#Do some statistics
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss
import math

#Nice graphing tools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls

#Machine learning tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import sparse

## Keras for deep learning
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.models import Sequential
from keras import regularizers
from keras import optimizers

## Performance measures
from sklearn.metrics import mean_squared_error


# ## Prepare functions
# I initialize my functions in the beginning of the script to make the whole seem cleaner..

# In[ ]:


#Merge info
def mergeData(df):
    features =pd.read_csv('../input/walmart/features.csv')
    storesdata =pd.read_csv('../input/walmart/stores.csv')
    df = pd.merge(df, features, on=['Store','Date','IsHoliday'],
                  how='inner')
    df = pd.merge(df, storesdata, on=['Store'],
                  how='inner')
    return df

#http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html
def plot_prediction(predicted,true,desciption):
    fig, ax = plt.subplots()
    ax.scatter(true, predicted, edgecolors=(0, 0, 0))
    ax.plot([true.min(), true.max()], [true.min(), true.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted by '+desciption)
    ax.plot([-30,30], [0,0], 'k-')   
    ax.plot([0,0], [-30,30], 'k-')
    plt.show()
def binary(movement):
    """
    Converts percent change to a binary 1 or 0, where 1 is an increase and 0 is a decrease/no change
    
    """
    #Empty arrays where a 1 represents an increase in price and a 0 represents a decrease in price
    direction = np.empty(movement.shape[0])
    #If the change in price is greater than zero, store it as a 1
    #If the change in price is less than zero, store it as a 0
    for i in range(movement.shape[0]):
        if movement[i] > 0:
            direction[i] = 1
        else:
            direction[i]= 0
    return direction

def scatterplots(feature, label):
    x = feature
    y = df['Weekly_Sales']
    plt.scatter(x, y)
    plt.ylabel('sales')
    plt.xlabel(label)
    plt.show()


# Identify data sources

# In[ ]:


print('Reading data...')
print(os.listdir('../input/walmart/'))
print(os.listdir('../input/walmarts'))


# There are two competitions that have more or less the same data. Choose which competition to participate in.
# [One](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) or [two](https://www.kaggle.com/c/walmart-sales-forecasting). All comments are based on number two.

# In[ ]:


dataSource = 1
if dataSource==1:
    train = mergeData(pd.read_csv('../input/walmart/train.csv'))
    test = mergeData(pd.read_csv('../input/walmart/test.csv'))
    train['Split'] = 'Train'
    test['Split'] = 'Test'
    test.head()
else: 
    train = pd.read_csv('../input/walmart/train.csv')
    test = pd.read_csv('../input/walmart/test.csv')
    train['Split'] = 'Train'
    test['Split'] = 'Test'
    test.head()


# In[ ]:



test.head(1)


# In order to efficiently modify our data, we merge the two datasets for now. We also keep track of the length of our training set so we know how to split it later.

# In[ ]:


t_len = len(train) # Get number of training examples
df = pd.concat([train,test],axis=0) # Join train and test
df.tail() # Get an overview of the data


# Let's get a clearer image of what our data actually looks like with the describe function. This will give use summary statistics of our numerical variables.

# In[ ]:


df.describe()


# Since we are in the Netherlands, and we don't understand Fahrenheit, let's do a quick change there.

# In[ ]:


df.columns


# In[ ]:


df['Temperature'] = (df['Temperature'] - 32) * 5/9


# Although there is not a large variety of variables, we can definitely work with this. In the next section, we will clean the data set, engineer some new features and add dummy variables. For now, let's try to find any obvious relations between our variables to get a feeling for the data. We begin with a correlation matrix.
# 

# In[ ]:


# Code from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Most of what we see in the correlation table is of little surprise. Discounts are correlated and higher unemployment means lower Consumer Price Index. More interestingly, it appears that higher department numbers have higher sales. Maybe because they are newer? Also, larger stores generate more sales, discounts generally generate higher sales values and larger unemployment result in a bit fewer sales. Unfortunately, there appears to be little relationship between holidays, temperatures or fuelprices with our weekly sales.
# 
# Next up, let's plot some of these relationships to get a clearer image.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

headers = list(df)
labels = headers
scatterplots(df['Fuel_Price'], 'Fuel_Price')
scatterplots(df['Size'], 'Size')
#scatterplots(df['Temperature'], 'Temperature')
#scatterplots(df['Unemployment'], 'Unemployment')
scatterplots(df['IsHoliday'], 'IsHoliday')
scatterplots(df['Type'], 'Type')


# From this plot, we notice that type C stores have fewer sales in general and holidays clearly show more sales.Although no further relationships appear evident from this analysis, there appears to be some outliers in our data. Let's take a bit of a closer look at these.

# In[ ]:


df.loc[df['Weekly_Sales'] >300000]


# It appears to be quite obvious. The end of November sees a lot of exceptionally large sales. This special day, better known as Black friday, causes sales to be on fire, and undoubtedly a dummy variable should be created for this day. Also, Christmas, appears here and there. Since it is not considered holiday, we will also make a dummy for this day. Let's see if we should consider some other special days as well.

# In[ ]:


df.loc[df['Weekly_Sales'] >240000,"Date"].value_counts()


# Except for a handful spurious other dates, it appears that the two days before Christmas and Black Friday will do the job.

# 
# 
# ## Scrub the data and engineer features
# 
# ### Missing values
# 
# We will start with filling in any blank values. There seem to be some missing values in the data. We have to make sure to deal with them before feeding anything into the network.

# In[ ]:


df.isnull().sum()


# We will do a bit of very basic feature engineering here by creating a feature which indicates whether a certain markdown was active at all.

# In[ ]:


df = df.assign(md1_present = df.MarkDown1.notnull())
df = df.assign(md2_present = df.MarkDown2.notnull())
df = df.assign(md3_present = df.MarkDown3.notnull())
df = df.assign(md4_present = df.MarkDown4.notnull())
df = df.assign(md5_present = df.MarkDown5.notnull())


# In[ ]:


md1_present = df.MarkDown1.notnull()
print(md1_present)


# We can probably safely fill all missing values with zero. For the markdowns this means that there was no markdown. For the weekly sales, the missing values are the ones we have to predict, so it does not really matter what we fill in there.

# In[ ]:


df.fillna(0, inplace=True)


# In[ ]:


df.isnull().sum()


# ### Dummy variables: Categorical Data
# 
# Now we have to create some dummy variebles for categorical data.

# In[ ]:


# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
df['IsHoliday'] = 'IsHoliday_' + df['IsHoliday'].map(str)


# In[ ]:


df['Type']


# In[ ]:


# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
holiday_dummies = pd.get_dummies(df['IsHoliday'])


# ### Dummy variables: Dates
# 
# From our earlier analysis, it has turned out that the date may be our best friend. As a general rule, it is a good start to already distinguish between different months in our model. This will create 12 dummy variables; one for each month.

# In[ ]:


df['DateType'] = [datetime.strptime(date, '%Y-%m-%d').date() for date in df['Date'].astype(str).values.tolist()]
df['Month'] = [date.month for date in df['DateType']]
df['Month'] = 'Month_' + df['Month'].map(str)
Month_dummies = pd.get_dummies(df['Month'] )


# Next, let's look at 'special dates'. One variable for Christmas, one for black friday. We have to manually look up the dates of black friday if we want to extrapolate our data to other years, but for now we know: 26 - 11 - 2010 and 25 - 11 - 2011.

# In[ ]:


df['Black_Friday'] = np.where((df['DateType']==datetime(2010, 11, 26).date()) | (df['DateType']==datetime(2011, 11, 25).date()), 'yes', 'no')
df['Pre_christmas'] = np.where((df['DateType']==datetime(2010, 12, 23).date()) | (df['DateType']==datetime(2010, 12, 24).date()) | (df['DateType']==datetime(2011, 12, 23).date()) | (df['DateType']==datetime(2011, 12, 24).date()), 'yes', 'no')
df['Black_Friday'] = 'Black_Friday_' + df['Black_Friday'].map(str)
df['Pre_christmas'] = 'Pre_christmas_' + df['Pre_christmas'].map(str)
Black_Friday_dummies = pd.get_dummies(df['Black_Friday'] )
Pre_christmas_dummies = pd.get_dummies(df['Pre_christmas'] )


# In[ ]:


# Add dummies
# We will actually skip some of these
#df = pd.concat([df,type_dummies,store_dummies,dept_dummies,holiday_dummies,Pre_christmas_dummies,Black_Friday_dummies,Month_dummies],axis=1)

df = pd.concat([df,holiday_dummies,Pre_christmas_dummies,Black_Friday_dummies],axis=1)


# In[ ]:


df.columns.values


# > ### Store median
# 
# We will take the store median in the available data as one of its properties

# In[ ]:


# Get dataframe with averages per store and department
medians = pd.DataFrame({'Median Sales' :df.loc[df['Split']=='Train'].groupby(by=['Type','Dept','Store','Month','IsHoliday'])['Weekly_Sales'].median()}).reset_index()


# In[ ]:


# Merge by type, store, department and month
df = df.merge(medians, how = 'outer', on = ['Type','Dept','Store','Month','IsHoliday'])


# In[ ]:


# Fill NA
df['Median Sales'].fillna(df['Median Sales'].loc[df['Split']=='Train'].median(), inplace=True) 

# Create a key for easy access

df['Key'] = df['Type'].map(str)+df['Dept'].map(str)+df['Store'].map(str)+df['Date'].map(str)+df['IsHoliday'].map(str)


# In[ ]:


df.columns


# ### Lagged Variables
# 
# We will take a lagged variable of our store's previous weeks sales. To do so, we will first add a column with a one week lagged date, sort the data, and then match the lagged sales with the initial dataframe using the department and store number.
# 
# We begin by adding a column with a one week lag.

# In[ ]:


# Attach variable of last weeks time
df['DateLagged'] = df['DateType']- timedelta(days=7)
df.head()


# Next, we create a sorted dataframe.

# In[ ]:


# Make a sorted dataframe. This will allow us to find lagged variables much faster!
sorted_df = df.sort_values(['Store', 'Dept','DateType'], ascending=[1, 1,1])
sorted_df = sorted_df.reset_index(drop=True) # Reinitialize the row indices for the loop to work


# Loop over its rows and check at each step if the previous week's sales are available. If not, fill with store and department average, which we retrieved before.

# In[ ]:


sorted_df.head(1)


# In[ ]:


df.head(1)


# In[ ]:


sorted_df=pd.read_csv('../input/walmarts/output.csv')


# In[ ]:


sorted_df.head(1)


# Now, merge this new info with our existing dataset.

# In[ ]:


df.rename(columns={"DateType": "Date"})
sorted_df[['Dept', 'Store','Date','LaggedSales','LaggedAvailable']].head(1)


# In[ ]:


intersect=sorted_df[['Dept', 'Store','Date','LaggedSales','LaggedAvailable']]


# In[ ]:


df = df.merge(intersect, how = 'inner', on = ['Dept', 'Store','Date'])


# In[ ]:





# In[ ]:


df['Sales_dif'] = df['Median Sales'] - df['LaggedSales']
df[['Dept', 'Store','DateType','LaggedSales','Weekly_Sales','Median Sales']].head()


# ### Remove redundant items
# 
# We will take the store average in the available data as one of its properties

# In[ ]:


switch= 1

if(switch):
    df_backup = df
else:
    df=df_backup
    display(df_backup.head())


# ### Scale Variables
# 
# To make the job of our models easier in the next phase, we normalize our continous data. This is also called feature scaling.

# In[ ]:


df['Unemployment'] = (df['Unemployment'] - df['Unemployment'].mean())/(df['Unemployment'].std())
df['Temperature'] = (df['Temperature'] - df['Temperature'].mean())/(df['Temperature'].std())
df['Fuel_Price'] = (df['Fuel_Price'] - df['Fuel_Price'].mean())/(df['Fuel_Price'].std())
df['CPI'] = (df['CPI'] - df['CPI'].mean())/(df['CPI'].std())
df['MarkDown1'] = (df['MarkDown1'] - df['MarkDown1'].mean())/(df['MarkDown1'].std())
df['MarkDown2'] = (df['MarkDown2'] - df['MarkDown2'].mean())/(df['MarkDown2'].std())
df['MarkDown3'] = (df['MarkDown3'] - df['MarkDown3'].mean())/(df['MarkDown3'].std())
df['MarkDown4'] = (df['MarkDown4'] - df['MarkDown4'].mean())/(df['MarkDown4'].std())
df['MarkDown5'] = (df['MarkDown5'] - df['MarkDown5'].mean())/(df['MarkDown5'].std())
df['LaggedSales']= (df['LaggedSales'] - df['LaggedSales'].mean())/(df['LaggedSales'].std())


# Now, let's change the variable to be forecasted to the difference from the median. Afterward, we can drop the weekly sales.

# In[ ]:


df['Difference'] = df['Median Sales'] - df['Weekly_Sales']


# Let's have a look at our data set before running our actual models.

# In[ ]:


df.head()


# In[ ]:


# Code from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ### Select variables to include in model
# 
# In this section, we can change the variables we ultimately want to include in our model training. 

# In[ ]:


selector = [
    #'Month',
    'CPI',
    'Fuel_Price',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5',
    'Size',
    'Temperature',
    'Unemployment',
    
    
    
    'md1_present',
    'md2_present',
    'md3_present',
    'md4_present',
    'md5_present',

    'IsHoliday_False',
    'IsHoliday_True',
    'Pre_christmas_no',
    'Pre_christmas_yes',
    'Black_Friday_no',
    'Black_Friday_yes',    
    'LaggedSales',
    'Sales_dif',
    'LaggedAvailable'
    ]
display(df[selector].describe())
display(df[selector].head())


# ### Split data into training and test sets
# 
# Now we can split train test again and of course remove the trivial weekly sales data from the test set.

# In[ ]:


train = df.loc[df['Split']=='Train']
test = df.loc[df['Split']=='Test']
test.head()
print(len(test))


# ### Test - dev
# 
# Usually, model performance can be evaluated on the out-of-sample test set. However, since that data is not available, it may be wise to split our training set one more time in order to be able to test out of sample performance. Let's give up 20% of our training set for this sanity check development set.

# In[ ]:


# Set seed for reproducability 
np.random.seed(42)
X_train, X_dev, y_train, y_dev = train_test_split(train[selector], train['Difference'], test_size=0.2, random_state=42)
print(X_dev.shape)
print(y_dev.shape)


# ## Model selection
# 
# As usual, let's start off with all our imports.

# ### Adam optimizer with regularization
# 
# In our next model, we will stick with the relu activator, but replace the momentum with an Adam optimizer. Adaptive momumtum estimator uses exponentially weighted averages of the gradients to optimize its momentum.  However, since this method is known to overfit the model because of its fast decent, we will make use of a regulizer to avoid overfitting. The l2 regulizer adds the sum of absolute values of the weights to the loss function, thus discouraging large weights that overemphasize single observations.

# In[ ]:


neural = False
if neural:
    # Sequential model
    
    adam_regularized = Sequential()

    # First hidden layer now regularized
    adam_regularized.add(Dense(32,activation='relu',
                    input_dim=X_train.shape[1],
                    kernel_regularizer = regularizers.l2(0.01)))

    # Second hidden layer now regularized
    adam_regularized.add(Dense(16,activation='relu',
                       kernel_regularizer = regularizers.l2(0.01)))

    # Output layer stayed sigmoid
    adam_regularized.add(Dense(1,activation='linear'))

    # Setup adam optimizer
    adam_optimizer=keras.optimizers.Adam(lr=0.01,
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=1e-08)

    # Compile the model
    adam_regularized.compile(optimizer=adam_optimizer,
                  loss='mean_absolute_error',
                  metrics=['acc'])

    # Train
    history_adam_regularized=adam_regularized.fit(X_train, y_train,validation_data=(X_dev, y_dev), # Train on training set
                                 epochs=10, # We will train over 1,000 epochs
                                 batch_size=2048, # Batch size 
                                 ) # Suppress Keras output
    adam_regularized.evaluate(x=X_dev,y=y_dev)

    # Plot network
    plt.plot(history_adam_regularized.history['loss'], label='Adam Regularized')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    y_pred_neural = adam_regularized.predict(X_dev)


# ### Random Forest
# 
# Train on random forest

# In[ ]:


y_train.shape


# In[ ]:


#Random forest model specification
regr = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=None, 
                      min_samples_split=2, min_samples_leaf=1, 
                      min_weight_fraction_leaf=0.0, max_features='auto', 
                      max_leaf_nodes=None, min_impurity_decrease=0.0, 
                      min_impurity_split=None, bootstrap=True, 
                      oob_score=False, n_jobs=1, random_state=None, 
                      verbose=2, warm_start=False)

#Train on data
regr.fit(X_train, y_train.ravel())


# ### Model evaluation
#  
#  To evaluate the model, we will look at MAE and accuracy in terms of the number of times it correctly estimated an upward or downward deviation from the median.
# 

# In[ ]:


y_pred_random = regr.predict(X_dev)

y_dev = y_dev.to_frame()


# In[ ]:


# Transform forest predictions to observe direction of change
direction_true1= binary(y_dev.values)
direction_predict = binary(y_pred_random)

## show confusion matrix random forest
cnf_matrix = confusion_matrix(direction_true1, direction_predict)

fig, ax = plt.subplots(1)
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
#ax.set_xticklabels(abbreviation)
#ax.set_yticklabels(abbreviation)
plt.title('Confusion matrix of random forest predictions')
plt.ylabel('True category')
plt.xlabel('Predicted category')
plt.show();


# In[ ]:


y_pred_random


# In[ ]:


y_dev['Predicted'] = y_pred_random
df_out = pd.merge(train,y_dev[['Predicted']],how = 'left',left_index = True, right_index = True,suffixes=['_True','_Pred'])
df_out = df_out[~pd.isnull(df_out['Predicted'])]
df_out.head()


# In[ ]:


df_out['prediction'] = df_out['Median Sales']-df_out['Predicted']
plot_prediction(df_out['Weekly_Sales'],df_out['prediction'],"Random Forest")
plot_prediction(y_pred_random,y_dev['Difference'].values,"Random Forest")


# In[ ]:


df_out.head()


# In[ ]:


df.to_csv("sampleoutpput.csv")


# In[ ]:


print("Medians: "+str(sum(abs(df_out['Difference']))/df_out.shape[0]))
print("Random Forest: "+str(sum(abs(df_out['Weekly_Sales']-df_out['prediction']))/df_out.shape[0]))


# Looks good! Let's train on our full data set to get the maximum amount of information in our model.

# In[ ]:


#Random forest model specification. Set n_estimators lower for faster performance
rf_model = RandomForestRegressor(n_estimators=80, criterion='mse', max_depth=None, 
                      min_samples_split=2, min_samples_leaf=1, 
                      min_weight_fraction_leaf=0.0, max_features='auto', 
                      max_leaf_nodes=None, min_impurity_decrease=0.0, 
                      min_impurity_split=None, bootstrap=True, 
                      oob_score=False, n_jobs=1, random_state=None, 
                      verbose=0, warm_start=False)

#Train on data
rf_model.fit(train[selector], train['Difference'])


# In[ ]:


#Use if large model skipped
#rf_model = regr


# ## Forecasting sales
# 
# After we have created our model, we can predict things with it on the test set

# 

# In[ ]:


test[selector]


# In[ ]:


final_y_prediction = adam_regularized.predict(test[selector])


# In[ ]:


print(test.head())


# In[ ]:


len(test[selector].index.values)


# In[ ]:


testfile = pd.concat([test.reset_index(drop=True), pd.DataFrame(final_y_prediction)], axis=1)
testfile['prediction'] = testfile['Median Sales']-testfile[0]
testfile.head()


# Now we create the submission. Once you run the kernel you can download the submission from its outputs and upload it to the Kaggle InClass competition page.

# In[ ]:


final_y_prediction[:,0]


# In[ ]:


submission = pd.DataFrame({'id':pd.Series([''.join(list(filter(str.isdigit, x))) for x in testfile['Store']]).map(str) + '_' +
                           pd.Series([''.join(list(filter(str.isdigit, x))) for x in testfile['Dept']]).map(str)  + '_' +
                           testfile['Date'].map(str),
                          'Weekly_Sales':final_y_prediction[:,0]})
submission.head()


# Check submission one more time

# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:


len(submission.index.values)


# In[ ]:


115064 rows

