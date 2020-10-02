#!/usr/bin/env python
# coding: utf-8

# Someone on Kaggle was good enough to scrape a lot of data from the PGA Tour site, and post it for others to use.
# I wondered if there was some chance to do a sort of "Moneyball Golf" analysis, where we might find a few key stats
# the people were not adequately considering, and which might tell us who might have a shot at winning a major
# in the near future. Who knows, you might even be able to profit from this by betting on some golfer 
# who has good stats but hasn't broken through to win a major yet. 
# 
# Unfortunately I don't think I'm going to get rich from this analysis. As you'll see, there's some correlation between some of the stats tracked by the PGA Tour and pro golf success, but I couldn't find a formula that had a great predictive ability. I haven't looked at the detailed Moneyball baseball data, but I'd expect the variety of situations a baseball player could face is much more limited than the ones a golfer could face during the course of a round of golf. So that's going to be harder to capture in the stats (for golf).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import os
import matplotlib.pyplot as plt # charts and graphs
import seaborn as sns # styling & pretty colors
import shutil
import tensorflow as tf

dataset = pd.read_csv('../input/PGA_Data_Historical.csv')
dataset.info()
dataset.head()
print(tf.__version__)


# In[ ]:


#show lots of columns and rows in pandas
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 3000)


# In[ ]:


# Transpose based on key value pairs
df = dataset.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()

print("original column count:\t" + str(len(dataset.columns)))
print("     new column count:\t" + str(len(df.columns)))


# In[ ]:


df.head()


# In[ ]:


#Narrow down to the more interesting X columns. You could add others.
Keep_Columns = ['Player Name','Season','Total Money (Official and Unofficial) - (MONEY)','3-Putt Avoidance - (%)',
                'Average Distance to Hole After Tee Shot - (AVG)','Scrambling from the Sand - (%)',
                'Scrambling from the Fringe - (%)','Scrambling from the Rough - (%)',
                'Driving Accuracy Percentage - (%)','Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))',
                'Ball Speed - (AVG.)','Birdie or Better Conversion Percentage - (%)'
               ]
Keep_Columns 


# In[ ]:


#Drop non-numeric
df=df[Keep_Columns].dropna()
#Rename the columns to something shorter
df.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)':'Money'}, inplace = True)
df.rename(columns = {'3-Putt Avoidance - (%)':'ThreePutt'}, inplace = True)
df.rename(columns = {'Average Distance to Hole After Tee Shot - (AVG)':'AverageDistance'}, inplace = True)
df.rename(columns = {'Scrambling from the Sand - (%)':'ScramblingSand'}, inplace = True)
df.rename(columns = {'Scrambling from the Fringe - (%)':'ScramblingFringe'}, inplace=True)
df.rename(columns = {'Scrambling from the Rough - (%)':'ScramblingRough'}, inplace=True)
df.rename(columns = {'Driving Accuracy Percentage - (%)':'DrivingAccuracy'}, inplace=True)
df.rename(columns = {'Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))':'Distance'}, inplace=True)
df.rename(columns = {'Ball Speed - (AVG.)':'BallSpeed'}, inplace=True)
df.rename(columns = {'Birdie or Better Conversion Percentage - (%)':'BirdieConversion'}, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


#Remove $ and commas from Money
df['Money']= df['Money'].str.replace('$','')
df['Money']= df['Money'].str.replace(',','')


# In[ ]:


#Make all variables into numbers
for col in  df.columns[2:]:
   df[col] = df[col].astype(float)
df


# In[ ]:


#Scale the data so that all features are of a comparable magnitude
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Feature_Columns=['ThreePutt', 'AverageDistance', 'ScramblingSand', 'ScramblingFringe', 'ScramblingRough', 'DrivingAccuracy', 'Distance', 'BallSpeed', 'BirdieConversion']
df[Feature_Columns]=scaler.fit_transform(df[Feature_Columns])
df


# In[ ]:


#check correlations with Money, to see which features are useful.  Driving accuracy seems to be not so useful. 
#The others look useful.
df.corr(method ='pearson') 


# In[ ]:


sns.regplot(x="BirdieConversion", y="Money", data=df);


# In[ ]:


#Three putts are negatively correlated with Money
sns.regplot(x="ThreePutt", y="Money", data=df);


# In[ ]:


sns.regplot(x="ScramblingFringe", y="Money", data=df);


# In[ ]:


sns.regplot(x="ScramblingSand", y="Money", data=df);


# In[ ]:


sns.regplot(x="ScramblingRough", y="Money", data=df);


# In[ ]:


sns.regplot(x="DrivingAccuracy", y="Money", data=df);


# In[ ]:


sns.regplot(x="BallSpeed", y="Money", data=df);


# In[ ]:


sns.regplot(x="Distance", y="Money", data=df);


# Based on the above, avoiding 3-putts is big. Distance matters more than I might have expected. My guess was that Driving Accuracy would be more important that distance. Nope. Converting on your birdie opportunities is huge. Scrambling is less important that I might have thought. Ball speed surprised me by actually having a correlation with money won, but when you think about a higher ball speed should get you more distance, so it makes sense.
# 
# For future work, we'd probably want to go deeper into putting.  They have many variables I didn't use here.
# I assumed many would be correlated, but there might be some useful info in there (e.g., long putting vs. short putting?).

# In[ ]:


# Imports for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


# Create a LinearRegression Object
lreg = LinearRegression()


# In[ ]:


# Drop a few columns to get the X (feature) columns
X=df.drop(['Money','Player Name', 'Season'], axis=1)
# Target
Y=df.Money


# In[ ]:


#I'm going to add some squared features and crossed features
#This did improve the regression results vs only linear features 
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)  


# In[ ]:


X


# In[ ]:


X.shape


# In[ ]:


#Split into training and test sets
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y)


# In[ ]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[ ]:


lreg.fit(X_train,Y_train)


# In[ ]:


lreg.coef_


# In[ ]:


lreg.intercept_


# In[ ]:


# Calculate predictions on training and test data sets
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)
pred_train


# In[ ]:


print("RMSE with Y_train: %.2f"  % np.sqrt(np.mean((Y_train - pred_train) ** 2)))
print("RMSE with Y_test: %.2f"  % np.sqrt(np.mean((Y_test - pred_test) ** 2)))


# Looks like the linear model didn't predict too well. We've got a large error on both the training and test data sets.

# In[ ]:


# Scatter plot the training data
train = plt.scatter(pred_train,(Y_train-pred_train),c='b',alpha=0.5)

# Scatter plot the testing data
test = plt.scatter(pred_test,(Y_test-pred_test),c='r',alpha=0.5)

# Plot a horizontal axis line at 0
plt.hlines(y=0,xmin=-10,xmax=50)

#Labels
plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('Residual Plot')


# It looks like there's an opportunity to improve on this model.

# Let's take a couple of golfers. Jason Day made 4.8 million, while Adam Hadwin made 1.86 million, on average for the years they competed. Day's stats are almost all better thatn Hadwin's, but only marginally so. So it appears that small differences can make you a lot of money on the PGA tour.

# In[ ]:


Jason_Day=df[df['Player Name'].str.contains('Day')]
Jason_Day.Money.mean()


# In[ ]:


Adam_Hadwin=df[df['Player Name'].str.contains('Hadwin')]
Adam_Hadwin.Money.mean()


# Let's try to predict their earnings for a few years.

# In[ ]:


X_predict=Jason_Day.drop(['Money','Player Name', 'Season'], axis=1)
lreg.predict(poly.fit_transform(X_predict)).mean()


# In[ ]:


X_predict=Adam_Hadwin.drop(['Money','Player Name', 'Season'], axis=1)
lreg.predict(poly.fit_transform(X_predict)).mean()


# Not bad for Day.  Not so great for Hadwin. We're not getting very accurate results, but at least 
# we see that the better golfer, based on the features, makes more money. Let's try one more golfer--Dustin Johnson

# In[ ]:


Dustin_Johnson=df[df['Player Name'].str.contains('Dustin Johnson')]
Dustin_Johnson.Money.mean()


# In[ ]:


X_predict=Dustin_Johnson.drop(['Money','Player Name', 'Season'], axis=1)
lreg.predict(poly.fit_transform(X_predict)).mean()


# In[ ]:





# We predicted he'd make more than Hadwin. In reality he made way more than Hadwin, but also more than Day over this time. We predicted he'd only make a bit more than Day. If you look at his stats vs. Day's, some are better, but not all. 

# These predictions aren't great, although generally better golfers (statistically) seem to be making more money than ones with worse stats, based on the model. 
# 
# The PGA Tour collects a lot more variables, but it looks like a lot of these would be correlated with each other (e.g., long putting, short putting, putting overall, etc.), or not that useful. 
# 
# My hunch is that these variables don't explain all that well why great golfers are great. Of course they avoid 3 putts, and convert on birdies when they have the chance, and the data shows an association between earning money and these factors. But meanwhile, other variables don't seem to mean much. Driving accuracy is surprisingly useless.
# 
# So I suspect there's something that's just not captured by these stats that determines whether a golfer can win the big ones. The stats that the PGA Tour provides don't necessarily reveal that. Dealing with pressure and pulling off a big birdie at a critical moment matters a lot more than driving accuracy.
# 
# Let's try a more complex model--a neural network. A challenge may be that we actually don't have a lot of data points: about 1700.

# In[ ]:


FEATURE_NAMES=list(df.columns.drop(['Money','Player Name', 'Season']))
FEATURE_NAMES


# In[ ]:


LABEL_NAME = 'Money'


# In[ ]:


# Drop a few columns to get a dataframe with the target (Money) as well as the X variables
DNNdata=df.drop(['Player Name', 'Season'], axis=1)


# In[ ]:


DNNdata


# In[ ]:


DNNdata.shape


# In[ ]:


#I got this code from Google's Github on Tensorflow training: 
#https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/b_estimator.ipynb

def make_train_input_fn(d, num_epochs):
  return tf.estimator.inputs.pandas_input_fn(
    x = d,
    y = d[LABEL_NAME],
    batch_size = 64,
    num_epochs = num_epochs,
    shuffle = True,
    queue_capacity = 1000
  )

def make_eval_input_fn(d):
  return tf.estimator.inputs.pandas_input_fn(
    x = d,
    y = d[LABEL_NAME],
    batch_size = 64,
    shuffle = False,
    queue_capacity = 1000
  )

def make_prediction_input_fn(d):
  return tf.estimator.inputs.pandas_input_fn(
    x = d,
    y = None,
    batch_size = 128,
    shuffle = False,
    queue_capacity = 1000
  )

def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURE_NAMES]
  return input_columns


# In[ ]:


DATASET_SIZE = DNNdata.shape[0]
DATASET_SIZE


# In[ ]:


#Divide into train and test sets
DATASET_SIZE = DNNdata.shape[0]

train_df=DNNdata.sample(frac=0.8,random_state=200)
test_df=DNNdata.drop(train_df.index)


# In[ ]:


train_df.shape[0], test_df.shape[0]


# In[ ]:


OUTDIR = "Golf_Training"

tf.logging.set_verbosity(tf.logging.INFO)

shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.DNNRegressor(hidden_units = [16,8],
      feature_columns = make_feature_cols(), model_dir = OUTDIR)


# In[ ]:


model.train(input_fn = make_train_input_fn(train_df, num_epochs = 500))


# In[ ]:


def print_rmse(model, d):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))
print_rmse(model, test_df)


# That's pretty bad.  In fact it's worse than then linear regression, above.

# In[ ]:


# Try a prediction for Jason Day
predictions = model.predict(input_fn = make_prediction_input_fn(Jason_Day))
for items in predictions:
  print(items)


# In[ ]:


predictions = model.predict(input_fn = make_prediction_input_fn(Adam_Hadwin))
for items in predictions:
  print(items)


# That's wierd.  We're predicting Day and Hadwin will make about the same, while Day is clearly the better golfer. 
# In fact the model seems to be predicting similar targets regardless of the inputs.

# In[ ]:


df.mean()


# Maybe someone has some ideas how to improve the DNN results. Different hidden layers don't seem to help much. Perhaps we're limited with what we can do with that much data (approx. 1700 records), and can't train the model all that well due to that. Maybe we should try some feature engineering. I threw out a lot of features that the PGA is tracking. Maybe there's something useful in there. Some of it was going to be a bit of a pain in the ass to deal with, such as distances like 5'7". It would take a little work to preprocess, but might improve the results. 

# In[ ]:




