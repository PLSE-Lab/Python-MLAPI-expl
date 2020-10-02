#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import relevant libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Objective - predict the bike-share demand (regression problem)
# 
# import the necessary libraries for EDA
# 
# ** Idea of this notebook is to explore basic feature selection/engg methods use multiple ML models on a well known dataset and see how they perform and learn how to improve results
# 
# ** Models used - KNN, Random forests, Decision trees with boosting
# ## I strongly suggest to ask "Why" for every step in this notebook and go through the questions at the end
# 
# 

# In[ ]:


bike = pd.read_csv('../input/train.csv')


# # Checklist to run once we have read the data and imported the libraries
# - What problem are we solving here - regression or classification? This will decide the algorithm to use
# - Does the data have missing values (.info() method)
# - Do we need to do feature selection? Or can we just use all the features? Which features to drop
#     - What EDA can be done to aid this process?
#     - Drop correlated features, outliers which might skew predictions (seaborn heatmap with correlations)
#     - What's the split in categorical/quant variables? (.info() method)
#         - Are the Quant variables normally distributed? (distplot of each variable or the summary pair plot)
# - Feature engineering
#     - which features have an impact on the outcome (bike rentals/hour)- (PAIR PLOT can help here)

# In[ ]:


# Check for missing values - AWESOME!! no missing values in the "Training" dataset
# The heatmap tool of seaborn can also be used for a visual representation for checking missing values
bike.info()


# In[ ]:


# check out the head to see categorical & quant variables and initial observations
bike.head()


# As per our initial checklist
# - no missing values
# - all numerical variables/features, though (season, holiday, workingday) are categorical variables
# - We can check for distribution & outliers of the quant variables
# - number of features is less, so we can look at correlated variables which can be dropped
# - datetime feature needs to be split up further to understand
#     - trends by yr, month, day, time of day

# In[ ]:


bike['datetime'].head()


# In[ ]:


import datetime
bike['datetime'].iloc[0]


# In[ ]:


bike['datetime'] = pd.to_datetime(bike['datetime'])
bike['Hour'] = bike['datetime'].apply(lambda time: time.hour)
bike['Month'] = bike['datetime'].apply(lambda time: time.month)
bike['year'] = bike['datetime'].apply(lambda time:time.year)
bike['Day of Week'] = bike['datetime'].apply(lambda time: time.dayofweek)


# In[ ]:


bike.head()


# In[ ]:


sns.countplot(x='Day of Week',data=bike,palette='viridis')


# # How does the "count" variable differ with other features/variables
# - BY DAY

# In[ ]:


plt.scatter('Hour', 'count', data = bike)


# ** 
# 1. One method to identify the top influencing variables (feature selection) is to check for correlation between the variables and the outcome/target variable
# 2. Other is to use SelecKBest optimized by Chi2 to select top features
# 3. There's also the "Common sense"way for selecting features (As Andrew Ng mentions in his basic ML course)
#     - How would someone work-out the answer by considering features which are most likely to impact the target variable.
#     - Lets think about this - When would it be most likely that you'll use a Bike & not an alternate mode of transport
#         - Weather is favorable (Temperature, Wind, Humidity etc)
#         - Do you have access to a bike?
#         - Is it affordable?
#         etc etc 
# 
# We'll limit our feature selection methods to the above for this example    

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# How do we select the 4 best predictive features out of the feature set available?
df = bike.drop('datetime', axis = 1)
X = df.drop('count', axis = 1)
#Example = yourdf.drop(['columnheading1', 'columnheading2'], axis=1, inplace=True)
y = df['count']


# In[ ]:


test = SelectKBest(score_func = chi2, k = 4) # Instantiating selectkbest 
fit = test.fit(X,y)  # Now we fit selectkbest to the data
print(fit.scores_)
features = fit.transform(X)
print(features[0:5,:]) # higher the score, better the rating


# Top 4 influencing features as per SelectKBest are (Selected as per the scores in descending order)
# - 10th feature - registered
# - 9th feature - casual
# - 7th feature - humidity
# - 11th feature - hour

# # Reviewing correlation for feature selection 

# In[ ]:


plt.figure(figsize = (18,18))
sns.heatmap(bike.corr(), cmap='coolwarm', annot = True)


# - temp/atemp are correlated and hence one of them can be dropped without losing predicting ability
# - count is highly correlated with registered (so that's an important feature to predict demand)
# - We need to further select fewer features which have a meaningful impact on the demand. With an initial glance, we can look at features/variables which have a significant correlation with the output ('count' variable)
#     - temp, casual, registered, hour and maybe humidity
# 
# For these variable, we can check for their distribution

# In[ ]:


select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']]


# In[ ]:


select_features.head()


# In[ ]:


sns.distplot(bike['temp'], bins = 10) # we'll assume a normal distribution and move ahead  


# In[ ]:


sns.countplot(x='Month',data=bike,palette='viridis')


# # Running any ML algo (Supervised) follows a standard pattern once we have completed - EDA, Feature engg (includes scaling of features), Feature selection
# 1. Import the necessary algo (Classifier or regressor)
# 2. Instantiate the above algorithm
# 3. Fit the training data in the above instantiated algorith
# 4. Make predictions
# 4. Check for accuracy of prediction of the model

# # Running a RF on the above 5 selected features only

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train, y_train) # This has been run on scaled features


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, rfc_pred))


# In[ ]:


rms


# # How good or bad is the above model, seeing the RMS score?
# - how can we improve this RMSE without any major changes to the model

# # Lets check the result in a KNN with the same above features

# In[ ]:


from sklearn.preprocessing import StandardScaler # need to scale feature set to fit KNN


# In[ ]:


scaler = StandardScaler() # initialise a scaler object to run on a dataframe


# In[ ]:


select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']]


# In[ ]:


scaler.fit(select_features) # run the above scaler method on the selected dataframe


# In[ ]:


scaled_features = scaler.transform(select_features)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=20) # initialise the KNN classifier with neighbours=20 in this case


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


pred = knn.predict(X_test) #run the KNN model on the test data


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, pred))


# In[ ]:


rms


# # Check the result with Adaboost

# In[ ]:


select_features = bike[['temp', 'casual', 'registered', 'Hour', 'humidity']] # Adding year though it has a low correlation with the target variable - 'count'


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(select_features,bike['count'],
                                                    test_size=0.30, random_state=101)


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


dt = DecisionTreeRegressor() 
clf = AdaBoostRegressor(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it accepts sample weight 
clf.fit(X_train,y_train)


# In[ ]:


clf_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, clf_pred))


# In[ ]:


rms


# In[ ]:


from sklearn import metrics
print (metrics.accuracy_score(y_test, clf_pred))


# ** How do we evaluate the above RMS scores for the various models
# One way is to compare it to the actual distribution of the "count" variable 

# In[ ]:


bike['count'].describe()


# In[ ]:


sns.distplot(bike['count'], bins = 10)


# **QUESTIONS TO PONDER ON**
# **
# - Which other date pre-processing steps could've been used? PCA?
# - What do the Fit and transform methods do (KNN method)
# - What's an ideal RMS value here?
# - What happens if we change the number of trees in Adaboost from the current one set at 100
#     - how about changing the learning rate**
# 
