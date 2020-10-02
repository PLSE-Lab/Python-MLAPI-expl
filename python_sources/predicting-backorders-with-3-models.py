#!/usr/bin/env python
# coding: utf-8

# #Introduction
# 
# The data used for this project comes from the "Can You Predict Product Backorders?" data set uploaded to Kaggle. The code was written to use the version 4 dataset uploaded on 28 April 2017 (Australian Eastern Standard Time).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting (graph plotting, not evil plotting)
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler       # Preprocessing method
from sklearn.neighbors import KNeighborsClassifier   # K-Nearest Neighbours Classifier model
from sklearn.svm import SVC                          # Support Vector Classifier model
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#%% Import data

# Dictionary containing values for representing NaNs
na_other = {'perf_6_month_avg':-99, 'perf_12_month_avg':-99}

# Import data from file
train_df = pd.read_csv('../input/Kaggle_Training_Dataset_v2.csv', na_values=na_other)
test_df = pd.read_csv('../input/Kaggle_Test_Dataset_v2.csv', na_values=na_other)


# ##Notes on importing data
# 
# On the discussion board, it is said that the -99 values in the perf_6_month_avg and perf_12_month_avg columns are for missing values. Hence when the data is imported, the -99's will be converted to NaNs.
# 
# There are negative values in the national_inv column, but it is said on the discussion board that these negative values mean that the shop ordered more stock than is available. The point is that these negative values are valid values.
# 
# The warning message is due to the sku column containing numbers in all rows except for one row that contains a string.

# In[ ]:


# Check size of each set
train_df.shape, test_df.shape


# In[ ]:


# Look at data types
train_df.dtypes


# ##Notes on data types
# 
# The dataset contains a mix of numerical and non-numerical data. We should take a look at some of the values in the columns.

# In[ ]:


# A prelimiary look at the individual columns in the training set
# The first 5 rows
train_df.head()


# In[ ]:


# Some columns are not shown, so show them here
train_df.ix[0:4,'sales_9_month':'potential_issue']


# In[ ]:


# The last 5 rows of the training dataset
train_df.tail()


# ##Notes on the last row of the dataset
# 
# The last row of the training dataset is not a valid sample and should be removed. We should check whether the same problem is in the test data set, and if so fix it too.

# In[ ]:


# The last 5 rows of the testing dataset
test_df.tail()


# In[ ]:


# Summarise the numerical data in train_df
train_df.describe()


# In[ ]:


# Summarise the non-numerical data in train_df
train_df.describe(include=['O'])


# ##Notes on the data overall
# 1. The data are a mix of string and floating point values.
# 
# 2. The sku has a unique value for each row, so it is the index column and should be dropped.
# 
# 3. The features with string values, apart from sku, are categorical features that only contain 'yes' and 'no'. They can be changed to represent the same information in numerical values.
# 
# 4. The numerical features have different scales, which may be a problem for some machine learning algorithms. The features should be rescaled to have similar scale.
# 
# 5. There are missing values in lead_time, perf_6_month_avg and perf_12_month_avg. These missing values need to be replaced or the samples with missing values need to be removed.
# 
# - lead_time has (1687860 - 1586967 = 100,893) (5.98%) missing values
# - perf_6_month_avg has (1687860 - 1558382 = 129,478) (7.67%) missing values 
# - perf_12_month_avg has (1687860 - 1565810 = 122,050) (7.23%)    missing values
# 
# 6. The last row is not a valid sample in both the training and testing datasets, and thus should be dropped.
# 
# 7. There are 1,676,567 samples when the product did not go on backorder. There are 11,293 (0.67%) samples when the product did go on backorder.

# In[ ]:


#%% Data cleaning

# Drop the sku column
train_df = train_df.drop('sku', axis=1)
test_df = test_df.drop('sku', axis=1)


# In[ ]:


# Drop the last row
train_df = train_df[:-1]
test_df = test_df[:-1]


# In[ ]:


# Change categorical features from string to numerical
Cols_for_str_to_bool = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
                        'stop_auto_buy', 'rev_stop', 'went_on_backorder']

for col_name in Cols_for_str_to_bool:
    train_df[col_name] = train_df[col_name].map({'No':0, 'Yes':1})
    test_df[col_name] = test_df[col_name].map({'No':0, 'Yes':1})


# In[ ]:


# Look at replacing NaNs

# Look at histogram of perf_6_month_avg
train_df.perf_6_month_avg.plot.hist()


# ##Notes on histogram for perf_6_month_avg
# 
# The distribution of performance values is left skewed. Most values fall in the range 0.7-1. It should be OK to assume that samples with missing performance values will have values close to the median value.

# In[ ]:


# Look at histogram of perf_12_month_avg
train_df.perf_12_month_avg.plot.hist()


# ##Notes on histogram for perf_12_month_avg
# 
# This is similar to the histogram for perf_6_month_avg. The distribution of performance values is left skewed. Most values fall in the range 0.7-1. The same assumption should be OK.

# In[ ]:


# Look at histogram of lead_time
train_df.lead_time.plot.hist()


# ##Notes on histogram for lead_time
# 
# The distribution of lead times is right skewed. Most lead times fall in the range 0-20. It should be OK to assume that samples with missing lead times will have lead times close to the median lead time.

# In[ ]:


# Replace NaNs in the dataset

# perf_6_month_avg
train_df.perf_6_month_avg = train_df.perf_6_month_avg.fillna(train_df.perf_6_month_avg.median())
test_df.perf_6_month_avg = test_df.perf_6_month_avg.fillna(test_df.perf_6_month_avg.median())

# perf_12_month_avg
train_df.perf_12_month_avg = train_df.perf_6_month_avg.fillna(train_df.perf_12_month_avg.median())
test_df.perf_12_month_avg = test_df.perf_6_month_avg.fillna(test_df.perf_12_month_avg.median())

# lead_time
train_df.lead_time = train_df.lead_time.fillna(train_df.lead_time.median())
test_df.lead_time = test_df.lead_time.fillna(test_df.lead_time.median())


# In[ ]:


#%% Data visualisation

# Look at correlations between features and the label

# Set figure size 
fig = plt.figure(figsize=(8,8)) 

# Plot a correlation matrix
plt.imshow(train_df.corr(), cmap=plt.cm.Blues, interpolation='nearest', aspect='auto')

# Display legend showing what the colours mean
plt.colorbar()

# Add tick marks and feature names for ease of reading
tick_marks = [i for i in range(len(train_df.columns))]
plt.xticks(tick_marks, train_df.columns, rotation='vertical')
plt.yticks(tick_marks, train_df.columns)

# Show the plot
plt.show()


# ##Notes on correlation matrix
# 
# The correlation matrix shows that the quantity in transit, the forecast sales over 3/6/9 months, the actual sales over the previous 1/3/6/9 months, and minimum recommended stock level are highly correlated. This is not surprising because if an item had high real sales over the last 1/3/6/9 months, then it is reasonable for the forecast sales over the next 3/6/9 months to also be high. If forecast sales are high, then it would be useful to have more of the stock in hand and to have more shipped in.
# 
# Besides that, the average performance over the last 6 months strongly correlates with that over the last 12 months.
# 
# Overall, the correlation matrix suggests that the number of features used for predicting whether an item goes on back order can be lower than the number of features in the dataset. In other words, the dimensionality of the problem may be reduced.

# In[ ]:


# Take a closer look at correlations with scatter plots.

# Forecast columns
forecasts = ['forecast_3_month','forecast_6_month', 'forecast_9_month']

# Pair-wise scatter plot for the forecasts
sns.pairplot(train_df, vars=forecasts, hue='went_on_backorder', size=3)

# Show the plot
sns.plt.show()


# ##Notes on the forecasts' pair-wise scatter plots
# 
# The forecast values over each time frame have very close linear correlation with each other, as expected from the correlation matrix. The forecast values cover a wide range from 0 to over 1 million. Backorders only occur when the forecast value is low.

# In[ ]:


# Do a pair-wise scatter plot for sales
sales = ['sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month']
sns.pairplot(train_df, vars=sales, hue='went_on_backorder', size=3)
sns.plt.show()


# ##Notes on the sales' pair-wise scatter plots
# 
# The sales over each time frame have good linear correlations with each other, as expected from the correlation matrix. There are some instances when the sales at different time frames fall away from the linear correlation. The sales range from 0 to over 1 million. Backorders only occur when sales are low.

# In[ ]:


# Do some data separation for more plots

# Separate data by going on backorder or not
no_bo = train_df.ix[train_df['went_on_backorder'] == 0]       
is_bo = train_df.ix[train_df['went_on_backorder'] == 1]


# In[ ]:


# Make scatter plots of the 3-month forecast against each of the sales
for col in sales:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    no_bo.plot(kind='scatter', x=col, y='forecast_3_month', ax=ax, color='DarkBlue', legend=True)
    is_bo.plot(kind='scatter', x=col, y='forecast_3_month', ax=ax, color='Red')


# ##Notes on forecast and sales scatter plots
# 
# There is a good linear relationship between sales and forecasts, as expected from the correlation matrix. Backorder happens when sales and forecasts are low.
# 

# In[ ]:


# Look at forecast, sales, in transit and recommended stock level in a pair-wise scatter plot
feature_set_1 = ['forecast_3_month', 'sales_1_month', 'in_transit_qty', 'min_bank']
sns.pairplot(train_df, vars=feature_set_1, hue='went_on_backorder', size=3)
sns.plt.show()


# ##Notes on the pair-wise scatter plot of forecast, sales, in transit and recommended stock level
# 
# The scatter plots show okay linear relationships between forecast, sales, in transit and recommended stock level. All the features range from 0 to over 300,000. Backorders only occur when the features are at low values.
# 
# Due to the good correlations and sufficiently linear relationships between these features, they will all be represented by a single feature in the machine learning models. The feature chosen is sales_1_month. This is because past sales is measured, whereas the quantity in transit, recommended minimum stock and forecasts are likely derived from past sales.

# In[ ]:


# Look at the two performance columns
fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
no_bo.plot(kind='scatter', x='perf_6_month_avg', y='perf_12_month_avg', ax=ax, color='DarkBlue')
is_bo.plot(kind='scatter', x='perf_6_month_avg', y='perf_12_month_avg', ax=ax, color='Red')


# ##Notes on the performance columns
# 
# There is a perfect linear relationship between perf_6_month_avg and perf_12_month_avg. As such, only one of the two features is enough for use in a machine learning model. Backorders occur for all performance values.

# In[ ]:


#%% Machine learning models

# Filter out the data that will be used

# Features chosen
features = ['national_inv', 'lead_time', 'sales_1_month', 'pieces_past_due', 'perf_6_month_avg',
            'local_bo_qty', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']

reduced_train_df = train_df[features]
reduced_test_df = test_df[features]

# Set labels
train_label = train_df['went_on_backorder']
test_label = test_df['went_on_backorder']


# In[ ]:


# Change scale of data

# Use MinMaxScaler to convert features to range 0-1
# The label is already in the range 0-1, so it won't be affected by this.
pp_method = MinMaxScaler()
pp_method.fit(reduced_train_df)

reduced_train_df = pp_method.transform(reduced_train_df)
reduced_train_df = pd.DataFrame(reduced_train_df, columns=features)

reduced_test_df = pp_method.transform(reduced_test_df)
reduced_test_df = pd.DataFrame(reduced_test_df, columns=features)


# In[ ]:


# KNN
#model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
#model.fit(reduced_train_df, train_label)
#score = model.score(reduced_test_df, test_label)
#print('KNN model score is', score)


# ##Notes on KNN
# 
# Due to the large number of samples, the kernel times out before the code can complete. The score I got from running the code on my PC was 0.988445729629.

# In[ ]:


## SVC
##model = SVC(C=1, kernel='rbf', random_state=10)  # Random state to get a consistent score
#model.fit(reduced_train_df, train_label)
#score = model.score(reduced_test_df, test_label)
#print('SVC model score is', score)


# ##Notes on SVC
# 
# The score I got from running the code on my PC was 0.988896003305.

# In[ ]:


# Random Forest
#model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,
#                               oob_score=True, random_state=10) 
                                # Random state to get a consistent score
#model.fit(reduced_train_df, train_label)
#score = model.score(reduced_test_df, test_label)
#print('Random Forest model score is', score)


# ##Notes on Random Forest
# The score I got from running the code on my PC was 0.988805122379

# ##Results
# 
# When the models were run with default settings, they produced very good scores. The KNN model scored 0.9884, the SVC model scored 0.9889 and the Random Forest model scored 0.9889 too. The parameters of each model could be optimised, but the improvement in score would not be large.
