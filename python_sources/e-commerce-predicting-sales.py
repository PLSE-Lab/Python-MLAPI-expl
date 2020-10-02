#!/usr/bin/env python
# coding: utf-8

# We live in the world of e-commerce. We see tons of different stores here and there through the web. Internet made it possible to trade with anyone and everywhere. We can buy goods without leaving our house, we can compare prices in different stores within seconds, we can find what we really want and do not accept just the first more or less suitable offer. And I believe it would be really interesting to look at this world through the data it produces. That's why I decided to play around with e-commerce numbers and try to understand it better.
# 
# The data used in this analysis is taken from Kaggle dataset ["E-Commerce Data | Actual transactions of UK retailer"](https://www.kaggle.com/carrie1/ecommerce-data). 
# 
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

# As always, we start our analysis by setting up our environment and by importing necessary libraries.
# 
# We import standard numpy and pandas to be able to perform analysis with Python, also we need data visualization libraries matplotlib and seaborn to output interesting visual findings, aaaaand some settings to make our kernel prettier.

# # 1. Import libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import our data using *.read_csv()* method and we also add a parameter "encoding='latin'" as default encoding engine wasn't able to process this particular dataset. So next time you have difficulties importing data and everything seems to be correct and OK, check out encoding. That might save you some time of googling to try to understand what's wrong.

# In[ ]:


# for Kaggle
df = pd.read_csv('/kaggle/input/ecommerce-data/data.csv', encoding='latin')
# df = pd.read_csv('data.csv', encoding='latin')
df.head()


# Just by looking at first 5 rows of our table we can understand the structure and datatypes present in our dataset. We can notice that we will have to deal with timeseries data, integers and floats, categorical and text data.

# # 2. Exploratory data analysis

# Every data science project starts with EDA as we have to understand what do we have to deal with. I divide EDA into 2 types: visual and numerical. Let's start with numerical as the simple pndas method *.describe()* gives us a lot of useful information.

# ## 2.1. Quick statistical overview

# In[ ]:


df.describe()


# Just a quick look at data with *.describe()* method gives us a lot of space to think. We see negative quantities and prices, we can see that not all records have CustomerID data, we can also see that the majority of transactions are for quantites from 3 to 10 items, majority of items have price up to 5 pounds and that we have a bunch of huge outliers we will have to deal with later.

# ## 2.2. Dealing with types

# *.read_csv()* method performs basic type check, but it doesn't do that perfectly. That's why it is much better to deal with data types in our dataframe before any modifications to prevent additional difficulties. Every pandas dataframe has an attribute *.dtypes* which will help us understand what we currently have and what data has to be casted to correct types.

# In[ ]:


df.dtypes


# If we have datetime data it's better to cast it to datetime type. We don't touch InvoiceNo for now as it seems like data in this column has not only numbers. (we saw just first 5 rows, while pandas during import scanned all the data and found that the type here is not numerical).

# In[ ]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.set_index('InvoiceDate')


# ## 2.3. Dealing with null values

# Next and very important step is dealing with missing values. Normally if you encounter null values in the dataset you have to understand nature of those null values and possible impact they could have on the model. There are few strategies that we can use to fix our issue with null values: 
# * delete rows with null values
# * delete the feature with null values
# * impute data with mean or median values or use another imputing strategy (method *.fillna()*)
# 
# Let's check out what we have here.

# In[ ]:


df.isnull().sum()


# CustomerID has too much null values and this feature cannot predict a lot so we can just drop it. Also it could be reasonable to create another feature "Amount of orders per customer", but.... next time ;)

# In[ ]:


df = df.drop(columns=['CustomerID'])


# Let's check out what kind of nulls we have in Description

# In[ ]:


df[df['Description'].isnull()].head()


# The data in these rows is pretty strange as UnitPrice is 0, so these orders do not generate any sales. I think, we can impute it with "UNKNOWN ITEM" at the moment and deal with those later during the analysis.

# In[ ]:


df['Description'] = df['Description'].fillna('UNKNOWN ITEM')
df.isnull().sum()


# ## 2.4. Checking out columns separately

# Also it makes sense to go feature by feature and check what pitfalls we have in our data and also to understand our numbers better. 

# Let's continue checking Description column. Here we can see items that were bought most often. 

# In[ ]:


df['Description'].value_counts().head()


# Here we can see our best selling products, items that appear in orders the most often. Also to make it visually more appealing let's create a bar chart for 15 top items.

# In[ ]:


item_counts = df['Description'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(item_counts.index, item_counts.values, palette=sns.cubehelix_palette(15))
plt.ylabel("Counts")
plt.title("Which items were bought more often?");
plt.xticks(rotation=90);


# In[ ]:


df['Description'].value_counts().tail()


# We also notice from above code that valid items are normally uppercased and non-valid or cancelations are in lower case

# In[ ]:


df[~df['Description'].str.isupper()]['Description'].value_counts().head()


# Quick check of the case of letters in Description says that there are some units with lower case letters in their name and also that lower case records are for canceled items. Here we can understand that data management in the store can be improved.

# In[ ]:


lcase_counts = df[~df['Description'].str.isupper()]['Description'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(lcase_counts.index, lcase_counts.values, palette=sns.color_palette("hls", 15))
plt.ylabel("Counts")
plt.title("Not full upper case items");
plt.xticks(rotation=90);


# ALso checking out stoke codes, looks like they are deeply correlated with descriptions - which makes perfect sense.

# In[ ]:


df['StockCode'].value_counts().head()


# In[ ]:


stock_counts = df['StockCode'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(stock_counts.index, stock_counts.values, palette=sns.color_palette("GnBu_d"))
plt.ylabel("Counts")
plt.title("Which stock codes were used the most?");
plt.xticks(rotation=90);


# Checking out also InvoiceNo feature.

# In[ ]:


df['InvoiceNo'].value_counts().tail()


# In[ ]:


inv_counts = df['InvoiceNo'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(inv_counts.index, inv_counts.values, palette=sns.color_palette("BuGn_d"))
plt.ylabel("Counts")
plt.title("Which invoices had the most items?");
plt.xticks(rotation=90);


# In[ ]:


df[df['InvoiceNo'].str.startswith('C')].describe()


# Looks like Invoices that start with 'C' are the "Canceling"/"Returning" invoices. This resolves the mistery with negative quantities. 
# 
# Although, we should've gotten deeper into analysis of those returns, for the sake of simplicity let's just ignore those values for the moment.
# 
# We can actually start a separate project based on that data and predict the returning/cancelling rates for the store.

# In[ ]:


df = df[~df['InvoiceNo'].str.startswith('C')]


# In[ ]:


df.describe()


# During exploratory data analysis we can go back to the same operations and checks, just to understand how our actions affected the dataset. EDA is the series of repetitive tasks to understand better our data. And here, for example we get back to *.describe()* method to get an overall picture of our data after some manipulations. 
# 
# We still see negative quantities and negative prices, let's get into those records.
# 
# 

# In[ ]:


# df[df['Quantity'] < 0]
df[df['Quantity'] < 0].head()


# Here we can see that other "Negative quantities" appear to be damaged/lost/unknown items. Again, we will just ignore them for the sake of simplicity of analysis for this project.

# In[ ]:


df = df[df['Quantity'] > 0]
df.describe()


# We also see negative UnitPrice, which is not normal as well. Let's check this out.

# In[ ]:


df[df['UnitPrice'] < 0].describe()


# In[ ]:


df[df['UnitPrice'] == -11062.06]


# As there are just two rows, let's ignore them for the moment (description gives us enough warnings, althoug we still need some context to understand it better)

# In[ ]:


df = df[df['UnitPrice'] > 0]
df.describe()


# As we have finished cleaning our data and removed all suspicious records we can start creating some new features for our model. Let's start with the most obvious one - Sales. We have quantities, we have prices - we can calculate the revenue.

# In[ ]:


df['Sales'] = df['Quantity'] * df['UnitPrice']
df.head()


# # 3. Visual EDA

# In[ ]:


plt.figure(figsize=(3,6))
sns.countplot(df[df['Country'] == 'United Kingdom']['Country'])
plt.xticks(rotation=90)


# In[ ]:


plt.figure(figsize=(18,6))
sns.countplot(df[df['Country'] != 'United Kingdom']['Country'])
plt.xticks(rotation=90)


# In[ ]:


uk_count = df[df['Country'] == 'United Kingdom']['Country'].count()
all_count = df['Country'].count()
uk_perc = uk_count/all_count
print(str('{0:.2f}%').format(uk_perc*100))


# From above plots and calculations we can see that vast majority of sales were made in UK and just 8.49% went abroad. We can say our dataset is skewed to the UK side :D.

# ## 3.1. Detecting outliers

# There are few different methods to detect outliers: box plots, using [IQR](https://en.wikipedia.org/wiki/Interquartile_range), scatter plot also works in some cases (and this is one of those). Also, detecting outliers using scatter plot is pretty intuitive. You plot your data and remove data points that visually are definitely out of range. Like in the chart below.

# In[ ]:


plt.figure(figsize=(18,6))
plt.scatter(x=df.index, y=df['Sales'])


# Let's remove obvious outliers

# In[ ]:


df = df[df['Sales'] < 25000]
plt.figure(figsize=(18,6))
plt.scatter(x=df.index, y=df['Sales'])
plt.xticks(rotation=90)


# After removing obvious outliers we still see some values that are out of normal distribution. To understand better the distribution of our data let's check out different percentiles of our numeric features. 

# In[ ]:


df.quantile([0.05, 0.95, 0.98, 0.99, 0.999])


# We can see that if we remove top 2% of our data points we will get rid of absolute outliers and will have more balaced dataset.

# In[ ]:


df_quantile = df[df['Sales'] < 125]
plt.scatter(x=df_quantile.index, y=df_quantile['Sales'])
plt.xticks(rotation=90)


# In[ ]:


df_quantile.describe()


# Looks like our data is almost ready for modelling. We performed a clean up, we removed outliers that were disturbing the balance of our dataset, we removed invalid records - now our data looks much better! and it doesn't lose it's value.

# ## 3.2. Visually checking distribution of numeric features

# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['UnitPrice'] < 10]['UnitPrice'].values, kde=True, bins=10)


# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['UnitPrice'] < 5]['UnitPrice'].values, kde=True, bins=10, color='green')


# From these histograms we can see that vast majority of items sold in this store has low price range - 0 to 3 pounds. 

# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Quantity'] <= 30]['Quantity'], kde=True, bins=10, color='red')


# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Quantity'] <= 15]['Quantity'], kde=True, bins=10, color='orange')


# From these histograms we that people bought normally 1-5 items or 10-12 - maybe there were some kind of offers for sets?

# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Sales'] < 60]['Sales'], kde=True, bins=10, color='purple')


# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Sales'] < 30]['Sales'], kde=True, bins=10, color='grey')


# From these histograms we can understand that majority of sales per order were in range 1-15 pounds each.

# ## 3.3. Analysing sales over time

# In[ ]:


df_ts = df[['Sales']]
df_ts.head()


# As we can see every invoice has it's own timestamp (definitely based on time the order was made). We can resample time data by, for example weeks, and try see if there is any patterns in our sales.

# In[ ]:


plt.figure(figsize=(18,6))
df_resample = df_ts.resample('W').sum()
df_resample.plot()


# That week with 0 sales in January looks suspicious, let's check it closer

# In[ ]:


df_resample['12-2010':'01-2011']


# Now it makes sense - possibly, during the New Year holidays period the store was closed and didn't process orders, that's why they didn't make any sales.

# # 4. Preparing data for modeling and feature creation

# Now it comes the most fun part of the project - building a model. To do this we will need to create few more additional features to make our model more sophisticated.

# In[ ]:


df_clean = df[df['UnitPrice'] < 15]
df_clean.describe()


# In[ ]:


df_clean.index


# ## 4.1. Quantity per invoice feature

# A feature that could influence the sales output could be "Quantity per invoice". Let's find the data for this feature.

# In[ ]:


df_join = df_clean.groupby('InvoiceNo')[['Quantity']].sum()


# In[ ]:


df_join = df_join.reset_index()
df_join.head()


# In[ ]:


df_clean['InvoiceDate'] = df_clean.index
df_clean = df_clean.merge(df_join, how='left', on='InvoiceNo')
df_clean = df_clean.rename(columns={'Quantity_x' : 'Quantity', 'Quantity_y' : 'QuantityInv'})
df_clean.tail(15)


# In[ ]:


df_clean.describe()


# In[ ]:


df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])


# In[ ]:


df_clean.dtypes


# ## 4.2. Bucketizing Quantity and UnitPrice features

# Based on the EDA done previously we can group these features into 6 buckets for Quantity and 5 for UnitePrice using pandas .cut() method.

# In[ ]:


bins_q = pd.IntervalIndex.from_tuples([(0, 2), (2, 5), (5, 8), (8, 11), (11, 14), (15, 5000)])
df_clean['QuantityRange'] = pd.cut(df_clean['Quantity'], bins=bins_q)
bins_p = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 20)])
df_clean['PriceRange'] = pd.cut(df_clean['UnitPrice'], bins=bins_p)
df_clean.head()


# ## 4.3. Extracting and bucketizing dates

# We have noticed that depends on a season gifts sell differently: pick of sales is in the Q4, then it drastically drops in Q1 of the next year and continues to grow till its new pick in Q4 again. From this observation we can create another feature that could improve our model.

# In[ ]:


df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean.head()


# In[ ]:


bins_d = pd.IntervalIndex.from_tuples([(0,3),(3,6),(6,9),(9,12)])
df_clean['DateRange'] = pd.cut(df_clean['Month'], bins=bins_d, labels=['q1','q2','q3','q4'])
df_clean.tail()


# # 5. Building a model

# ## 5.1. Splitting data into UK and non-UK

# We have to analyze these 2 datasets separately to have more standardized data for a model, because there can be some patterns that work for other countries and do not for UK or vise versa. Also a hypothesis to test - does the model built for UK performs good on data for other countries? 

# In[ ]:


df_uk = df_clean[df_clean['Country'] == 'United Kingdom']
df_abroad = df_clean[df_clean['Country'] != 'United Kingdom']


# In[ ]:


df_uk.head()


# ## 5.2. Extracting features and creating dummy variables

# In[ ]:


df_uk_model = df_uk[['Sales', 'QuantityInv', 'QuantityRange', 'PriceRange', 'DateRange']]
df_uk_model.head()


# In[ ]:


df_data = df_uk_model.copy()
df_data = pd.get_dummies(df_data, columns=['QuantityRange'], prefix='qr')
df_data = pd.get_dummies(df_data, columns=['PriceRange'], prefix='pr')
df_data = pd.get_dummies(df_data, columns=['DateRange'], prefix='dr')
df_data.head()


# ## 5.3. Scaling

# As the majority of our features are in 0-1 range it would make sense to scale "QuantityInv" feature too. In general, scaling features is normally a good idea.

# In[ ]:


from sklearn.preprocessing import scale
df_data['QuantityInv'] = scale(df_data['QuantityInv'])


# ## 5.4. Train-Test Split

# Now we have to split our data into train-test data to be able to train our model and validate its capabilities.

# In[ ]:


y = df_data['Sales']
X = df_data.drop(columns=['Sales'])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# ## 5.5. Testing and validating different models

# Here we use GridSearch and CrossValidation to test three types of regressors: Linear, DecisionTree and RandomForest. This can take a while...

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Linear Regression
fit_intercepts = [True, False]
param_grid_linear = dict(fit_intercept=fit_intercepts)
linear_model = LinearRegression()

# Decision Tree
min_tree_splits = range(2,3)
min_tree_leaves = range(2,3)
param_grid_tree = dict(min_samples_split=min_tree_splits,
                       min_samples_leaf=min_tree_leaves)
tree_model = DecisionTreeRegressor()

# Random Forest
estimators_space = [100]
min_sample_splits = range(2,4)
min_sample_leaves = range(2,3)
param_grid_forest = dict(min_samples_split=min_sample_splits,
                       min_samples_leaf=min_sample_leaves,
                       n_estimators=estimators_space)
forest_model = RandomForestRegressor()

cv = 5

models_to_test = ['LinearRegression','DecisionTreeRegressor','RandomForest']
regression_dict = dict(LinearRegression=linear_model,
                       DecisionTreeRegressor=tree_model,
                       RandomForest=forest_model)
param_grid_dict = dict(LinearRegression=param_grid_linear,
                       DecisionTreeRegressor=param_grid_tree,
                       RandomForest=param_grid_forest)

score_dict = {}
params_dict = {}
mae_dict = {}
mse_dict = {}
r2_dict = {}
best_est_dict = {}

for model in models_to_test:
  regressor = GridSearchCV(regression_dict[model], param_grid_dict[model], cv=cv, n_jobs=-1)

  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)

  # Print the tuned parameters and score
  print(" === Start report for regressor {} ===".format(model))
  score_dict[model] = regressor.best_score_
  print("Tuned Parameters: {}".format(regressor.best_params_)) 
  params_dict = regressor.best_params_
  print("Best score is {}".format(regressor.best_score_))

  # Compute metrics
  mae_dict[model] = mean_absolute_error(y_test, y_pred)
  print("MAE for {}".format(model))
  print(mean_absolute_error(y_test, y_pred))
  mse_dict[model] = mean_squared_error(y_test, y_pred)
  print("MSE for {}".format(model))
  print(mean_squared_error(y_test, y_pred))
  r2_dict[model] = r2_score(y_test, y_pred)
  print("R2 score for {}".format(model))
  print(r2_score(y_test, y_pred))
  print(" === End of report for regressor {} === \n".format(model))
  
  # Add best estimator to the dict
  best_est_dict[model] = regressor.best_estimator_


# In[ ]:


# Creating summary report
summary_cols = ['Best Score']
summary = pd.DataFrame.from_dict(r2_dict, orient='index')
summary.index.name = 'Regressor'
summary.columns = summary_cols
summary = summary.reset_index()

# Visualizing results
plt.figure(figsize=(12,4))
plt.xlabel('Best score')
plt.title('Regressor Comparison')

sns.barplot(x='Best Score', y='Regressor', data=summary)


# # Conclusions
# 

# This is a basic analysis of a transactions dataset with a model that predicts sales. Still a lot of things can be improved:
# 
# 
# *   Perform cluster analysis and create features based on it
# *   Make a deeper split of dates
# *   Get more insights from Descriptions and Stock numbers
# *   Compare domestic and abroad sales
# *   Try deep learning models
# 
# Also we can play much more with tuning of hyperparameters of our models and give it more time for training.
# 
# Random Forest Regressor appears to be the best model for our prediction with R2 score more than 0.6 which is not that bad. 
# 
# 
