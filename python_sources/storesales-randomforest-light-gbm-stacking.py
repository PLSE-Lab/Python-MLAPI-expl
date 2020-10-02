#!/usr/bin/env python
# coding: utf-8

# # Import Libraries, Datasets & Declare Functions

# In[ ]:


# Import libraries | Standard
import pandas as pd
import numpy as np
import os
import datetime
import warnings
from time import time

# Import libraries | Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import libraries | Sk-learn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import xgboost as xgb
from lightgbm import LGBMRegressor

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


pd.set_option('display.max_columns', None)  


# In[ ]:


#pd.set_option('display.max_rows', None)


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


def distribution(data, features, transformed = False):
    """
    Visualization code for displaying distributions of features
    """
    
    # Create figure
    fig = plt.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(features):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Data Features",             fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Distributions of Continuous Data Features",             fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


# In[ ]:


def eval_train_predict(learner, sample_size, train_X, train_y, test_X, test_y, transform_y, log_constant): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set       
       - train_X: features training set
       - train_y: sales training set
       - test_X: features testing set
       - test_y: sales testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data
    start = time() # Get start time
    learner = learner.fit(train_X[:sample_size], train_y[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['time_train'] = end - start
        
    # Get the predictions on the test set(X_test),
    start = time() # Get start time
    predictions = learner.predict(test_X)
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['time_pred'] = end - start
            
    # Compute Weighted Mean Absolute Error on Test Set
    if transform_y == 'log':
        results['WMAE'] = weighted_mean_absolute_error(np.exp(test_y) - 1 - log_constant, 
                                                       np.exp(predictions) - 1 - log_constant, 
                                                       compute_weights(test_X['IsHoliday']))
    else:
        results['WMAE'] = weighted_mean_absolute_error(test_y, predictions, compute_weights(test_X['IsHoliday']))
                   
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# In[ ]:


def eval_visualize(results):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
    """
  
    # Create figure
    fig, ax = plt.subplots(1, 3, figsize = (18,8))

    # Constants
    bar_width = 0.1
    colors = ['#A00000','#00A0A0','#00A000','#E3DAC9','#555555', '#87CEEB']
    metrics = ['time_train', 'time_pred', 'WMAE']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(metrics):
            # Creative plot code
            ax[j%3].bar(0+k*bar_width, results[learner][0][metric], width = bar_width, color = colors[k])
            ax[j%3].set_xlabel("Models")
            ax[j%3].set_xticklabels([''])
                
    # Add unique y-labels
    ax[0].set_ylabel("Time (in seconds)")
    ax[1].set_ylabel("Time (in seconds)")
    ax[2].set_ylabel("WMAE")
    
    # Add titles
    ax[0].set_title("Model Training")
    ax[1].set_title("Model Predicting")
    ax[2].set_title("WMAE on Testing Set")
 
    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.43),                loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    plt.suptitle("Performance Metrics for Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()
    plt.show()


# In[ ]:


def train_predict(learner, train_X, train_y, test_X, test_y, transform_y, log_constant, verbose=0): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - train_X: features training set
       - train_y: sales training set
       - test_X: features testing set
       - test_y: sales testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data
    start = time() # Get start time
    learner = learner.fit(train_X, train_y)
    end = time() # Get end time
    
    # Calculate the training time
    results['time_train'] = end - start
        
    # Get the predictions on the test set(X_test),
    start = time() # Get start time
    predictions = learner.predict(test_X)
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['time_pred'] = end - start
            
    # Compute Weighted Mean Absolute Error on Test Set
    if transform_y == 'log':
        results['WMAE'] = weighted_mean_absolute_error(np.exp(test_y) - 1 - log_constant, 
                                                       np.exp(predictions) - 1 - log_constant, 
                                                       compute_weights(test_X['IsHoliday']))
    else:
        results['WMAE'] = weighted_mean_absolute_error(test_y, predictions, compute_weights(test_X['IsHoliday']))
    

    #Extract the feature importances
    importances = learner.feature_importances_

    # Success
    print("Learner Name :", learner.__class__.__name__)
    print("Training     :", round(results['time_train'],2), "secs /", len(train_y), "records")
    print("Predicting   :", round(results['time_pred'],2), "secs /", len(test_y), "records")
    print("Weighted MAE :", round(results['WMAE'],2))

    if verbose == 1:
        # Plot
        print("\n<Feature Importance>\n")
        feature_plot(importances, train_X, train_y, 10)

        print("\n<Feature Weightage>\n")
        topk = len(train_X.columns)
        indices = np.argsort(importances)[::-1]
        columns = train_X.columns.values[indices[:topk]]
        values = importances[indices][:topk]

        for i in range(topk):
            print('\t' + columns[i] + (' ' * (15 - len(columns[i])) + ': ' + str(values[i])))
            
        print("\n<Learner Params>\n", model.get_params())
    
    # Return the model & predictions
    return (learner, predictions)


# In[ ]:


def feature_plot(importances, train_X, train_y, topk=5):
    
    # Display the most important features
    indices = np.argsort(importances)[::-1]
    columns = train_X.columns.values[indices[:topk]]
    values = importances[indices][:topk]

    # Creat the plot
    fig = plt.figure(figsize = (18,5))
    plt.title("Normalized Weights for First " + str(topk) + " Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(topk), values, width = 0.6, align="center", color = '#00A000',           label = "Feature Weight")
    plt.bar(np.arange(topk) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0',           label = "Cumulative Feature Weight")
    plt.xticks(np.arange(topk), columns)
    plt.xlim((-0.5, 9.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.show()  


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))    
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


def compute_weights(holidays):
    return holidays.apply(lambda x: 1 if x==0 else 5)


# In[ ]:


def weighted_mean_absolute_error(pred_y, test_y, weights):
    return 1/sum(weights) * sum(weights * abs(test_y - pred_y))


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

cnt = 0
env = 'Outside Kaggle'

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        cnt += 1
        print(os.path.join(dirname, filename))
        
if cnt > 0:
    env = 'Kaggle Kernel'


# In[ ]:


print('Environment:', env)


# In[ ]:


# Read input files
if env == 'Kaggle Kernel':
    features = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv")
    stores = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
    train = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv")
    test = pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv")
else:    
    features = pd.read_csv("data/features.csv")
    stores = pd.read_csv("data/stores.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")


# train=reduce_mem_usage(train)
# test=reduce_mem_usage(test)

# # Data Exploration

# ## 1. Stores Data

# In[ ]:


stores.head()


# In[ ]:


stores.info()


# In[ ]:


stores.describe()


# In[ ]:


#missing data
total = stores.isnull().sum().sort_values(ascending=False)
percent = (stores.isnull().sum()/stores.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


stores['Size'].groupby(stores['Type']).mean()


# In[ ]:


# Create figure
#plt.figure()
#plt.scatter(stores['Type'], stores['Store'])
#plt.ylabel('Store ID')
#plt.xlabel('Store Type')

fig, ax = plt.subplots(1, 2, figsize = (15,6))
ax[0].bar(stores['Type'].unique(), stores['Size'].groupby(stores['Type']).count())
ax[0].set_ylabel('# of Stores')
ax[0].set_xlabel('Store Type')
ax[0].yaxis.grid(True, linewidth=0.3)

ax[1].scatter(stores['Type'], stores['Size'])
ax[1].scatter(stores['Type'].unique(), stores['Size'].groupby(stores['Type']).mean()) #Store Type Average Store Size Vs 
ax[1].set_ylabel('Store Size (Total / Average)')
ax[1].set_xlabel('Store Type')
ax[1].yaxis.grid(True, linewidth=0.3)

#plt.figure(figsize=(6,6))
#plt.yticks(np.arange(len(features_missing)),features_missing.index,rotation='horizontal')
#plt.xlabel('fraction of rows with missing data')
#plt.barh(np.arange(len(features_missing)), features_missing)


# In[ ]:


stores[(stores['Size'] < 40000) & (~stores['Type'].isin(['C']))]


# In[ ]:


#Explore Weekly Sales - histogram
sns.distplot(stores['Size'])


# ###### Takeaways: 
# 1. Column TYPE is a candidate for one-hot encoding. 
# 2. Most stores are of TYPE='A'. Only a few stores are of TYPE='C'.
# 3. TYPE columns seem to be linked to Store Size. Average store size of TYPE 'A' is ~ 175k, TYPE 'B' is ~ 100k and TYPE 'C' is ~40k
# 4. Four stores [3, 5, 33 & 36] whose size is < 40k, seem to have been incorrectly tagged as Types A & B

# ## 2. Features Data

# In[ ]:


features.head()


# In[ ]:


features.info()


# In[ ]:


features.describe()


# In[ ]:


#missing data
total = features.isnull().sum().sort_values(ascending=False)
percent = (features.isnull().sum()/features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# Distribution of NaNs for all columns
features_missing = features.isna().sum()/len(features) * 100


# In[ ]:


plt.figure(figsize=(6,6))
plt.yticks(np.arange(len(features_missing)),features_missing.index,rotation='horizontal')
plt.xlabel('fraction of rows with missing data')
plt.barh(np.arange(len(features_missing)), features_missing)


# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15,12))

# Plot 1: Year Vs # of Records
ax[0,0].barh(features['Date'].str.slice(start=0, stop=4).unique(), 
          features['Date'].str.slice(start=0, stop=4).value_counts())
ax[0,0].set_xlabel('# of Records')
ax[0,0].set_ylabel('Year')
ax[0,0].yaxis.grid(True, linewidth=0.3)

# Plot 2: Month Vs # of Records with Missing Values - Unemployment
ax[1,0].barh(features['Date'].str.slice(start=0, stop=7)[features['Unemployment'].isna()].unique(), 
          features['Date'].str.slice(start=0, stop=7)[features['Unemployment'].isna()].value_counts())
ax[1,0].set_xlabel('# of Records with Missing Values - Unemployment')
ax[1,0].set_ylabel('Month')
ax[1,0].yaxis.grid(True, linewidth=0.3)

# Plot 3: Month Vs # of Records with Missing Values - CPI
ax[1,1].barh(features['Date'].str.slice(start=0, stop=7)[features['CPI'].isna()].unique(), 
          features['Date'].str.slice(start=0, stop=7)[features['CPI'].isna()].value_counts())
ax[1,1].set_xlabel('# of Records with Missing Values - CPI')
ax[1,1].set_ylabel('Month')
ax[1,1].yaxis.grid(True, linewidth=0.3)

#plt.figure(figsize=(6,6))
#plt.yticks(np.arange(len(features_missing)),features_missing.index,rotation='horizontal')
#plt.xlabel('fraction of rows with missing data')
#plt.barh(np.arange(len(features_missing)), features_missing)


# In[ ]:


holidays = ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08', #Super Bowl
           '2010-09-10', '2011-09-09', '2012-09-07', '2013-02-06',  #Labor Day
           '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',  #Thanksgiving
           '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']  #Christmas


# In[ ]:


# Validate Holidays
features['IsHoliday'][features['Date'].isin(holidays)].value_counts()


# In[ ]:


features['Date'][features['IsHoliday'].isin([1])][~features['Date'].isin(holidays)].value_counts()


# In[ ]:


features[['CPI','Unemployment']].groupby([features['Store'], features['Date'].str.slice(start=0, stop=7)]).mean().head(84)


# In[ ]:


features.groupby(features['Date'].str.slice(start=0, stop=7))['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'].count()


# In[ ]:


#Explore Distribution
distribution(features, ['CPI','Unemployment'])


# In[ ]:


#Explore Distribution
distribution(features, ['Temperature','Fuel_Price'])


# In[ ]:


#Explore Distribution
distribution(features, ['MarkDown1','MarkDown2'])


# In[ ]:


#Explore Distribution
distribution(features, ['MarkDown3','MarkDown4'])


# In[ ]:


#Explore Distribution
distribution(features, ['MarkDown5'])


# ###### Takeaways: 
# 1. Data requires pre-processing
# 2. Column(s) ISHOLIDAY has been validated
# 3. Column(s) UNEMPLOYMENT & CPI have missing values for May, Jun & Jul 2013. For these columns as the values dont change significantly month on month, value from Apr 2013 would be propogated over for each store. 
# 4. Column(s) MARKDOWN* have missing values for 2010 (entire year) and 2011 (until Nov). Additionally, there are missing values for other other dates as well. 
# 5. CPI and UNEMPLOYMENT value are a bit skewed. MARKDOWN* columns are skewed. 

# ## 3. Train Data

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


# Explore Date Range
train['Date'].str.slice(start=0, stop=4).value_counts()


# In[ ]:


# Validate Holidays
train['IsHoliday'][train['Date'].isin(holidays)].value_counts()


# In[ ]:


train['Date'][train['IsHoliday'].isin([1])][~train['Date'].isin(holidays)].value_counts()


# In[ ]:


#Explore Distribution
distribution(train, ['Weekly_Sales'])


# In[ ]:


train['Store'][train['Weekly_Sales'] < 0].count()


# In[ ]:


train_outliers = pd.merge(train, stores, how='left', on=['Store'])


# In[ ]:


# Average Weekly Sales by Store Type
train_outliers.groupby(['Type'])['Weekly_Sales'].mean()


# In[ ]:


# Average Weekly Sales for possibly misclassified Stores
train_outliers = train_outliers[train_outliers['Store'].isin([3,5,33,36])]
train_outliers.groupby(['Store','Type'])['Weekly_Sales'].mean()


# In[ ]:


# Average Weekly Sales by Store Type
fig, ax = plt.subplots(1, 2, figsize = (15,6))
ax[0].bar(train_outliers['Type'].unique(), train_outliers.groupby(['Type'])['Weekly_Sales'].mean())
ax[0].set_ylabel('Average Weekly Sales')
ax[0].set_xlabel('Store Type')
ax[0].yaxis.grid(True, linewidth=0.3)

ax[1].bar([3,5,33,36], train_outliers.groupby(['Store','Type'])['Weekly_Sales'].mean())
ax[1].set_ylabel('Average Weekly Sales')
ax[1].set_xlabel('Store ID')
ax[1].yaxis.grid(True, linewidth=0.3)


# In[ ]:


train_outliers = None


# ###### Takeaways: 
# 1. Column DATE is non-numeric and is a candidate for pre-processing.
# 2. 1285 records with Weekly Sales < 0
# 3. Data spans years 2010, 2011 and 2012
# 4. As suspected above, four stores [3, 5, 33 & 36] seem to have incorrectly classified as Type A & B. Average Weekly Sales for these stores is in line with the average for Type C. Hence, these would need to be reclassified as Type C.

# ## 4. Test Data

# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


test.describe()


# In[ ]:


test['Date'].str.slice(start=0, stop=4).value_counts()


# In[ ]:


# Validate Holidays
test['IsHoliday'][test['Date'].isin(holidays)].value_counts()


# In[ ]:


test['Date'][test['IsHoliday'].isin([1])][~test['Date'].isin(holidays)].value_counts()


# ###### Takeaways: 
# 1. Column DATE is non-numeric and is a candidate for pre-processing.
# 2. Data spans years 2012 and 2013

# # Data Pre-Processing

# ## 1. Missing/Incorrect Values

# ### Stores Data | Correct Type for 4 stores

# In[ ]:


stores[stores['Store'].isin([3,5,33,36])].index


# In[ ]:


stores.iat[2, 1] = stores.iat[4, 1] = stores.iat[32, 1] = stores.iat[35, 1] = 'C'


# ### Features Data | Negative values for MarkDowns:

# In[ ]:


features['MarkDown1'] = features['MarkDown1'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown2'] = features['MarkDown2'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown3'] = features['MarkDown3'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown4'] = features['MarkDown4'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown5'] = features['MarkDown5'].apply(lambda x: 0 if x < 0 else x)


# ### Features Data | NaN values for multiple columns:

# #### Columns: CPI and Unemployment
# As noted above, columns are missing values for 3 months May, Jun & Jul 2013. Values from Apr 2019 would be propogated to records with missing values. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "# For each Store, propogate values of CPI & Unemployment to the rows with NaN values\nfor i in range(len(features)):\n\n    if features.iloc[i]['Date'] == '2013-04-26':\n        CPI_new = features.iloc[i]['CPI']\n        Unemployment_new = features.iloc[i]['Unemployment']\n    \n    if np.isnan(features.iloc[i]['CPI']):\n        features.iat[i, 9] = CPI_new\n        features.iat[i, 10] = Unemployment_new")


# #### Columns: MarkDown1, MarkDown2, MarkDown3, MarkDown4 & MarkDown5
# As noted above, columns MARKDOWN* are missing values for the whole of 2010 and 2011 (upto Nov). For each store, 2012 values would be copied over to records with missing values. Also, to facilitate the copy, new columns WEEK and YEAR would be derived from DATE.

# In[ ]:


get_ipython().run_cell_magic('time', '', "# For each date, retrive the corresponding week number\nfeatures['Week'] = 0\n\nfor i in range(len(features)):\n    features.iat[i, 12] = datetime.date(int(features.iloc[i]['Date'][0:4]), \n                                        int(features.iloc[i]['Date'][5:7]), \n                                        int(features.iloc[i]['Date'][8:10])).isocalendar()[1]")


# In[ ]:


features['Year'] = features['Date'].str.slice(start=0, stop=4)


# In[ ]:


#missing data for 2012 & 2013
total = features[features['Year'].isin(['2012','2013'])].isnull().sum().sort_values(ascending=False)
percent = (features[features['Year'].isin(['2012','2013'])].isnull().sum()/
           features[features['Year'].isin(['2012','2013'])].isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(4)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# For 2010 & 2011 records, for each store, copy over MarkDown values from 2012\n\n# Iterate through stores\nfor i in range(1, len(features['Store'].unique())):\n    \n    # For 2010, iterate through weeks 5 thru 52\n    for j in range(5, 52):\n        idx = features.loc[(features.Year == '2010') & (features.Store == i) & (features.Week == j),['Date']].index[0]\n        \n        features.iat[idx, 4] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown1']].values[0]\n        features.iat[idx, 5] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown2']].values[0]\n        features.iat[idx, 6] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown3']].values[0]\n        features.iat[idx, 7] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown4']].values[0]\n        features.iat[idx, 8] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown5']].values[0]\n        \n    # For 2011, iterate through weeks 1 thru 44\n    for j in range(1, 44):\n        idx = features.loc[(features.Year == '2011') & (features.Store == i) & (features.Week == j),['Date']].index[0]\n        \n        features.iat[idx, 4] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown1']].values[0]\n        features.iat[idx, 5] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown2']].values[0]\n        features.iat[idx, 6] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown3']].values[0]\n        features.iat[idx, 7] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown4']].values[0]\n        features.iat[idx, 8] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown5']].values[0]        ")


# In[ ]:


features.drop(columns=['Year'], axis=1, inplace=True)


# In[ ]:


# Now fill all the missing MarkDown values with 0
features.fillna(0, inplace=True)


# ### Train Data | Negative Values for Weekly Sales

# In[ ]:


train['Weekly_Sales'] = train['Weekly_Sales'].apply(lambda x: 0 if x < 0 else x)


# ## 2. Merge Datasets

# ### Merge the following datasets:
# 1. Stores + Features + Train
# 2. Stores + Features + Test
# 3. Remove duplicate columns from each dataset

# In[ ]:


train = pd.merge(train, stores, how='left', on=['Store'])
train = pd.merge(train, features, how='left', on=['Store','Date'])

test = pd.merge(test, stores, how='left', on=['Store'])
test = pd.merge(test, features, how='left', on=['Store','Date'])


# In[ ]:


train['Store'][train['IsHoliday_x'] != train['IsHoliday_y']].count()


# In[ ]:


test['Store'][test['IsHoliday_x'] != test['IsHoliday_y']].count()


# In[ ]:


train.drop(columns=['IsHoliday_y'], axis=1, inplace=True)
test.drop(columns=['IsHoliday_y'], axis=1, inplace=True)


# In[ ]:


train.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
test.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)


# ## 3. Feature Engineering

# ### Column #1: IsHoliday
# Column has boolean values and would ned converted to numeric. 

# In[ ]:


train['IsHoliday'] = train['IsHoliday'].apply(lambda x: 1 if x==True else 0)
test['IsHoliday'] = test['IsHoliday'].apply(lambda x: 1 if x==True else 0)


# ### Column #2: Type
# Column is categorical and would be converted to numeric via one-hot encoding. 

# In[ ]:


train = pd.get_dummies(train, columns=['Type'])
test = pd.get_dummies(test, columns=['Type'])


# ### Column #3: Week
# New numeric column being created to replace YEAR. 

# In[ ]:


train['Week'] = test['Week'] = 0


# In[ ]:


get_ipython().run_cell_magic('time', '', "# For each date, retrive the corresponding week number\nfor i in range(len(train)):\n    train.iat[i, 15] = datetime.date(int(train.iloc[i]['Date'][0:4]), \n                                     int(train.iloc[i]['Date'][5:7]), \n                                     int(train.iloc[i]['Date'][8:10])).isocalendar()[1]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# For each date, retrive the corresponding week number\nfor i in range(len(test)):\n    test.iat[i, 14] = datetime.date(int(test.iloc[i]['Date'][0:4]), \n                                    int(test.iloc[i]['Date'][5:7]), \n                                    int(test.iloc[i]['Date'][8:10])).isocalendar()[1]")


# In[ ]:


# Create checkpoint
train.to_csv('train_prescaled.csv', index=False)
test.to_csv('test_prescaled.csv', index=False)


# In[ ]:


# Restore checkpoint
train = pd.read_csv("train_prescaled.csv")
test = pd.read_csv("test_prescaled.csv")


# In[ ]:


# Create Submission dataframe
submission = test[['Store', 'Dept', 'Date']].copy()
submission['Id'] = submission['Store'].map(str) + '_' + submission['Dept'].map(str) + '_' + submission['Date'].map(str)
submission.drop(['Store', 'Dept', 'Date'], axis=1, inplace=True)


# In[ ]:


train['Year'] = train['Date'].str.slice(start=0, stop=4)
test['Year'] = test['Date'].str.slice(start=0, stop=4)


# In[ ]:


# Drop non-numeric columns
train.drop(columns=['Date'], axis=1, inplace=True)
test.drop(columns=['Date'], axis=1, inplace=True)


# ### Log Transform Skewed Features

# In[ ]:


skewed = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
train[skewed] = train[skewed].apply(lambda x: np.log(x + 1))
test[skewed] = test[skewed].apply(lambda x: np.log(x + 1))


# MarkDown1_min = abs(min(train['MarkDown1'].min(),test['MarkDown1'].min()))
# MarkDown2_min = abs(min(train['MarkDown2'].min(),test['MarkDown2'].min()))
# MarkDown3_min = abs(min(train['MarkDown3'].min(),test['MarkDown3'].min()))
# MarkDown4_min = abs(min(train['MarkDown4'].min(),test['MarkDown4'].min()))
# MarkDown5_min = abs(min(train['MarkDown5'].min(),test['MarkDown5'].min()))

# train['MarkDown1'] = train['MarkDown1'].apply(lambda x: np.log(x + 1 + MarkDown1_min))
# train['MarkDown2'] = train['MarkDown2'].apply(lambda x: np.log(x + 1 + MarkDown2_min))
# train['MarkDown3'] = train['MarkDown3'].apply(lambda x: np.log(x + 1 + MarkDown3_min))
# train['MarkDown4'] = train['MarkDown4'].apply(lambda x: np.log(x + 1 + MarkDown4_min))
# train['MarkDown5'] = train['MarkDown5'].apply(lambda x: np.log(x + 1 + MarkDown5_min))
# 
# test['MarkDown1'] = test['MarkDown1'].apply(lambda x: np.log(x + 1 + MarkDown1_min))
# test['MarkDown2'] = test['MarkDown2'].apply(lambda x: np.log(x + 1 + MarkDown2_min))
# test['MarkDown3'] = test['MarkDown3'].apply(lambda x: np.log(x + 1 + MarkDown3_min))
# test['MarkDown4'] = test['MarkDown4'].apply(lambda x: np.log(x + 1 + MarkDown4_min))
# test['MarkDown5'] = test['MarkDown5'].apply(lambda x: np.log(x + 1 + MarkDown5_min))

# In[ ]:


log_constant = 0


# In[ ]:


train['Weekly_Sales'] = train['Weekly_Sales'].apply(lambda x: np.log(x + 1 + log_constant))


# In[ ]:


distribution(train, ['Weekly_Sales'])


# ### Analyze Feature Correlation

# In[ ]:


colormap = plt.cm.RdBu
corr = train.astype(float).corr()

plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.set(font_scale=0.9)
sns.heatmap(round(corr,2),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)


# In[ ]:


corr_cutoff = 0.8
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= corr_cutoff:
            if columns[j]:
                columns[j] = False
                
selected_columns = train.columns[columns]
highcorr_columns = train.columns.difference(selected_columns)


# In[ ]:


highcorr_columns


# In[ ]:


train.drop(columns=highcorr_columns, axis=1, inplace=True)
test.drop(columns=highcorr_columns, axis=1, inplace=True)


# ###### Takeaway: 
# 1. MarkDown4 and Type_A are highly correlated to other existing features and have been dropped. 

# ### Split Training dataset into Train & Validation

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(train.drop('Weekly_Sales', axis = 1), 
                                                  train['Weekly_Sales'], 
                                                  test_size = 0.2, 
                                                  random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(train_X.shape[0]))
print("Validation set has {} samples.".format(val_X.shape[0]))


# In[ ]:


# Validate shape
train_X.shape, train_y.shape, val_X.shape, val_y.shape, test.shape


# ### Scale Datasets

# In[ ]:


# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

numerical = ['Store', 'Dept', 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price', 
             'CPI', 'Unemployment', 'Week', 'Type_B', 'Type_C',
             'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']

train_scaled = pd.DataFrame(data = train_X)
train_scaled[numerical] = scaler.fit_transform(train_X[numerical])

# Show an example of a record with scaling applied
display(train_scaled.head(n = 5))


# In[ ]:


val_scaled = pd.DataFrame(data = val_X)
val_scaled[numerical] = scaler.transform(val_X[numerical])

# Show an example of a record with scaling applied
display(val_scaled.head(n = 5))


# In[ ]:


test_scaled = pd.DataFrame(data = test)
test_scaled[numerical] = scaler.transform(test[numerical])

# Show an example of a record with scaling applied
display(test_scaled.head(n = 5))


# In[ ]:


# Free up memory
train = test = features = stores = None


# In[ ]:


# Create checkpoint
train_scaled.to_csv('train_X_scaled.csv', index=False)
val_scaled.to_csv('val_X_scaled.csv', index=False)
train_y.to_csv('train_y.csv', index=False, header=['Weekly_Sales'])
val_y.to_csv('val_y.csv', index=False, header=['Weekly_Sales'])
test_scaled.to_csv('test_X_scaled.csv', index=False)


# In[ ]:


# Restore checkpoint
train_scaled = pd.read_csv("train_X_scaled.csv")
val_scaled = pd.read_csv("val_X_scaled.csv")
train_y = pd.read_csv("train_y.csv")
val_y = pd.read_csv("val_y.csv")
test_scaled = pd.read_csv("test_X_scaled.csv")


# In[ ]:


# Reduce memory usage
#train_scaled=reduce_mem_usage(train_scaled)
#test_scaled=reduce_mem_usage(test_scaled)


# In[ ]:


train_X = train_scaled
val_X = val_scaled


# train_scaled.drop(columns=['Temperature', 'Fuel_Price'], axis=1, inplace=True)
# test_scaled.drop(columns=['Temperature', 'Fuel_Price'], axis=1, inplace=True)

# train_y = train_scaled['Weekly_Sales']
# train_X = train_scaled.drop('Weekly_Sales', axis = 1)
# 
# val_y = val_scaled['Weekly_Sales']
# val_X = val_scaled.drop('Weekly_Sales', axis = 1)

# In[ ]:


# Free up memory
train_scaled = val_scaled = None


# In[ ]:


# Convert Dataframe to Series
train_y = train_y.iloc[:,0]
val_y = val_y.iloc[:,0]


# ## 4. Modelling

# ### Select and evaluate candidate models

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Initialize base models\nmodel_A = LinearRegression()\nmodel_B = ElasticNet(random_state=1)\nmodel_C = RandomForestRegressor(random_state=1)\nmodel_D = GradientBoostingRegressor(random_state=1)\nmodel_E = xgb.XGBRegressor()\nmodel_F = LGBMRegressor(random_state=1)\n\nsamples = len(train_y) # 100% of training set\n\n# Collect results on the learners\nresults = {}\nfor model in [model_A, model_B, model_C, model_D, model_E, model_F]:\n    model_name = model.__class__.__name__\n    results[model_name] = {}\n    for i, samples in enumerate([samples]):\n        results[model_name][i] = eval_train_predict(model, samples, train_X, train_y, val_X, val_y, 'log', log_constant)")


# In[ ]:


# Evaluate Metrics
eval_visualize(results)


# In[ ]:


results


# ###### Takeaway: With respect to WMAE, Random Forest and Light GBM have turned out to be the top performing base models and would be further evaluated.

# ### Evaluate Random Forest (Ensemble)

# #### Default Model

# In[ ]:


model_rf_base = RandomForestRegressor(random_state=42, verbose=1)


# In[ ]:


model_rf_base, pred_y_rf_val = train_predict(model, train_X, train_y, val_X, val_y, 'log', log_constant, verbose=1)


# In[ ]:


pred_y_rf_test = model_rf_base.predict(test_scaled)


# In[ ]:


param_grid = { 
    'n_estimators': [10, 50, 100, 150],
    'max_features': [None, 'auto'],
    'bootstrap': [True, False],
    'max_depth':[None],
    'random_state': [42], 
    'verbose': [1]
}


# In[ ]:


#%%time
#CV = GridSearchCV(estimator=model_rf_base, param_grid=param_grid, cv=2, verbose=1)
#CV.fit(train_X, train_y)


# In[ ]:


#CV.best_params_ # latest


# In[ ]:


# Using best params from GridSearch
#model.set_params(**CV.best_params_)


# #### Tuned Model

# In[ ]:


model = RandomForestRegressor(random_state=42, 
                              n_estimators=150, 
                              bootstrap=True, 
                              max_features=None, 
                              max_depth=None, 
                              min_samples_leaf=1,
                              min_samples_split=3,
                              verbose=1)


# In[ ]:


model, pred_y_rf_val = train_predict(model, train_X, train_y, val_X, val_y, 'log', log_constant, verbose=1)


# In[ ]:


pred_y_rf_test = model.predict(test_scaled)


# ### Evaluate Light GBM (Boosting)

# #### Default Model

# In[ ]:


# Default model
model = LGBMRegressor()


# In[ ]:


model, pred_y_lgbm_val = train_predict(model, train_X, train_y, val_X, val_y, 'log', log_constant, verbose=1)


# In[ ]:


param_grid = {
    'boosting_type': ['gbdt'], 
    'objective': ['regression'],
    'random_state': [42],
    'min_data_in_leaf':[3],
    'min_depth':[2],
    'learning_rate': [0.3],
    #'n_estimators': [1000, 3000],
    'n_estimators': [3000],
    #'num_leaves': [60, 70, 80],
    'max_bin': [150,200,255,300]
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "#CV_lgbm = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, scoring='neg_mean_absolute_error')\n#CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)\n#CV.fit(train_X, train_y)")


# In[ ]:


#print("Best parameter (CV score=%0.3f):" % CV.best_score_)
#print(CV.best_params_)


# In[ ]:


# Using best params from GridSearch
#model.set_params(**CV.best_params_)


# #### Tuned Model

# model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#                       importance_type='split', learning_rate=0.3, max_depth=-1,
#                       min_child_samples=5, min_child_weight=0.001, min_data_in_leaf=2,
#                       min_depth=3, min_split_gain=0.0, n_estimators=3000, n_jobs=-1,
#                       num_leaves=80, objective='regression', random_state=42,
#                       reg_alpha=0.1, reg_lambda=2, silent=True, subsample=1.0,
#                       subsample_for_bin=200000, subsample_freq=0,
#                       verbose=1)
# #Weighted MAE : 1275.72

# model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#                       importance_type='split', learning_rate=0.3, max_depth=-1,
#                       min_child_samples=5, min_child_weight=0.001, min_data_in_leaf=2,
#                       min_depth=3, min_split_gain=0.0, n_estimators=3000, n_jobs=-1,
#                       num_leaves=80, objective='regression', random_state=42,
#                       reg_alpha=0.1, reg_lambda=2, silent=True, subsample=1.0,
#                       subsample_for_bin=200000, subsample_freq=0,
#                       verbose=1)
# #Weighted MAE : 1324.72

# In[ ]:


model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.3, max_bin=150,
       max_depth=-1, min_child_samples=5, min_child_weight=0.001,
       min_data_in_leaf=3, min_depth=2, min_split_gain=0.0,
       n_estimators=3000, n_jobs=-1, num_leaves=80, objective='regression',
       random_state=42, reg_alpha=0.1, reg_lambda=2, silent=True,
       subsample=1.0, subsample_for_bin=200000, subsample_freq=0,
       verbose=1)
#Weighted MAE : 1238.72


# In[ ]:


model, pred_y_lgbm_val = train_predict(model, train_X, train_y, val_X, val_y, 'log', log_constant, verbose=1)


# In[ ]:


pred_y_lgbm_test = model.predict(test_scaled)


# ## Model Stacking

# In[ ]:


# Blend the results of the two regressors and save the prediction to a CSV file.
pred_y_val = ((np.exp(pred_y_rf_val) - 1 - log_constant) * 0.7) + ((np.exp(pred_y_lgbm_val) - 1 - log_constant) * 0.3)
pred_y = ((np.exp(pred_y_rf_test) - 1 - log_constant) * 0.7) + ((np.exp(pred_y_lgbm_test) - 1 - log_constant) * 0.3)


# In[ ]:


val_y = np.exp(val_y) - 1 - log_constant


# In[ ]:


# make predictions
print("Weighted Mean Absolute Error: ", weighted_mean_absolute_error(pred_y_val, val_y, compute_weights(val_X['IsHoliday'])))


# In[ ]:


submission['Weekly_Sales'] = pred_y


# In[ ]:


submission[['Id','Weekly_Sales']].to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:


val_X.columns


# In[ ]:


tmp = pd.DataFrame(scaler.inverse_transform(val_X), columns = val_X.columns)


# In[ ]:


tmp = tmp.assign(weekly_sales=val_y.values)


# In[ ]:


tmp = pd.concat([tmp, pd.DataFrame(pred_y_val)], axis=1)


# In[ ]:


tmp.head(5000).to_csv('tmp5000.csv', index=False)


# In[ ]:


tmp.to_csv('tmp5000.csv', index=False)


# In[ ]:




