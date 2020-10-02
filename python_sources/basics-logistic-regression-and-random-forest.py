#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# This dataset is from https://www.kaggle.com/shebrahimi/financial-distress.
# 
# The goal is to predict whether a currently healthy company will become distressed, before it becomes distressed.
# 
# We'll use **F-1 score** as our main evaluation metric to deal with the unbalanced set. 
# 
# We'll pay particular attention to **recall** (of all companies that truly do become distressed, how often can we predict their distress before they become distressed?). 
# 
# We can imagine that if this model were being used to guide investment choices or loans, it would be much more costly to accidentally classify a bad company as a good one (false negative - make a type II error) than to miss out on a good company because we falsely thought it was distress-prone.
# 
# Note that this desire to avoid type II errors (with regards to being afraid of failing to identify "badness") is characteristic of many processes (companies that hire elite talent, universities with high admissions standards, highly-regarded VC firms).

# In[ ]:


# Basics
import sys
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Imports for data loading
# import psycopg2
# import sqlalchemy
# import imp
# import os

# Sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import TimeSeriesSplit


# In[ ]:


# secrets_filepath = '/home/casey/secrets.py'
# secrets = imp.load_source('secrets', secrets_filepath)

# # Postgres connection info
# POSTGRES_ADDRESS = secrets.psql_ad
# POSTGRES_PORT = secrets.psql_port
# POSTGRES_USERNAME = secrets.psql_username
# POSTGRES_DBNAME = secrets.psql_db
# POSTGRES_PASSWORD = secrets.psql_pw

# # Form string
# postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
#                 .format(username=POSTGRES_USERNAME, 
#                         password=POSTGRES_PASSWORD, 
#                         ipaddress=POSTGRES_ADDRESS, 
#                         port=POSTGRES_PORT, 
#                         dbname=POSTGRES_DBNAME)) 

# # Make connection
# cnx = sqlalchemy.create_engine(postgres_str)


# # Loading Data
# 
# I've loaded this into my local PostgreSQL db, but this can easily be replaced with a load from the csv file.

# In[ ]:


# companies = pd.read_sql_query('''SELECT * from casey;''', cnx)

### UNCOMMENT BELOW TO LOAD FROM FILE ###

companies = pd.read_csv('../input/Financial Distress.csv')
companies.rename(index=str, columns={"Company": "company", "Time": "time", "Financial Distress": "financial_distress"}, inplace=True)


# The Kaggle description tells us that if the number in the **financal_distress** column < -0.5, the company should be considered distressed. 
# 
# We can imagine that this might be a financial ratio of some sort - ratio of income to capital or something.

# In[ ]:


# Take a look at our loaded data to ensure all is in order
companies.head()

# Print some summaries and checks

 # shape
print(companies.shape)

# dtypes
print(companies.iloc[:5,:5].dtypes)

# check for nulls
print(companies.iloc[:5,:5].isnull().any())

# Describe
print(companies.describe(percentiles=[0.25,0.5,0.75,0.99]))


# Hm. Features x1, x7, and x81 look a little funny in terms of their maxes being much higher than their 99th percentile. If we knew what these features were we could have a decent interpretation, but unfortunately we do not.

# # Quick validity checks
# 
# Get number of unique companies.
# 
# Check how many of these companies ever reach a distressed state (should be 136 by Kaggle description).
# 
# Get a list of feature names.

# In[ ]:


total_n = len(companies.groupby('company')['company'].nunique())
print(total_n)

distress_companies = companies[companies['financial_distress'] < -0.5]
u_distress = distress_companies['company'].unique()
print(u_distress.shape)

feature_names = list(companies.columns.values)[3:] # ignore first 3: company, time, financial_distress
print(feature_names)


# # We know feature 80 is categorical...
# ...so let's pull it out as a list for use later.

# In[ ]:


f80 = list(companies.groupby('company')['x80'].agg('mean'))
f80 = [int(c) for c in f80]

# print(f80)
# print(len(f80))


# # Temporal cross validation: how to do it?
# 
# Let's follow the guidance set out in https://github.com/dssg/hitchhikers-guide/blob/master/curriculum/3_modeling_and_machine_learning/temporal-cross-validation/temporal_cross_validation_slides.pdf.
# 
# In order to pick a good date to separate train/test, we should ideally pick a date that allows most entities to appear in both the train and test data.
# 
# Unfortunately not all the companies live for the same amount of time, so if we pick a date that is too early or late, we may cut many of the companies out of the test set.
# 
# Let's generate a histogram of counts for each time period so we can pick a reasonable place to cut.

# In[ ]:


companies.hist(column=['time'], bins=14)


# We notice a bit of a decline, then uptick in the histogram around time period 10.
# 
# Declines imply that a company dies out of the dataset, so if we set our cut around t=10, we should still get a decent number of distress events in the training data.

# In[ ]:


# We can see from this that most companies start at time period 1, 
# but there are some which start their life much later.

# print(companies.groupby(['company'])['time'].agg('min'))


# # Does distress occur uniformly over time periods?

# In[ ]:


# What about the histogram of the timestamps when the distress event occurs?
distress_companies.hist(column=['time'], bins=14)


# Interesting...the frequency of distress definitely does not seem to be uniform across the time periods. 
# 
# That indicates that it may be bad science to obtain validation or test sets by simply picking out some companies, as we cannot assume that different companies are independent. The timestamp itself may be a useful signal (i.e. if a certain time period represents a macroeconomic state of decline for a certain industry, or the economy as a whole). Ok then, onto...

# # ...roll-forward cross validation
# 
# We'll now output a new set of features per training row: sum over each feature during time t, t-1, t-2...t-n. Note that this differs from the average by a constant, so while these features may represent something that shouldn't be summed (like average "Google maps rating" - I don't know), it'll just get normalized out later.
# 
# The training targets will be whether or not a distress event occured at the end of the period (t).

# In[ ]:


# Generate new train/val/test sets.

# Populate the entire pandas array into a dict for easier processing

datadict = {}
distress_dict = {}

for i in range (1, total_n+1):
    datadict[i] = {}
    distress_dict[i] = {}

print("Populating dictionary...")
for idx, row in companies.iterrows():
    company = row['company']
    time = int(row['time'])
    
    datadict[company][time] = {}
    
    if row['financial_distress'] < -0.5:
        distress_dict[company][time] = 1
    else:
        distress_dict[company][time] = 0
        
    for feat_idx, column in enumerate(row[3:]):
        feat = feature_names[feat_idx]
        datadict[company][time][feat] = column
        
# print('Dict population complete. Sample below:')
# print("\nData for company 1, time 1:")
# print(datadict[1][1])

# print("\nDistress history for company 1:")
# print(distress_dict[1])

print('We can encode categorical feature 80 as a one-hot vector with this many dimensions:')
print(len(list(set(f80))))

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(max(f80)))
f80_oh = label_binarizer.transform(f80)

# print(f80_oh[0:5])


# # Data generation

# In[ ]:


# Make new features as np array. We'll even add x80 back!

def rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods):

    for company in range(1, total_n+1):
            
            all_periods_exist = True
            for j in range(0, lookback_periods):
                if not time-j in distress_dict[company]:
                    all_periods_exist = False
            if not all_periods_exist:
                continue
            
            distress_at_eop = distress_dict[company][time]
            new_row = [company]

            for feature in feature_names:
                if feature == 'x80':
                    continue
                feat_sum = 0.0
                variance_arr = []
                for j in range(0, lookback_periods):
                    feat_sum += datadict[company][time-j][feature]
                    variance_arr.append(datadict[company][time-j][feature])
                new_row.append(feat_sum)
                new_row.append(np.var(variance_arr))
                
            for j in range(0,len(f80_oh[0])):
                new_row.append(f80_oh[company-1][j])

            if len(new_row) == ((len(feature_names)-1)*2 + 1 + len(f80_oh[0])) : # we have a complete row
                new_row.append(distress_at_eop)
                new_row_np = np.asarray(new_row)
                train_array.append(new_row_np)
    

def custom_timeseries_cv(datadict, distress_dict, feature_names, total_n, val_time, test_time, 
                         lookback_periods, total_periods=14):

    # Train data
    train_array = []
    for _t in range(1, val_time+1):
        time = (val_time+1) -_t # Start from time period 10 and work backwards
        train_array_np = rolling_operation(time, train_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    train_array_np = np.asarray(train_array)
    print(train_array_np.shape)
    # print(train_array_np[0])
    
    # Val data
    if val_time != test_time:
        val_array = []
        for time in range(val_time+1, test_time+1):
            val_array_np = rolling_operation(time, val_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

        val_array_np = np.asarray(val_array)
        print(val_array_np.shape)
        # print(val_array_np[0])
    else:
        val_array_np = None

    # Test data
    test_array = []
    # start from time period 11 and work forwards
    for time in range(test_time+1,total_periods+1):
        test_array_np = rolling_operation(time, test_array, datadict, distress_dict, feature_names, total_n,
                         lookback_periods)

    test_array_np = np.asarray(test_array)
    print(test_array_np.shape)
    # print(test_array_np[0])
    
    return train_array_np, val_array_np, test_array_np

# Generate our sets
train_array_np, val_array_np, test_array_np = custom_timeseries_cv(datadict, distress_dict, feature_names, total_n,
                                                     val_time=9, test_time=12, lookback_periods=3, total_periods=14)


# # Pull out last column as labels
# 
# 

# In[ ]:


X_train = train_array_np[:,0:train_array_np.shape[1]-1]
y_train = train_array_np[:,-1].astype(int)

X_val = val_array_np[:,0:val_array_np.shape[1]-1]
y_val = val_array_np[:,-1].astype(int)

X_test = test_array_np[:,0:test_array_np.shape[1]-1]
y_test = test_array_np[:,-1].astype(int)

np.set_printoptions(threshold=sys.maxsize)
print(X_train[0,:])
print(y_train)

print(X_val[0,:])
print(y_val)

print(X_test[0,:])
print(y_test)


# # Now try some models! Just the super basic, intro on Udacity stuff. :)

# In[ ]:


# Try a couple of different basic classification models

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def model_trial(model_type, hyperparam):
    if model_type in ['logistic-regression']:
        # Logistic Regression. Try 11, l2 penalty, understand one-vs-rest vs multinomial (cross-entropy) 
        model = LogisticRegression(penalty=hyperparam, solver='saga', max_iter=4000)
    elif model_type in ['decision-tree']:
        model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None)
    elif model_type in ['random-forest']:
        model = RandomForestClassifier(n_estimators=hyperparam)
    else:
        print("Warning: model {} not recognized.".format(model_type))
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print("Mean acc: %f" % model.score(X_val, y_val))
    print("F1: %f" % f1)
    print("Recall: %f" % recall)


# In[ ]:


print("-"*20 + "Logistic regression, l1:" + "-"*20)
model_trial('logistic-regression', 'l1')

print("-"*20 + "Logistic regression, l2:" + "-"*20)
model_trial('logistic-regression', 'l2')

print("-"*20 + "Decision tree:" + "-"*20)
model_trial('decision-tree', None)

for i in [2, 4, 10, 50, 100, 1000]:
    print("-"*20 + "Random forest, {} estimators:".format(i) + "-"*20)
    model_trial('random-forest', i)

