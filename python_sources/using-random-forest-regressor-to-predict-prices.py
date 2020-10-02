#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This noteboook is the fruit of our final project in [SI 370](https://www.si.umich.edu/programs/courses/370) class (*Undergrad Data Exploration* taught at the University of Michigan School of Information)
# 
# Reference Notebooks that we used to get started: 
# * https://www.kaggle.com/dfitzgerald3/randomforestregressor
# * https://www.kaggle.com/goldens/house-prices-on-the-top-with-a-simple-model
# * https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing

# # Data Preprocessing

# ## Steps to take
# 1. Import modules 
# 2. Read in data 
# 3. Remove skew by taking log
# 4. Cleaning data by dropping columns with more than 10% missing values
# 5. Impute missing values for continuous variables
# 6. Impute missing values for categorical variables
# 7. Combine categorical and continuous datasets after imputation

# **Step 1: Importing**

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import sklearn.ensemble as skens


# **Step 2: Read In Data**

# In[ ]:


# #read in files
# train = files.upload()
# test = files.upload()

# Store data in a Pandas Dataframe
# df_train_csv = pd.read_csv(io.BytesIO(train['train.csv']))
# df_test_csv = pd.read_csv(io.BytesIO(test['test.csv']))

df_train_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

#drop ID column
df_train = df_train_csv.drop(['Id'], axis=1)
df_test =  df_test_csv.drop(['Id'], axis=1)


# **Step 3: Remove Skew**

# In[ ]:


#Check the distribution of salesprice
sns.distplot(df_train['SalePrice']).set_title('Sale Price Distribution Training Dataset');


# In[ ]:


#Remove the skew of salesprice by taking the log of the salesprice
df_train['logSalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea']) # temp, remove

target = df_train['SalePrice']
target_log = df_train['logSalePrice']
sns.distplot(df_train['logSalePrice']).set_title('Log Sale Price Distribution Training Dataset');


# **Step 4: Cleaning data**

# In[ ]:


#combine train and test data
df = df_train_csv.append(df_test_csv, ignore_index = True, sort=False)


# In[ ]:


#check the shape of dataframe
df.shape


# In[ ]:


#drop entire column if missing more than 10 percent of values
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
dropped_columns = []
for row in missing_value_df.values:
    if row[1] > 10 and row[0] != 'SalePrice':
        dropped_columns.append(row[0])
# dropped_columns.delete('SalePrice')
df = df.drop(dropped_columns,axis=1)


# In[ ]:


#checking number of columns after dropping columns with many NaN values
df.shape


# In[ ]:


#checking for NaN values in remaining columns
na = df.isnull().sum()
na[na>0]


# In[ ]:


# Converting the numerical columns that should be categorical into type objects
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)


# In[ ]:


df.dtypes.to_frame().reset_index()[0].value_counts()


# In[ ]:


#separate into continuous and categorical columns
continuous_cols = []
for col in df.columns.values:
    if df[col].dtype != 'object':
        continuous_cols.append(col)
continuous_cols
df_cont = df[continuous_cols]
df_cat = df.drop(continuous_cols,axis=1)


# **Step 5: Preprocessing for Continuous Variables**

# In[ ]:


#impute missing values with median 
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_cont = imp_mean.fit_transform(df_cont)
#convert output into df
newlst=[]
for array in imputed_cont:
  newlst.append(array)
imputed_df= pd.DataFrame(newlst)
#add column names
labels = df_cont.columns.tolist()
imputed_df.columns = labels
df_cont = imputed_df


# **Step 6: Preprocessing for Categorical Variables**
# 
# 

# In[ ]:


#impute missing values with most frequent value
from sklearn.impute import SimpleImputer
df_cat.fillna(0, inplace=True)
imp_mean = SimpleImputer(missing_values=0, strategy='most_frequent')
imp_mean.fit(df_cat)
imputed_cat = imp_mean.transform(df_cat)
#'get column labels
lst = []
for array in imputed_cat:
    lst.append(array)
imputed_df = pd.DataFrame(lst)
labels = df_cat.columns.tolist()
imputed_df.columns = labels
# Make a copy without dummies for viz after random forest
df_cat_original = df_cat.copy()


# In[ ]:


#manual hot encoding
dummies = pd.get_dummies(imputed_df)
df_cat = dummies


# **Step 7: Combine Categorical and Continuous Datasets**

# In[ ]:


# joining the two dataframes
df_cleaned = df_cont.join(df_cat)


# In[ ]:


# adding the log prices to the training data frame
df_train = df_cleaned.iloc[:len(df_train)]
df_train = df_train.join(target_log)


# In[ ]:


# splitting the test and train data
df_test = df_cleaned.iloc[len(df_train):]

X_train = df_train.drop(['SalePrice','logSalePrice'], axis=1)
y_train = df_train['logSalePrice']

X_test = df_test.drop(['SalePrice'], axis=1)


# # Build a Random Forest
# 

# ## Steps to take
# 1. Split the training data into the _test and _train
# 2. train the Random Forest on the _test
# 3. measure the accuracy
# 4. Tune the huperparameters
# 5. Train the Random Forest on the original Train set with selected hyper parameters
# 6. Look at the feature importance
# 7. Predict the prices based on the original Test dataset

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = X_train
y = y_train
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


def train_model(params, X_train, y_train):
    rf_model = RandomForestRegressor(**params)
    rf_model.fit(X_train, y_train)
    return rf_model


# In[ ]:


def viz_results(model, X_test, y_test = None, score = None):
    score = round(model.score(X_test, y_test)*100,2) if score else "unknown"
    predicted_labels = model.predict(X_test)
    unlog_lables = np.exp(predicted_labels)
    title = f'Predicted Sale Price Distribution (score: {score} %)'
    sns.distplot(unlog_lables).set_title(title);


# In[ ]:


def viz_feature_importance(model, df_train, n_features):
    feat_importance = model.feature_importances_
    df_feat_importance = pd.DataFrame({'Feature Importance':feat_importance},
                index=df_train.columns[:len(df_train.columns)])

    df_feat_imp_sorted = df_feat_importance.sort_values(by='Feature Importance', ascending=False)
    df_feat_imp_sorted.iloc[:n_features].plot(kind='barh')


# In[ ]:


rf_model = train_model({'n_estimators': 5, 'max_depth': 10}, X_train_t, y_train_t)
viz_results(rf_model, X_test_t, y_test_t, score = True)


# In[ ]:


viz_feature_importance(rf_model, X, 10)


# ### Hyperparameters tweaking

# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# grid of possible hyperparameters
param_grid = {
              'n_estimators': [200, 1200, 1500, 2000],
              'bootstrap': [False],  
              'max_features': [70],
              'max_depth': [15, 30, 40, 50, 100], 
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 5]
             }
rf = RandomForestRegressor()

# We chose two options for finding the best hyperparameters: Grid Search and Random Search
# running the Grid Search
# clf_search = GridSearchCV(rf, 
#                                      param_grid, 
#                                      cv=3, 
#                                      verbose=10, 
#                                      n_jobs = -1,
#                                      )
# clf_search.fit(X_train_t, y_train_t)
# best_params_search = clf_search.best_params_

# TODO(vanya): make the params continious, np.instead of a list
# running the Randomized Search
clf_rand = RandomizedSearchCV(estimator = rf, 
                               param_distributions = param_grid, 
                               n_iter = 5, 
                               cv = 3, 
                               verbose=10, 
                               random_state=42, 
                               n_jobs = -1)
clf_rand.fit(X_train_t, y_train_t)
clf_rand.best_estimator_
best_params_rand = clf_rand.best_params_

# print(clf_search.best_estimator_)
# print(best_params_search)
# print(best_params_rand)


# Traininig the model with the found 'best' parameters

# In[ ]:


best_params = best_params_rand
best_params


# In[ ]:


rf_model = train_model(best_params, X_train_t, y_train_t)
rf_model.score(X_test_t, y_test_t)
viz_results(rf_model,X_test_t, y_test_t, score=True)


# In[ ]:


# Perform K-Fold CV
scores = cross_val_score(rf_model, X, y, cv=5)
print(f'Accuracy: {scores.mean():.2f} (+/- {scores.std()*2:.2f})')


# Running the predictions on the 'test.csv' that has gone through data preprocessing

# In[ ]:


rf_model = train_model(best_params, X_train, y_train)
viz_results(rf_model,X_test)

predicted_labels = rf_model.predict(X_test)
unlog_lables = np.exp(predicted_labels)


# In[ ]:


viz_feature_importance(rf_model, X_test, 10)


# Preparing the predicted value for submition on Kaggle

# In[ ]:


unlog_lables = pd.Series(unlog_lables)
submission = df_test_csv['Id']
submission = submission.to_frame()
submission['SalePrice'] = unlog_lables
submission.to_csv('submission.csv', index=False)


# # Accounting for the categorical variables
# 

# ## Steps to take
# 1. Make price categorical: high-price houses, low-price, and middle range
# 2. Make paired (high and low prices) stacked bar graphs for each categorical variable to see if the results are uniform
# 3. Run chi-square test to see if the difference is statistically significant
# 4. What categorical variables have the smallest p-value
# 5. Interpret the result
# 

# In[ ]:


# Cut prices into 4 buckets by quartile (low, middle 1, middle 2, high)
df_cat_original['Price_Cat'] = pd.qcut(df_train['SalePrice'], 4, labels=["low", "middle1", "middle2", "high"])

# Drop houses in the middle price range
df_cat_cleaned = df_cat_original[(df_cat_original['Price_Cat'] =='low') | (df_cat_original['Price_Cat'] =='high' ) ]

# Drop any categorical variables with only one category
# Variables with only one category would be a meaningless predictor
nunique = df_cat_cleaned.nunique()
df_cat_cleaned = df_cat_cleaned.drop( columns = nunique[nunique<=1].index.values )


# In[ ]:


df_cat_cleaned.head()


# In[ ]:


from matplotlib import rc
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency


# In[ ]:


# Draw a mosaic plot and return chi-square test result (p-value) for a given categorical variable
def association (col):
  # Draw Mosaic plot
  # fig, ax = plt.subplots(figsize = (15, 10))
  # props = lambda key: {'color': 'orange' if 'low' in key else 'lightblue'}
  # labels = lambda key: '\n'.join(key) if ('low' in key) or ('high' in key) else ''

  # m = mosaic(df_cat_cleaned, [col, 'Price_Cat'], title = "{0} v.s. Price".format(col),\
  #           ax = ax, axes_label = False, horizontal = True, \
  #           properties = props, labelizer=labels)
  # fig.show()
  
  # Make Crosstab and return p-value from the chi-square test
  ct = pd.crosstab(df_cat_cleaned[col], df_cat_cleaned['Price_Cat'], margins = True)
  chi2, p, dof, ex = chi2_contingency(ct)
  print(col + ' v.s. Price'+ '     '+"p-value: "+"{0:.4f}".format(round(p,4)))
  return p


# In[ ]:


# Loop through all the columns except for price_cat using the association function defined earlier
# Store the pvalues into dictionary called pvalue. Key: Column name, Value: p-value
pvalues = {}
for col in df_cat_cleaned.columns.values[:-1]:
  pvalues[col] = association (col)


# In[ ]:



# Return the categorical values with the smallest p-value
min(pvalues,key=pvalues.get)


# In[ ]:


# Extract the categorical variables with p-value greater than 0.05
# P-value that's greater than 0.05 are insignificant

{ key:value for (key,value) in pvalues.items() if value > 0.05}


# **Future Work that we are considering to improve the accuracy:**
# * Working on further preparing the data before running the model
# * Running a random search on hyperparameters from a large range instead of a list of descrete values
# * Trying different models to predict the sell price[[](http://)](http://)

# In[ ]:




