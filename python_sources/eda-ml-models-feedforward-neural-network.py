#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msm
import pandas as pd
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras


# # All the remarks and comment are welcome. Thank you....

# *`Problem statement`*: Predicting the sale price of bulldozers sold at auctions.
# 
# *`Data`*: the data for this competition is split into three parts, which are trainset validationset and testset.
# 
# *`The key fields are in train.csv are`*:
# - `SalesID`: the unique identifier of the sale
# - `MachineID`: the unique identifier of a machine.  A machine can be sold multiple times
# - `saleprice`: what the machine sold for at auction (only provided in train.csv)
# - `saledate`: the date of the sale.
# It is important to underline that the 

# First of all let's create a function to unzip the trainset.

# In[ ]:


data = pd.read_csv('../input/trainset/Train.csv', low_memory=False, parse_dates=['saledate'])
data.head()


# In[ ]:


data.describe()


# #EDA

# In[ ]:


df_eda = data.copy()


# 1. Dataset shape (rows and columns): (401125, 53)
# 2. Target attribute name: SalePrices
# 3. Data types of features:  2 x float64, 6 x int64(6), 45 x object
# 4. Understand the problem:  We'll look at each variable and do an analysis about their meaning and importance for this problem.
# 5. Analysis of different features
#   - Missing values analysis
#   - Categorical feature
#   - Numerical feature (discret and continue feature)
#   - Univariable study: We'll focus on the dependent variable ('SalePrice') and try to know a little bit more about it.
#   - Multivariate study. We'll try to understand how the dependent variable and independent variables relate.
# 8. Data cleaning: We'll clean the dataset and handle the missing data, outliers and categorical variables.

# After looking at each variable and try to understand their meaning and relevance to this problem ( we have selected based on our understanding about the meaning then we have ploted those feature versus SalePrice). The result has shown Drive_System, Engine_Horsepower can play an important role in this problem. However, none of the above mentioned key fields has strong relationship with salePrice.

# In[ ]:


plt.figure(figsize=(13, 5))
i = 0
for attribute in ('Engine_Horsepower', 'Drive_System'):
  i += 1
  plt.subplot(1, 2, i)
  sns.boxplot(df_eda[attribute], df_eda['SalePrice'])
plt.show()


# Engine Horsepower, Drive System seem to be related with SalePrice. The relationship seems to be based on  different category. The box plot shows how sales prices increase with the different category.
# It is important to understand we just analysed four variables. However, there are many other that we should analyse. 

# Univarible analysis:
# - Independant variable analysis

# In[ ]:


# Missing value
msm.bar(df_eda, figsize=(24, 5), fontsize=16, labels=True, log=False )
plt.show()


# In[ ]:


(df_eda.isnull().sum()/len(df_eda)).sort_values()


# The missing value varie between 0.08% to 94% depending on feature. 
# Based on that information, it is important to underline that all the features that contain more than 90% can be dropped during data processing (One can argues that they cannot help the ML model to generalize its prediction).However, we will keep them.

# In[ ]:


# Let's identify the which feature contains more than 90% missing value
heigh_nan_features = [feature for feature in df_eda.columns  if (df_eda[feature].isna().sum()/len(df_eda)) > 0.9]
heigh_nan_features


# In[ ]:


# Categorical features
categorical_feature_list = [feature for feature in df_eda.columns if feature != 'saledate' and df_eda[feature].dtype == 'O']

for feature in categorical_feature_list:
  print(f'{feature :-<50} {df_eda[feature].nunique()}')


# The categories per feature varie between 2 and 4999. Disaggregation of fiModelDesc have the highest number of categories

# In[ ]:


# Numerical categories analysis
numerical_feature_list = [feature for feature in df_eda.columns if feature not in ('SalePrice', 'saledate') and df_eda[feature].dtype != 'O']


# In[ ]:


discret_value_feature = [feature for feature in numerical_feature_list if len(df_eda[feature].unique()) < 25]
sns.countplot(x='datasource', data=df_eda[['datasource']]);


# In[ ]:


continuous_value_feature = [feature for feature in numerical_feature_list if df_eda[feature].nunique() > 25]
def plot_hist (nrow=1, ncol=1, feature_list=None, figsize=(24, 10)):
  plt.figure(figsize=figsize)
  i = 0
  for feature in feature_list:
    i += 1
    plt.subplot(nrow, ncol, i)
    sns.distplot(df_eda[feature], bins=50)
  plt.show()


# In[ ]:


plot_hist(nrow=2, ncol=3, feature_list=continuous_value_feature)


# All the independant features that contain continuouse value are skewed. Therefore,one can use log transformation to reshape the data distribution to normal one.

# ***Now let's analysis dependant variable (target feature)***

# In[ ]:


#2 SalePrice
df_eda['SalePrice'].describe()


# In[ ]:


sns.distplot(df_eda['SalePrice'], bins=50);


#  The SalePrice deviate from the normal distribution. It is positively skewed. Hence, this would mean that many  Bulldozers were being sold for less than the average value (31099.712848).

# In[ ]:


df_eda['SalePrice'].skew(), df_eda['SalePrice'].kurt()


# In[ ]:


sns.boxplot(df_eda['SalePrice']);


# Boxplot has shown that salePrice variable contains some outliers.

# ***Multivariate study*
# We are going to investigate how target variable (SalePrice) and independent variables are related.***

# In[ ]:


# Descret value features
sns.boxplot(x='datasource', y='SalePrice', data=df_eda);


# In[ ]:


# continuous value feature
def scatter_plot (nrow=1, ncol=1, feature_list=None, figsize=(24, 10), target='SalePrice'):
  plt.figure(figsize=figsize)
  i = 0
  for feature in feature_list:
    i += 1
    plt.subplot(nrow, ncol, i)
    sns.scatterplot(df_eda[feature], df_eda[target])
  plt.show()


# In[ ]:


scatter_plot(nrow=2, ncol=3, feature_list=continuous_value_feature)


# It is quiet hard to quantify the relation between SalePrice and numerical feature values. However, we are going one more approach which allows us to determine the corelation coefficient between SalePrice and different independant feature.

# In[ ]:


corr = df_eda.corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, cmap='YlGn_r', cbar=False, annot=True);


# In[ ]:


# Let's plot what we have seen in heatmap using pairplot
sns.pairplot(df_eda[numerical_feature_list + ['SalePrice']], height = 2.5);


# Although we already know some of the figures, this scatter plot gives us a reasonable idea about relationships between variables.
# The correlation between dependandt and independant features are very close to zero as one can see on heatmap plot. Also, the above mega scatter plot as alo shwon that.

# ***Outliers***
# As already mentioned the feature SalePrice contains outliers, which can affect our models. But it can be at the same time a valuable source of information about specific behaviours. We will do a quick analysis through the standard deviation.
# 
# We will first establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. Moreover, data standardization means converting data values to have mean of 0 and a standard deviation of 1.

# In[ ]:


saleprice_scaled = StandardScaler().fit_transform(df_eda['SalePrice'].to_numpy().reshape(-1, 1));
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


# Low range values are similar and not too far from 0. However, high range values are far from 0.
# Note that for now, we'll not consider any of these values as an outlier but we should be pay more attention on those high rnge values.

# ***Statistical bases for multivariate analysis***
# -  We have already done some data analysis and discovered a lot about dependant variable('SalePrice'). Now it's time to go even deeper and understand how 'SalePrice' complies with the statistical assumptions that enables us to apply multivariate techniques.
#  

# - Histogram - Kurtosis and skewness.
# - Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

# In[ ]:


def statical_analysis (df, attribute_name, figsize=(15, 5)):

  plt.figure(figsize=figsize)
  i = 0
  for item in (1, 2):
    i +=1
    plt.subplot(1, 2, i)
    if i == 1: sns.distplot(df[attribute_name], bins=50, fit=stats.norm)
    else:  stats.probplot(df[attribute_name], plot=plt)
    
  plt.show()


# In[ ]:


# For sale price 
statical_analysis(df_eda, 'SalePrice')


# 'SalePrice' is not normally distributed. It depicts 'peakedness', positive skewness and does not follow the diagonal line, see probability plot. Hence, we will need a log transformation to solve the problem. 
# It is import to underline that apart of SalePrice the majority of numerical variable in dataset are just Id. Therefore, will not apply log transformation to those attributes.

# In[ ]:


# let's apply log transform to SalePrice
df_log = df_eda.copy()
df_log['SalePrice'] = np.log1p(df_log['SalePrice'])
statical_analysis(df_log, 'SalePrice')


# In[ ]:


numerical_feature_list


# In[ ]:


statical_analysis(df_eda, 'MachineHoursCurrentMeter')


# In[ ]:


#apply log transform
df_log['MachineHoursCurrentMeter'] = np.log1p(df_log['MachineHoursCurrentMeter'])
statical_analysis(df_log, 'MachineHoursCurrentMeter')


# Feature ingineering

# In[ ]:


df = data.copy()


# In[ ]:


# let's first work on sale date 
def date_preprocessing (dataFrame, feature='saledate'):
  dataFrame['saleYear'] = dataFrame[feature].dt.year
  dataFrame['saleMonth'] = dataFrame[feature].dt.month
  dataFrame['saleDay'] = dataFrame[feature].dt.day
  dataFrame['saleDayOfWeek'] = dataFrame[feature].dt.dayofweek
  dataFrame['saleDayOfYear'] = dataFrame[feature].dt.dayofyear
  dataFrame.drop(feature, inplace=True, axis=1)


# In[ ]:


date_preprocessing(df)


# In[ ]:


def Ordinal_encoder (dataFrame, feature_list):
  mask = {'Mini': 1, 'Small': 1, 'Medium': 2, 'Large / Medium': 3,  'Large': 3, 'Low': 1, 'High': 3}
  for label in feature_list:
    dataFrame[label] = dataFrame[label].map(mask)


# In[ ]:


Ordinal_encoder(df, ['ProductSize', 'UsageBand'])


# In[ ]:


# Now let's handle Missing value
class FillMissing ():

  def fill_categorical (self,  dataFrame):
    for label, content in  dataFrame.items():
      if pd.api.types.is_string_dtype(content):
         dataFrame[label].fillna('missing', inplace=True)
    # return df

  def fill_numerical (self,  dataFrame):
    for label, content in  dataFrame.items():
     if pd.api.types.is_numeric_dtype(content):
        dataFrame[label] = content.fillna(content.median())
    # return df


# In[ ]:


fill_missing_value = FillMissing()


# In[ ]:


fill_missing_value.fill_categorical(df)


# In[ ]:


fill_missing_value.fill_numerical(df)


# In[ ]:


def nominal_encoder (dataFrame, label_list):

  for label, content in  dataFrame.items():
    if pd.api.types.is_string_dtype(content):
       dataFrame[label] = content.astype('category').cat.as_ordered()
       dataFrame[label] = pd.Categorical(content).codes + 3


# In[ ]:


label_list = [feature for feature in df.columns if df[feature].dtype == 'O']
nominal_encoder(df, label_list)


# In[ ]:


#drop IDs 
df =df.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)


# # Feature selection
# - SelectKBest will be used along decision tree during feature selection process. It is important to underline that we have choose decision because it doesn't require a lot data preparation and it is also quiet fast.
# - Also we will split trainset. Hence, this will allow us to check if our model is not overfitting.

# In[ ]:


# We will be using decision to simply findout which value of k that gives small MSE.
feature_selection = make_pipeline(SelectKBest(score_func=f_regression, k=52), DecisionTreeRegressor(max_depth=10, random_state=0))


# In[ ]:


# For the purpose will split the training set into train and test. It is important to bear in mind that the mentioned splited dataset will be use only for finding best K value
X_1, y_1 = df.drop('SalePrice', axis=1), df['SalePrice']
X_1, X_2, y_1, y_2 = train_test_split(X_1, y_1, test_size=.2)
feature_selection.fit(X_1, y_1)


# In[ ]:


y_1_pred, y_2_pred= feature_selection.predict(X_1),  feature_selection.predict(X_2)


# In[ ]:


print(f'mse_1: {np.sqrt(mean_squared_error(y_1, y_1_pred))}, mse_2:{np.sqrt(mean_squared_error(y_2, y_2_pred))}')


# We will keep all feature because they have provided the smallest MSE

# # Feature scaling

# In[ ]:


def scale_feature (X):
  scaled = StandardScaler().fit_transform(X)
  return scaled


# In[ ]:


X_train, y_train = df.drop('SalePrice', axis=1), df['SalePrice']
X_train = scale_feature(X_train)


# # Let's train our models (model experimentation)

# In[ ]:


def model_training (models, X_train, y_train, cv=3):
  for name, model in models.items():
    print(name)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    print({'score': np.sqrt(-scores).mean(), 'std': np.sqrt(-scores).std() })


# In[ ]:


models = {'KNeighborsRegressor': KNeighborsRegressor(),
          'RandomForestRegressor': RandomForestRegressor(), 
          'ExtraTreesRegressor': ExtraTreesRegressor(),  
          'AdaBoostRegressor': AdaBoostRegressor(), 
          'GradientBoostingRegressor':  GradientBoostingRegressor(), 
          'XGBRegressor':  XGBRegressor()
          }

model_training(models, X_train, y_train)


# # Hyper parameters tuning
# We will proceed hyperparameters tune for ExtraTreesRegressor since it provides the smallest MSE value

# In[ ]:


def hyperparameters_tuning (model, X_train, y_train, param_grid, cv=5):
  grid = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=2, cv=cv)
  grid.fit(X_train, y_train)
  
  return grid.best_estimator_


# In[ ]:


extra_forest_param_grid = {'n_estimators':[150, 300, 600], 
                          'min_samples_leaf':[2, 3],  
                          'min_samples_split':[14, 12], 
                          'max_features':[0.7, 1],
                          'max_samples':[10000], # 10000 has been used due to the limited available memory of notebook (in Kaggle).
                          'bootstrap':[None, True]
              }
extra_forest = hyperparameters_tuning(ExtraTreesRegressor(), X_train, y_train, extra_forest_param_grid)
extra_forest


# # Evaluation of best estimator (model)

# In[ ]:


# let's prepare validation data 
X_valid = pd.read_csv('../input/bluebook-for-bulldozers/Valid.csv', parse_dates=['saledate'])
y_valid = pd.read_csv('../input/bluebook-for-bulldozers/ValidSolution.csv')
y_valid = y_valid['SalePrice']


# In[ ]:


#drop IDs 
X_valid =X_valid.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)


# In[ ]:


date_preprocessing(X_valid)


# In[ ]:


Ordinal_encoder(X_valid, ['ProductSize', 'UsageBand'])


# In[ ]:


fill_missing_value.fill_categorical(X_valid)


# In[ ]:


fill_missing_value.fill_numerical(X_valid)


# In[ ]:


nominal_encoder(X_valid, label_list)


# In[ ]:


X_valid = scale_feature(X_valid)


# In[ ]:


def evaluate_best_model (X_train, y_train, X_valid, y_valid):

    estimator = ExtraTreesRegressor(bootstrap=None, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features=0.7, max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=12, min_weight_fraction_leaf=0.0,
                    n_estimators=300, n_jobs=None, oob_score=False,
                    random_state=None, verbose=0, warm_start=False)
    
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_valid_pred = estimator.predict(X_valid)

    return {
          'MSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
          'MSLE_train':np.sqrt( mean_squared_log_error(y_train, y_train_pred)),
          'MSE_valid': np.sqrt(mean_squared_error(y_valid, y_valid_pred)),
          'MSLE_valid': np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
      }


# In[ ]:


evaluate_best_model(X_train, y_train, X_valid, y_valid)


# The best estimator seems to overfit our training data. One can perform more hyper parameters tuning in order to solve the overfitting issue (or intance set max_depth to a value or check more possible value for min_samples_leaf, n_estimators value etc ... ). However, I will leave as it is for this moment.
# Next, I will build feedforward neural netwok and train and evaluate. 

# # FeedForward Neural Network

# In[ ]:


FNN = keras.models.Sequential([
                               Flatten(input_shape=[52]),
                               Dense(350, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.3),
                               Dense(250, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.3),
                               Dense(150, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.4),
                               Dense(100, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.5),
                               Dense(70, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.5),
                               Dense(1)
])


# In[ ]:


FNN.compile(optimizer='adam', loss='mse', metrics=['mse'])


# In[ ]:


history = FNN.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=10, callbacks=[keras.callbacks.EarlyStopping(patience=10)])


# In[ ]:


plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend();


# In[ ]:


def evaluate_feedforwardNN (X_train, y_train, X_valid, y_valid):
    y_train_pred = FNN.predict(X_train)
    y_valid_pred = FNN.predict(X_valid)

    
    return {
          'MSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
          'MSLE_train':np.sqrt( mean_squared_log_error(y_train, y_train_pred)),
          'MSE_valid': np.sqrt(mean_squared_error(y_valid, y_valid_pred)),
          'MSLE_valid': np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
      }


# In[ ]:


evaluate_feedforwardNN (X_train, y_train, X_valid, y_valid)


# # Predictions

# In[ ]:


# Let's prepare the test data for prediction
testset =  pd.read_csv('../input/bluebook-for-bulldozers/Test.csv', parse_dates=['saledate'])
#drop IDs 
X_test =testset.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)


# In[ ]:


date_preprocessing(X_test)


# In[ ]:


Ordinal_encoder(X_test, ['ProductSize', 'UsageBand'])


# In[ ]:


fill_missing_value.fill_categorical(X_test)


# In[ ]:


fill_missing_value.fill_numerical(X_test)


# In[ ]:


nominal_encoder(X_test, label_list)


# In[ ]:


X_test = scale_feature(X_test)


# In[ ]:


X_test_predict = FNN.predict(X_test)


# In[ ]:


prediction = pd.DataFrame()
prediction['SalesID'] = testset['SalesID']
prediction['SalePrice']= X_test_predict


# In[ ]:


prediction


# In[ ]:




