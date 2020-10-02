#!/usr/bin/env python
# coding: utf-8

# ## SUMMARY OF THE APPROACH - 
# 
# ## EDA
# 
# **NAN Handling**
# * The train data provided had a lot of NANs in them.
# * The columns with more than 3/4 th of the total values (50000) were dropped from the analysis.
# * Different techniques like filling NANs with mean, median, mode, values attained by the model were tried and best choices were made depending on the correlation values with wave_height.
# * The NANs in some of the categorical variables were converted into a new class and then processed.
# 
# **Transformations**
# * Some of the continuous features were skewed, i.e, they were Left-tailed or Right-tailed.
# * Appropriate transformations like BoxCox Transformation and Square Transformation were used.
# * The dependent variable "wave_height" was also right-tailed.
# * Cube-Root Transformation was used for standardizing it.
# * Predictions were made for the cube-root of the wave heights, which were later raised to the power 3 to get the actual values of the predictions.
# 
# ## Feature Selection
# 
# * Techniques like L1 regularization, L2 regularization, Recursive Feature Elimination were used for optimal feature selection.
# * Using RFECV with Gradient Boosting Regressor for Feature Selection gave the most optimal Cross-Validation results. So, this was used finally for the prediction on the test set.
# 
# ## Parameter Tuning
# 
# * Bayesian Hyper-parameter tuning was used to find the optimal paramters for the learning algorithm.
# 
# ## Fitting and Prediction
# 
# * The models tried out were - 
#         1.Elastic Net Regression
#         2.Lasso Regression
#         3.Ridge Regression
#         4.Random Forest Regression
#         5.Support Vector Regression
#         6.Gradient Boosting Regression
#         7.XGBoost Regressor
#         8.LightGBM Regressor
# 
# * The model giving the best result was Gradient Boosting Regressor, which was finally used for the prediction on the test set.
# 
# ## Scope of Improvement
# 
# **Further things which I could not try due to time as well as submission constraints - **
# 
# * Ensembling the different well performing models to get an even better prediction
# * Meta-Models and Stacking for better predictions.
# 
# 

# ## Moving on to the code for implementing the above mentioned ideas....

# ### Importing packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.tabular import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from scipy.stats import norm, skew
from sklearn.ensemble import IsolationForest
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Loading dataset

# In[ ]:


train = pd.read_csv('../input/train__updated.csv')
test = pd.read_csv('../input/test__updated.csv')


# In[ ]:


train.describe()


# ## Droping Index column

# In[ ]:


train.drop('index',axis=1,inplace=True)
idx=test['index']


# In[ ]:


test.drop('index',axis=1,inplace=True)


# ## Data PreProcessing

# **Making the categorical variables in the form of integers into categorical variables and making the NAN values as a New Category. **

# In[ ]:


train['platform_type'] = np.where(train['platform_type']==5.0,0,1)
train['sst_measurement_method'] = train['sst_measurement_method'].fillna(value=2.0)
train['sst_measurement_method'] = train['sst_measurement_method'].apply(str)
train['nightday_flag'] = np.where(train['nightday_flag']==1,0,1)
train['national_source_indicator'] = train['national_source_indicator'].fillna(value=0)
train['id_indicator'] = train['id_indicator'].apply(str)
train['wind_speed_indicator'] = train['wind_speed_indicator'].apply(str)
train['deck'] = train['deck'].apply(str)
train['source_id'] = np.where(train['source_id']==103,0,1)
train['ship_course'] = train['ship_course'].fillna(value=9.0)
train['ship_course'] = train['ship_course'].apply(str)
train['ten_degree_box_number'] = train['ten_degree_box_number'].apply(str)
train['characteristic_of_ppp'] = train['characteristic_of_ppp'].fillna(value=9)
train['characteristic_of_ppp'] = train['characteristic_of_ppp'].apply(str)

test['platform_type'] = np.where(test['platform_type']==5.0,0,1)
test['sst_measurement_method'] = test['sst_measurement_method'].fillna(value=2.0)
test['sst_measurement_method'] = test['sst_measurement_method'].apply(str)
test['nightday_flag'] = np.where(test['nightday_flag']==1,0,1)
test['national_source_indicator'] = test['national_source_indicator'].fillna(value=0)
test['id_indicator'] = test['id_indicator'].apply(str)
test['wind_speed_indicator'] = test['wind_speed_indicator'].apply(str)
test['deck'] = test['deck'].apply(str)
test['source_id'] = np.where(test['source_id']==103,0,1)
test['ship_course'] = test['ship_course'].fillna(value=9.0)
test['ship_course'] = test['ship_course'].apply(str)
test['ten_degree_box_number'] = test['ten_degree_box_number'].apply(str)
test['characteristic_of_ppp'] = test['characteristic_of_ppp'].fillna(value=9)
test['characteristic_of_ppp'] = test['characteristic_of_ppp'].apply(str)


# **Managing NAN values for the Continuous variables depending on how the correlation value varies for different imputing methods like Mean, Median, Mode, a few common values that variable attains.**

# In[ ]:


NANwind_speed = train['wind_speed'].median()
NANsea_level_pressure = train['sea_level_pressure'].mode()[0]
NANamt_pressure_tend = train['amt_pressure_tend'].mode()[0]
NANair_temperature = train['air_temperature'].mode()[0]
NANsea_surface_temp = train['sea_surface_temp'].mode()[0]
NANswell_height = train['swell_height'].median()
NANwave_period = train['wave_period'].median()

train['wind_speed'] = train['wind_speed'].fillna(-1)
train['sea_level_pressure'] = train['sea_level_pressure'].fillna(NANsea_level_pressure)
train['amt_pressure_tend'] = train['amt_pressure_tend'].fillna(NANamt_pressure_tend)
train['air_temperature'] = train['air_temperature'].fillna(NANair_temperature)
train['sea_surface_temp'] = train['sea_surface_temp'].fillna(NANsea_surface_temp)
train['swell_height'] = train['swell_height'].fillna(5.0)
train['wave_period'] = train['wave_period'].fillna(0)

test['wind_speed'] = test['wind_speed'].fillna(-1)
test['sea_level_pressure'] = test['sea_level_pressure'].fillna(NANsea_level_pressure)
test['amt_pressure_tend'] = test['amt_pressure_tend'].fillna(NANamt_pressure_tend)
test['air_temperature'] = test['air_temperature'].fillna(NANair_temperature)
test['sea_surface_temp'] = test['sea_surface_temp'].fillna(NANsea_surface_temp)
test['swell_height'] = test['swell_height'].fillna(5.0)
test['wave_period'] = test['wave_period'].fillna(0)


# **There were many columns which had no information at all contained in them - **
# 
# **1.All the rows were having a single value for that categorical variable **
# 
# **2.More than 3/4 th of the values were missing (NANs)**
# 
# **3.Very Low correlation with the Dependent variable "wave_height"**

# In[ ]:


#train.value_counts()


# In[ ]:


train.drop(columns=['wbt_indicator','time_indicator','dup_check','indicator_for_temp','imma_version','attm_count','latlong_indicator','wind_direction_indicator','dpt_indicator','source_exclusion_flags','release_no_primary','release_no_secondary','release_no_tertiary','intermediate_reject_flag','release_status_indicator'],inplace=True)
test.drop(columns=['wbt_indicator','time_indicator','dup_check','indicator_for_temp','imma_version','attm_count','latlong_indicator','wind_direction_indicator','dpt_indicator','source_exclusion_flags','release_no_primary','release_no_secondary','release_no_tertiary','intermediate_reject_flag','release_status_indicator'],inplace=True)

train.drop(columns=['day','hour','national_source_indicator','year','ship_speed','visibility','present_weather','past_weather','wetbulb_temperature','dewpoint_temperature','total_cloud_amount','lower_cloud_amount','swell_direction','swell_period'],inplace=True)
test.drop(columns=['day','hour','national_source_indicator','year','ship_speed','visibility','present_weather','past_weather','wetbulb_temperature','dewpoint_temperature','total_cloud_amount','lower_cloud_amount','swell_direction','swell_period'],inplace=True)


# **Separating out columns for the continuous variables for carrying out operations later....**

# In[ ]:


categoricalCols = ['ten_degree_box_number','characteristic_of_ppp','sst_measurement_method','id_indicator','wind_speed_indicator','deck','ship_course']
asa = set(train.columns) - set(categoricalCols)
asb = asa - set(['platform_type','nightday_flag','source_id','dup_status'])
asb = list(asb)
asb = set(asb) - set(['wave_height'])
asb = list(asb)


# **One-Hot Encoding certain categorical variables which were not ordinal for better representation in the learning model**

# In[ ]:


train = pd.get_dummies(train,columns=['ten_degree_box_number','characteristic_of_ppp','sst_measurement_method','id_indicator','wind_speed_indicator','deck','ship_course'])
test = pd.get_dummies(test,columns=['ten_degree_box_number','characteristic_of_ppp','sst_measurement_method','id_indicator','wind_speed_indicator','deck','ship_course'])


# ## Outlier Removal

# Used **IsolationForest **for detecting the outliers in the given train set. 
# A threshold of 5% outliers was set and IsolationForest model was fit in, which gave indication about which rows were outliers based on individual feature analysis.
# 
# This individaul feature analysis was combined to form a **Tri-variate outlier detection**, which removed the suspected outliers from our train data.

# In[ ]:


trainer = train.copy()
fig, axs = plt.subplots(3,4 , figsize=(22, 12), facecolor='w', edgecolor='k')
axs = axs.ravel()
outlierPreds = np.ones((len(asb),len(train)))
for i, column in enumerate(asb):
    print (column)
    isolation_forest = IsolationForest(contamination=0.05,behaviour='old')
    isolation_forest.fit(train[column].values.reshape(-1,1))

    xx = np.linspace(train[column].min(), train[column].max(), len(train)).reshape(-1,1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)
    
    outlierPreds[i] = isolation_forest.predict(train[column].values.reshape(-1,1))
    
    axs[i].plot(xx, anomaly_score, label='anomaly score')
    axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                     where=outlier==-1, color='r', 
                     alpha=.4, label='outlier region')
    axs[i].legend()
    axs[i].set_title(column)


# In[ ]:


tempArr = outlierPreds.T

for i in range(0,len(tempArr)):
  count = 0
  for j in tempArr[i]:
    if j == -1:
      count = count+1
  if count >= 3:
    train.drop([i],inplace=True)


# **Viewing the continuous variables Histogram plots **

# In[ ]:


# for k in asb:
#     print(k)
#     plt.hist(train[k])
#     plt.show()


# It is quite evident that some of the continuous variables are **skewed**,i.e, some are Left-skewed and some are Right-skewed distributions in the train set.
# 
# We set a **threshold of 0.75** for skewness and fix this issue for the  skewed variables using -
# 
# 1.**BoxCox Transformations** for Right-tailed distributions
# 
# 2.**Square Transformations** for Left-tailed distributions

# In[ ]:


skewed_feats = train[asb].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness


# In[ ]:


skewnessBox = skewness[(skewness.Skew)>0.75]
skewnessSquare = skewness[(skewness.Skew)<-0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewnessBox.shape[0]))
print("There are {} skewed numerical features to Box Cox transform".format(skewnessSquare.shape[0]))


# In[ ]:


from scipy.special import boxcox1p
skewed_features1 = skewnessBox.index
skewed_features2 = skewnessSquare.index
lam = 0.15
for feat in skewed_features1:
    train[feat] = boxcox1p(train[feat], lam)
    test[feat] = boxcox1p(test[feat], lam)
for feat in skewed_features2:
    train[feat] = np.square(train[feat])
    test[feat] = np.square(test[feat])


# **Scaling the Continuous variables for facilitating better learning by the model.**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train[asb] = scaler.fit_transform(train[asb])
test[asb] = scaler.transform(test[asb])


# ## Preparing Data for feeding into the model

# Visualising the dependent variable "wave_height" and its distribution. 
# 
# It is quite evident that the **wave_height is right-tailed** and hence, different transformations to fix the skewness were evaluated. The **cube root transformation** was finally used as the graph is the most standard for that transformation.
# 
# Then the data was split into X and y for future training.

# In[ ]:


plt.subplot(1, 4, 1)
sns.distplot(train.wave_height, kde=False, fit = norm)

plt.subplot(1, 4, 2)
sns.distplot(np.log(train.wave_height + 1), kde=False, fit = norm)
plt.xlabel('Log WaveHeight')

plt.subplot(1, 4, 3)
sns.distplot(np.sqrt(train.wave_height), kde=False, fit = norm)
plt.xlabel('Square WaveHeight')


plt.subplot(1, 4, 4)
sns.distplot(np.cbrt(train.wave_height), kde=False, fit = norm)
plt.xlabel('Cube WaveHeight')


# It can be clearly seen that the **cube root transformation** gives back the **most standardized distribution** of the wave height.

# In[ ]:


y = train['wave_height']
yTrue = y.copy()
train['wave_height'] = np.cbrt(y)
y = train['wave_height']
train.drop(columns=['wave_height'],inplace=True)


# ## Feature Selection

# Using **RFECV** for the optimal feature selection process, using "**Negative of the Mean Squared Error**" as the metric to maximize, in turn minimizing the RMSE

# In[ ]:


from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[ ]:


gdb = GradientBoostingRegressor()
selector1 = RFECV(gdb, step=1, cv=5,n_jobs =-1,scoring="neg_mean_squared_error",verbose = 2)
start_time = timer(None)
selector1.fit(train, y)
timer(start_time)


# In[ ]:


selected1 = train.columns[selector1.support_]
print(selected1)


# This is the optimal feature selection which i got during training - 
# 
# **['month', 'latitude', 'longitude', 'wind_speed', 'sea_level_pressure',
#         'air_temperature', 'sea_surface_temp', 'wave_period', 'swell_height',
#        'one_degree_box_number', 'id_indicator_1']**

# In[ ]:


trainSelected1 = train[selected1]


# ## Hyper-parameter Tuning for optimal learning model

# Using **Bayesian HyperParamter Optimisation** for finding the best parameter tuning for Gradient Boosting Regressor

# In[ ]:


def logit_eval(learning_rate,n_estimators,max_features):
    params={'penalty':'l1'}
    params['learning_rate'] = max(learning_rate, 0)
    params['n_estimators'] = int(n_estimators)
    params['max_features'] = int(max_features)
    score = logitModel(params)
    return score
def logitModel(params):
    reg = GradientBoostingRegressor(learning_rate = params['learning_rate'], n_estimators = params['n_estimators'], max_features = params['max_features'])
    rocAuc = cross_val_score(reg, trainSelected1, y, cv=5,scoring='neg_mean_squared_error')
    return rocAuc.mean()
lgbBO = BayesianOptimization(logit_eval, {'learning_rate': (0.001,1.5),'n_estimators':(10,1000),'max_features':(1,len(trainSelected1.columns)+0.9)})
lgbBO.maximize(init_points=30, n_iter=300)


# Using the Optimal parameter setting obtained from the Bayesian optimisation, defining the optimal Gradient Boosting learner.
# 
# The optimal parameters I found during the training were -** learning_rate = 0.1848, n_estimators = 187, max_features = 9**

# In[ ]:


reg_opt = GradientBoostingRegressor(learning_rate = 0.1848, n_estimators = 187, max_features = 9)


# ## Fitting the model and Predicting

# In[ ]:


reg_opt.fit(trainSelected1,y)

pred =reg_opt.predict(test[selected1])
preds = np.power(pred,3)

sam=pd.read_csv('../input/sample_sub.csv')

sam['wave_height']=pred


# ## Saving to CSV

# In[ ]:


sam.to_csv('sandbox_sub.csv',index=False)


# ## Downloading file

# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(sam)


# 
