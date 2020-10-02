#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# >With environmental issues and health becoming trending topics, usage of bicycles as a mode of transportation has gained traction in recent years. To encourage bike usage, cities across the world have successfully rolled out bike sharing programs. Under such schemes, riders can rent bicycles using manual/automated kiosks spread across the city for defined periods. In most cases, riders can pick up bikes from one location and return them to any other designated place.
# 
# >The bike sharing platforms from across the world are hotspots of all sorts of data, ranging from travel time, start and end location, demographics of riders, and so on. This data along with alternate sources of information such as weather, traffic, terrain, and so on makes it an attractive proposition for different research areas.
# 
# >The Capital Bike Sharing dataset contains information related to one such bike sharing program underway in Washington DC. Given this augmented (bike sharing details along with weather information) dataset, can we forecast bike rental demand for this program?

# In[ ]:


# data manipulation 
import numpy as np
import pandas as pd


# modeling utilities
import scipy.stats as stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import  linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# plotting
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

sn.set_style('whitegrid')
sn.set_context('talk')

plt.rcParams.update(params)
pd.options.display.max_colwidth = 600

# pandas display data frames as tables
from IPython.display import display, HTML


# ### PREPROCESSING EDA

# In[ ]:


## Step 1: loading data
hour_df = pd.read_csv('../input/hour.csv')
hour_df.head(2)


# In[ ]:


print("Shape of dataset::{}".format(hour_df.shape))


# In[ ]:


## Step 2: Data types
hour_df.dtypes


# In[ ]:


### Renaming column namees to being more readable
hour_df.rename(columns={'instant':'rec_id',
                            'dteday':'datetime',
                            'holiday':'is_holiday',
                            'workingday':'is_workingday',
                            'weathersit':'weather_condition',
                            'hum':'humidity',
                            'mnth':'month',
                            'cnt':'total_count',
                            'hr':'hour',
                            'yr':'year'},inplace=True)


# In[ ]:


hour_df.columns


# In[ ]:


# dataset summary stats
hour_df.describe()


# ### Setting weather to proper datetime and categorical data

# In[ ]:


# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')


# In[ ]:


hour_df.dtypes


# In[ ]:


hour_df.head(1)


# ## Visualize Attributes, Trends and Relationships
# 
# ### Hourly distribution of Total Counts
# + Seasons are encoded as 1:spring, 2:summer, 3:fall, 4:winter
# + Exercise: Convert season names to readable strings and visualize data again

# In[ ]:


fig,ax = plt.subplots()
sn.pointplot(data=hour_df[['hour',
                           'total_count',
                           'season']],
             x='hour',y='total_count',
             hue='season',ax=ax)
ax.set(title="Season wise hourly distribution of counts")


# In[ ]:


hour_df[['hour',
                           'total_count',
                           'season']].head(10)


# Analysis From Chart
# + The above plot shows peaks around 8am and 5pm (office hours)
# + Overall higher usage in the second half of the day

# In[ ]:


fig,ax = plt.subplots()
sn.pointplot(data=hour_df[['hour','total_count','weekday']],x='hour',y='total_count',hue='weekday',ax=ax)
ax.set(title="Weekday wise hourly distribution of counts")


# + Weekends (0 and 6) and Weekdays (1-5) show different usage trends with weekend's peak usage in during afternoon hours
# + Weekdays follow the overall trend, similar to one visualized in the previous plot
# + Weekdays have higher usage as compared to weekends
# + It would be interesting to see the trends for casual and registered users separately

# In[ ]:


fig,ax = plt.subplots()
sn.boxplot(data=hour_df[['hour','total_count']],x="hour",y="total_count",ax=ax)
ax.set(title="Box Plot for hourly distribution of counts")


# + Early hours (0-4) and late nights (21-23) have low counts but significant outliers
# + Afternoon hours also have outliers
# + Peak hours have higher medians and overall counts with virtually no outliers

# ### Monthly distribution of Total Counts

# In[ ]:


fig,ax = plt.subplots()
sn.barplot(data=hour_df[['month',
                         'total_count']],
           x="month",y="total_count")
ax.set(title="Monthly distribution of counts")


# + Months June-Oct have highest counts. Fall seems to be favorite time of the year to use cycles

# In[ ]:


df_col_list = ['month','weekday','total_count']
plot_col_list= ['month','total_count']
spring_df = hour_df[hour_df.season==1][df_col_list]
summer_df = hour_df[hour_df.season==2][df_col_list]
fall_df = hour_df[hour_df.season==3][df_col_list]
winter_df = hour_df[hour_df.season==4][df_col_list]

fig,ax= plt.subplots(nrows=2,ncols=2)
sn.barplot(data=spring_df[plot_col_list],x="month",y="total_count",ax=ax[0][0],)
ax[0][0].set(title="Spring")

sn.barplot(data=summer_df[plot_col_list],x="month",y="total_count",ax=ax[0][1])
ax[0][1].set(title="Summer")

sn.barplot(data=fall_df[plot_col_list],x="month",y="total_count",ax=ax[1][0])
ax[1][0].set(title="Fall")

sn.barplot(data=winter_df[plot_col_list],x="month",y="total_count",ax=ax[1][1])  
ax[1][1].set(title="Winter")


# ### Year Wise Count Distributions

# In[ ]:


sn.violinplot(data=hour_df[['year',
                            'total_count']],
              x="year",y="total_count")


# + Both years have multimodal distributions
# + 2011 has lower counts overall with a lower median
# + 2012 has a higher max count though the peaks are around 100 and 300 which is then tapering off

# ### Working Day Vs Holiday Distribution

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2)
sn.barplot(data=hour_df,x='is_holiday',y='total_count',hue='season',ax=ax1)
sn.barplot(data=hour_df,x='is_workingday',y='total_count',hue='season',ax=ax2)


# ### Outliers
# 
# While exploring and learning about any dataset , it is imperative that we check for extreme and unlikely values. Though we handle missing and incorrect information while preprocessing the dataset, outliers are usually caught during EDA. Outliers can severely and adversely impact the downstream steps like modeling and the results.
# We usually utilize boxplots to check for outliers in the data. In the following snippet, we analyze outliers for numeric attributes like total_count, temperature, and wind_speed.

# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2)
sn.boxplot(data=hour_df[['total_count',
                         'casual','registered']],ax=ax1)
sn.boxplot(data=hour_df[['temp','windspeed']],ax=ax2)


# The generated plot is shown in Figure 6-6. We can easily mark out that for the three count related attributes, all of them seem to have a sizable number of outlier values. The casual rider distribution has overall lower numbers though. For weather attributes of temperature and wind speed, we find outliers only in the case of wind speed.

# In[ ]:


hour_df.columns


# In[ ]:


fig,ax1= plt.subplots()
sn.boxplot(data=hour_df[['total_count',
                         'hour']],x="hour",y="total_count",ax=ax1)


# ### CORRELATIONS
# Correlation helps us understand relationships between different attributes of the data. Since this chapter focuses on forecasting, correlations can help us understand and exploit relationships to build better models.
# Note
# It is important to understand that correlation does not imply causation. We strongly encourage you to explore more on the same.
# The following snippet first prepares a correlational matrix using the pandas utility function corr(). It then uses a heat map to plot the correlation matrix.

# In[ ]:


corrMatt = hour_df[["temp","atemp",
                    "humidity","windspeed",
                    "casual","registered",
                    "total_count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
sn.heatmap(corrMatt, mask=mask,
           vmax=.8, square=True,annot=True)


# + Correlation between temp and atemp is very high (as expected)
# + Same is te case with registered-total_count and casual-total_count
# + Windspeed to humidity has negative correlation
# + Overall correlational statistics are not very high.

# ### Regression Analysis
# 
# Regression analysis is a statistical modeling technique used by statisticians and Data Scientists alike. It is the process of investigating relationships between dependent and independent variables. Regression itself includes a variety of techniques for modeling and analyzing relationships between variables. It is widely used for predictive analysis, forecasting, and time series analysis.
# The dependent or target variable is estimated as a function of independent or predictor variables. The estimation function is called the regression function .
# Note
# In a very abstract sense, regression is referred to estimation of continuous response/target variables as opposed to classification, which estimates discrete targets.
# 
# 
# ### ASSUMPTIONS
# Regression analysis has a few general assumptions while specific analysis techniques have added (or reduced) assumptions as well. The following are important general assumptions for regression analysis:
# The training dataset needs to be representative of the population being modeled.
# The independent variables are linearly independent, i.e., one independent variable cannot be explained as a linear combination of others. In other words, there should be no multicollinearity.
# Homoscedasticity of error, i.e. the variance of error, is consistent across the sample .

# In[ ]:


def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
        column.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded

    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series

    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return le,ohe,features_df

# given label encoder and one hot encoder objects, 
# encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df


# In[ ]:


X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3], hour_df.iloc[:,-1], 
                                                    test_size=0.33, random_state=42)

X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()

print("Training set::{}{}".format(X.shape,y.shape))
print("Testing set::{}".format(X_test.shape))


# Normality Test

# In[ ]:


stats.probplot(y.total_count.tolist(), dist="norm", plot=plt)
plt.show()


# In[ ]:


cat_attr_list = ['season','is_holiday',
                 'weather_condition','is_workingday',
                 'hour','weekday','month','year']
numeric_feature_cols = ['temp','humidity','windspeed','hour','weekday','month','year']
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']


# In[ ]:


encoded_attr_list = []
for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


# In[ ]:


feature_df_list = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df']                         for enc in encoded_attr_list                         if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
print("Shape::{}".format(train_df_new.shape))


# In[ ]:


X = train_df_new
y= y.total_count.values.reshape(-1,1)

lin_reg = linear_model.LinearRegression()


# In[ ]:


#Cross Validation
predicted = cross_val_predict(lin_reg, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, y-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# In[ ]:


r2_scores = cross_val_score(lin_reg, X, y, cv=10)
mse_scores = cross_val_score(lin_reg, X, y, cv=10,scoring='neg_mean_squared_error')


# In[ ]:


fig, ax = plt.subplots()
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('R-Squared')
ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
plt.show()


# In[ ]:


print("R-squared::{}".format(r2_scores))
print("MSE::{}".format(mse_scores))


# In[ ]:


lin_reg.fit(X,y)


# In[ ]:


test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,
                                                              le,ohe,
                                                              col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df']                              for enc in test_encoded_attr_list                              if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Shape::{}".format(test_df_new.shape))


# In[ ]:


test_df_new.head()


# In[ ]:


X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)

y_pred = lin_reg.predict(X_test)

residuals = y_test-y_pred


# In[ ]:


r2_score = lin_reg.score(X_test,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f"
      % metrics.mean_squared_error(y_test, y_pred))


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
plt.show()


# In[ ]:


import statsmodels.api as sm

# Set the independent variable
X = X.values.tolist()

# This handles the intercept. 
# Statsmodel takes 0 intercept by default
X = sm.add_constant(X)

X_test = X_test.values.tolist()
X_test = sm.add_constant(X_test)


# Build OLS model
model = sm.OLS(y, X)
results = model.fit()

# Get the predicted values for dependent variable
pred_y = results.predict(X_test)

# View Model stats
print(results.summary())


# In[ ]:


plt.scatter(pred_y,y_test)

