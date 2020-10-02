#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction import FeatureHasher
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from pyearth import Earth


# In[ ]:


aggregated_data = pd.read_csv("../input/raw_aggregated_filteredCompany.csv")


# **Feature Engineering**

# In[ ]:


aggregated_data.head(5)


# Checking if nulls are present in the dataset in any of the columns

# In[ ]:


aggregated_data.isnull().sum()


# Replacing nulls with median for coutinous variables

# In[ ]:


aggregated_data['tolls'].fillna((aggregated_data['tolls'].median()), inplace=True)
aggregated_data['fare'].fillna((aggregated_data['fare'].median()), inplace=True)
aggregated_data['tips'].fillna((aggregated_data['tips'].median()), inplace=True)
aggregated_data['extras'].fillna((aggregated_data['extras'].median()), inplace=True)
aggregated_data['trip_total'].fillna((aggregated_data['trip_total'].median()), inplace=True)


# One hot encoding for categorical variables except for pick_up (and drop_off) census and community

# In[ ]:


dummy = pd.get_dummies(aggregated_data.month_of_year, prefix='Month_flag')
aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)

dummy = pd.get_dummies(aggregated_data.day_of_week, prefix='Day_week_flag')
aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)

dummy = pd.get_dummies(aggregated_data.payment_type, prefix='Card_Type_flag')
aggregated_data = pd.concat([aggregated_data,dummy], axis = 1)


# Hashing Code Hashing company column as number of distinct values is high in it -Unused

# In[ ]:


#fh = FeatureHasher(n_features=5, input_type='string')
#hashed_features = fh.fit_transform(aggregated_data['company'])
#hashed_features = hashed_features.toarray()
#aggregated_data=pd.concat([aggregated_data, pd.DataFrame(hashed_features,columns=['hasha', 'hashb', 'hashc','hashd','hashe'])], axis=1)


# **Univariate Analysis**

# Checking the distribution of taxi usage though the week

# In[ ]:


# Day_of_week has Sunday as 1, Monday as 2 ... and Saturday as 7
sns.countplot(aggregated_data.day_of_week)
plt.show()


# The usage of taxi is the highest during the weekends and least on Monday with subsequent increase as the week progress. This behaviour shows that people go out more over the weekends and take lesser taxis to go to their respective work place.

# Checking the taxi usage through different months

# In[ ]:


sns.countplot(aggregated_data.month_of_year)
plt.show()


# The taxi usage is highest in the month of June and not during the holiday season (December and January)

# Trip distance

# In[ ]:


plt.figure(figsize = (20,5))
sns.boxplot(aggregated_data.trip_miles)
plt.show()


# As we can see from the above plot there as trips that go beyond 700 miles, but since we have already applied the speed filters these trips may be outstation trips

# Fare distribution

# In[ ]:


plt.figure(figsize = (20,5))
sns.boxplot(aggregated_data.fare)
plt.show()


# In[ ]:


aggregated_data.fare.groupby(pd.cut(aggregated_data.fare, np.arange(1,max(aggregated_data.fare),100))).count()


# Removing all the records which has fare over 500 dollars as they are outliers and chicago is one of the cheapest place to travel in taxi with around 12 dollars fare for 5 Miles with 5 Minutes waiting time. Also, the fare for max miles rode is 314 dollars. Refrence: https://www.world-airport-codes.com/united-states/chicago/chicago-ohare-international/taxis.html

# In[ ]:


aggregated_data = aggregated_data[aggregated_data.fare <= 500]


# Analyzing trip_seconds

# In[ ]:


aggregated_data.trip_seconds.describe()


# In[ ]:


plt.figure(figsize = (20,5))
sns.boxplot(aggregated_data.trip_seconds)
plt.show()


# Some trip durations are over 80000 seconds which is close to 24 hours and can point towards a weekend out of town trip.

# In[ ]:


aggregated_data.trip_seconds.groupby(pd.cut(aggregated_data.trip_seconds, np.arange(1,max(aggregated_data.trip_seconds),3600))).count()


# The distribution depicts that the distrbution of response variable is right skewed

# In[ ]:


aggregated_data.trip_seconds.groupby(pd.cut(aggregated_data.trip_seconds, np.arange(1,7200,600))).count().plot(kind='barh')
plt.xlabel('Trip Counts')
plt.ylabel('Trip Duration (seconds)')
plt.show()


# Most of the trips took 0 - 30 mins to complete i.e. approx 1800 secs

# **Bi-Variate Analysis**

# Trip Seconds per hour

# In[ ]:


group1 = aggregated_data.groupby('peak_hours_flag').trip_seconds.mean().plot(kind='barh')
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Pickup Hour')
plt.show()


# This clearly depicts that average trip duration is higher for peak hours than non-peak hours.

# Trip Seconds per weekday

# In[ ]:


group2 = aggregated_data.groupby('day_of_week').trip_seconds.mean()
sns.pointplot(group2.index, group2.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Weekday')
plt.show()


# We can clearly see that the trip duration is varying between weekdays and weekends.

# Trip Seconds per Month

# In[ ]:


group2 = aggregated_data.groupby('month_of_year').trip_seconds.mean()
sns.pointplot(group2.index, group2.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('Month')
plt.show()


# Trip Seconds per payment type

# In[ ]:


group2 = aggregated_data.groupby('payment_type').trip_seconds.mean()
sns.pointplot(group2.index, group2.values)
plt.ylabel('Trip Duration (seconds)')
plt.xlabel('payment_type')
plt.show()


# The average duration within different payment methods only vary within a range of 350 seconds.

# **Feature Selection and Linear Regression**

# In[ ]:


#First chech the index of the features and label
list(zip( range(0,len(aggregated_data.columns)),aggregated_data.columns))


# In[ ]:


index=['peak_hours_flag','day_hours_flag','night_hours_flag','pickup_census_tract','dropoff_census_tract','pickup_community_area','dropoff_community_area',
      'tolls','fare','tips','extras','trip_total','trip_miles','Month_flag_1','Month_flag_2','Month_flag_3','Month_flag_4','Month_flag_5','Month_flag_6','Month_flag_7','Month_flag_8','Month_flag_9','Month_flag_10',
      'Month_flag_11','Month_flag_12','Day_week_flag_1','Day_week_flag_2','Day_week_flag_3','Day_week_flag_4','Day_week_flag_5','Day_week_flag_6','Day_week_flag_7','Card_Type_flag_Cash','Card_Type_flag_Credit Card',
      'Card_Type_flag_Dispute','Card_Type_flag_Mobile','Card_Type_flag_No Charge','Card_Type_flag_Pcard','Card_Type_flag_Prcard','Card_Type_flag_Prepaid','Card_Type_flag_Split','Card_Type_flag_Unknown',
      'Card_Type_flag_Way2ride']
X = aggregated_data[index].values
Y = aggregated_data.iloc[:,16].values


# Backward Elimination

# In[ ]:


#Select all the features in X array
X_opt = X[:,range(0,43)]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#Fetch p values for each feature
p_Vals = regressor_OLS.pvalues

#define significance level for accepting the feature.
sig_Level = 0.05

#Loop to iterate over features and remove the feature with p value less than the sig_level
while max(p_Vals) > sig_Level:
    print("Probability values of each feature \n")
    print(p_Vals)
    X_opt = np.delete(X_opt, np.argmax(p_Vals), axis = 1)
    print("\n")
    print("Feature at index {} is removed \n".format(str(np.argmax(p_Vals))))
    print(str(X_opt.shape[1]-1) + " dimensions remaining now... \n")
    regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
    p_Vals = regressor_OLS.pvalues
    print("=================================================================\n")
    
#Print final summary
print("Final stat summary with optimal {} features".format(str(X_opt.shape[1]-1)))
regressor_OLS.summary()


# Spliting the data in test and train

# In[ ]:


#Split raw data
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=4, test_size=0.2)

#Split data from the feature selection group
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_opt,Y, random_state=4, test_size=0.2)


# Applying PCA to reduce the dimensionality

# In[ ]:


X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,Y, random_state=4, test_size=0.2)


# Standerdizing the input to perform PCA

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)


# Applying PCA

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_train_pca)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("Cumulative explained variance")
plt.show()


# In[ ]:


arr = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
list(zip(range(1,len(arr)), arr))


# To capture 99% variance we need to have 38 features

# In[ ]:


pca_10 = PCA(n_components=38)
X_train_pca = pca_10.fit_transform(X_train_pca)
X_test_pca = pca_10.transform(X_test_pca)


# Using correlation to check if PCA successfully removed all the collinearity

# In[ ]:


#plt.figure(figsize=(15,15))
#corr = pd.DataFrame(X_train_pca).corr()
#corr.index = pd.DataFrame(X_train_pca).columns
#sns.heatmap(corr, cmap='RdYlGn', vmin=-1, vmax=1, square=True)
#plt.title("Correlation Heatmap", fontsize=16)
#plt.show()


# Using linear regression

# In[ ]:


#Linear regressor for the raw data
regressor = LinearRegression() 
regressor.fit(X_train,y_train) 

#Linear regressor for the Feature selection group
regressor1 = LinearRegression() 
regressor1.fit(X_train_fs,y_train_fs) 

#Linear regressor for the Feature extraction group
regressor2 = LinearRegression() 
regressor2.fit(X_train_pca,y_train_pca) 


# Model Prediction

# In[ ]:


#Predict from the test features of raw data
y_pred = regressor.predict(X_test) 

#Predict from the test features of Feature Selection group
y_pred = regressor1.predict(X_test_fs) 

#Predict from the test features of Feature Extraction group
y_pred_pca = regressor2.predict(X_test_pca) 


# Model Valuation As mentioned in requirement model selection should be done via MAPE

# In[ ]:


#Evaluate the regressor on the raw data
print('MAPE for the Multiple LR raw is : {}'.format(np.mean(np.abs((y_test-y_pred)/y_test))*100))
print("\n")

#Evaluate the regressor on the Feature selection group
print('MAPE for the Multiple LR FS is : {}'.format(np.mean(np.abs((y_test_fs-y_pred)/y_test_fs))*100))
print("\n")

#Evaluate the regressor on the Feature extraction group
print('MAPE for the Multiple LR PCA is : {}'.format(np.mean(np.abs((y_test_pca-y_pred_pca)/y_test_pca))*100))


# The output of all 3 are inline and accuracy is not upto the mark.

# Exploring other models as this depicts that there is very little linear dependencies between variables
