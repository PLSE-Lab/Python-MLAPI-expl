#!/usr/bin/env python
# coding: utf-8

# # **INTRODUCTION:**
# 
# This is a very simple implementation of XGBoost for this data. One of the great features of XGBoost is that it has built in functionality to easily see the top features F score. This means we can add a lot of features to the model and then easily see what really makes a difference in prediction. This is extremely important for this challenge since it is not really about the leaderboard results, it is more about determining useful features for models. 
# 
# I am working on collecting publicly available datasets to continue to add in new variables (features) to see what might be useful.

# In[ ]:


import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from google.cloud import bigquery


# # Load the data

# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")


# In[ ]:


test


# # Add Weather Data

# We will be doing this using the technique outlined in the great notebook https://www.kaggle.com/davidbnn92/weather-data?scriptVersionId=30695168

# In[ ]:


get_ipython().run_cell_magic('time', '', 'client = bigquery.Client()\ndataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")\ndataset = client.get_dataset(dataset_ref)\n\ntables = list(client.list_tables(dataset))\n\ntable_ref = dataset_ref.table("stations")\ntable = client.get_table(table_ref)\nstations_df = client.list_rows(table).to_dataframe()\n\ntable_ref = dataset_ref.table("gsod2020")\ntable = client.get_table(table_ref)\ntwenty_twenty_df = client.list_rows(table).to_dataframe()\n\nstations_df[\'STN\'] = stations_df[\'usaf\'] + \'-\' + stations_df[\'wban\']\ntwenty_twenty_df[\'STN\'] = twenty_twenty_df[\'stn\'] + \'-\' + twenty_twenty_df[\'wban\']\n\ncols_1 = [\'STN\', \'mo\', \'da\', \'temp\', \'min\', \'max\', \'stp\', \'wdsp\', \'prcp\', \'fog\']\ncols_2 = [\'STN\', \'country\', \'state\', \'call\', \'lat\', \'lon\', \'elev\']\nweather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index(\'STN\'), on=\'STN\')\n\nweather_df.tail(10)')


# In[ ]:


from scipy.spatial.distance import cdist

weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)
                                   + 31*(weather_df['mo']=='02') 
                                   + 60*(weather_df['mo']=='03')
                                   + 91*(weather_df['mo']=='04')  
                                   )

mo = train['Date'].apply(lambda x: x[5:7])
da = train['Date'].apply(lambda x: x[8:10])
train['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )

C = []
for j in train.index:
    df = train.iloc[j:(j+1)]
    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],
                weather_df[['lat','lon', 'day_from_jan_first']], 
                metric='euclidean')
    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)
    arr = new_df.values
    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)
    L = [i[i.astype(bool)].tolist()[0] for i in new_close]
    C.append(L[0])
    
train['closest_station'] = C

train = train.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)
train.sort_values(by=['Id'], inplace=True)
train.head()


# In[ ]:


from scipy.spatial.distance import cdist

weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)
                                   + 31*(weather_df['mo']=='02') 
                                   + 60*(weather_df['mo']=='03')
                                   + 91*(weather_df['mo']=='04')  
                                   )

mo = test['Date'].apply(lambda x: x[5:7])
da = test['Date'].apply(lambda x: x[8:10])
test['day_from_jan_first'] = (da.apply(int)
                               + 31*(mo=='02') 
                               + 60*(mo=='03')
                               + 91*(mo=='04')  
                              )

C = []
for j in test.index:
    df = test.iloc[j:(j+1)]
    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],
                weather_df[['lat','lon', 'day_from_jan_first']], 
                metric='euclidean')
    new_df = pd.DataFrame(mat, index=df.ForecastId, columns=weather_df.index)
    arr = new_df.values
    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)
    L = [i[i.astype(bool)].tolist()[0] for i in new_close]
    C.append(L[0])
    
test['closest_station'] = C

test = test.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)
test.sort_values(by=['ForecastId'], inplace=True)
test.head()


# Issue with wdsp and fog column being objects and not numeric, so change this

# In[ ]:


train["wdsp"] = pd.to_numeric(train["wdsp"])
test["wdsp"] = pd.to_numeric(test["wdsp"])


# In[ ]:


train["fog"] = pd.to_numeric(train["fog"])
test["fog"] = pd.to_numeric(test["fog"])


# Drop the two "y" columns

# In[ ]:


X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)


# In[ ]:


countries = X_train["Country/Region"]


# In[ ]:


countries.unique()


# Drop the Id column

# In[ ]:


X_train = X_train.drop(["Id"], axis=1)
X_test = test.drop(["ForecastId"], axis=1)


# Check the datatypes, they need to all be int, float, or bool for XGBoost

# In[ ]:


X_train.dtypes


# Change the Date column to be a datetime

# In[ ]:


X_train['Date']= pd.to_datetime(X_train['Date']) 
X_test['Date']= pd.to_datetime(X_test['Date']) 


# Set the index to the date

# In[ ]:


X_train = X_train.set_index(['Date'])
X_test = X_test.set_index(['Date'])


# # Create time features based on the new Date index

# In[ ]:


def create_time_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X


# In[ ]:


create_time_features(X_train)
create_time_features(X_test)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


X_train.drop("date", axis=1, inplace=True)
X_test.drop("date", axis=1, inplace=True)


# # Adding more variables to the mix

# I think now would be a good time to add a couple more variables from outside datasets to this mix to see if any of their data could provide further insight in our predictions. I have collected a couple from the World Bank as well as the UN. These datasets are nice since they have data listed for almost all countries in the world.
# 
# Lets start with the World Happiness Index dataset from the UN. It has some information related to GINI Coefficients, "social support", "Healthy Life Expectancy at Birth", Generosity, and Perceptions of Corruption. These indicators could capture some ideas around the healthcare setups in each country and also broad societal differences. They are very generic and broad so I wouldnt expect them to be extremely useful, and even if they show up as very informative on our predictions it would be tough to really break out true actionable insights from them but it is somewhere to start.
# 
# We will just grab the most recent value for each country (most this is 2018) to begin with. If you wanted to get a little more in depth you could probably take an average of the last 5 years or something like that but for now we will stay simple.

# In[ ]:


world_happiness_index = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")


# In[ ]:


world_happiness_grouped = world_happiness_index.groupby('Country name').nth(-1)


# In[ ]:


world_happiness_grouped.drop("Year", axis=1, inplace=True)


# In[ ]:


X_train = pd.merge(left=X_train, right=world_happiness_grouped, how='left', left_on='Country/Region', right_on='Country name')
X_test = pd.merge(left=X_test, right=world_happiness_grouped, how='left', left_on='Country/Region', right_on='Country name')


# In[ ]:


X_train


# In[ ]:


malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")


# In[ ]:


X_train = pd.merge(left=X_train, right=malaria_world_health, how='left', left_on='Country/Region', right_on='Country')
X_test = pd.merge(left=X_test, right=malaria_world_health, how='left', left_on='Country/Region', right_on='Country')


# In[ ]:


X_train


# In[ ]:


X_train.drop("Country", axis=1, inplace=True)
X_test.drop("Country", axis=1, inplace=True)


# In[ ]:


human_development_index = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")


# In[ ]:


X_train = pd.merge(left=X_train, right=human_development_index, how='left', left_on='Country/Region', right_on='Country')
X_test = pd.merge(left=X_test, right=human_development_index, how='left', left_on='Country/Region', right_on='Country')


# In[ ]:


X_train


# In[ ]:


X_train.drop(["Country", "Gross national income (GNI) per capita 2018"], axis=1, inplace=True)
X_test.drop(["Country", "Gross national income (GNI) per capita 2018"], axis=1, inplace=True)


# In[ ]:


night_ranger_predictors = pd.read_csv("../input/covid19-demographic-predictors/covid19_by_country.csv")


# In[ ]:


#There is a duplicate for Georgia in this dataset from Night Ranger, causing merge issues so we will just drop the Georgia rows
night_ranger_predictors = night_ranger_predictors[night_ranger_predictors.Country != "Georgia"]


# In[ ]:


X_train = pd.merge(left=X_train, right=night_ranger_predictors, how='left', left_on='Country/Region', right_on='Country')
X_test = pd.merge(left=X_test, right=night_ranger_predictors, how='left', left_on='Country/Region', right_on='Country')


# In[ ]:


X_train


# For now lets drop some of the columns I am not quite sure how to implement in my analysis yet. These may be very good variables to use but I will have to spend more time thinking about how to use them. Also MAKE SURE TO REMOVE the "Total Infected", "Total Deaths" columns as these are what we are trying to get our model to predict.

# In[ ]:


X_train.drop(["Country", "Restrictions","Quarantine", "Schools", "Total Infected", "Total Deaths"], axis=1, inplace=True)
X_test.drop(["Country", "Restrictions","Quarantine", "Schools", "Total Infected", "Total Deaths"], axis=1, inplace=True)


# In[ ]:


X_train.info(verbose=True)


# In[ ]:


X_test


# # One hot encode the Provice/State and the Country/Region columns

# In[ ]:


X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)
X_train.drop(['Province/State'],axis=1, inplace=True)
X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)
X_test.drop(['Province/State'],axis=1, inplace=True)


# In[ ]:


X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)
X_train.drop(['Country/Region'],axis=1, inplace=True)
X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)
X_test.drop(['Country/Region'],axis=1, inplace=True)


# # Grab the "y" variable we want to predict

# In[ ]:


y_train = train["Fatalities"]


# In[ ]:


y_train


# In[ ]:


X_train


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# In[ ]:


reg.fit(X_train, y_train, verbose=True)


# In[ ]:


plot = plot_importance(reg, height=0.9, max_num_features=20)


# # Use percentage change in the y variable instead of raw numbers
# 
# I think another interesting way to look at this might be through percentage change of the y variable. We really care about the percentage change in fatalities from day to day not the total number. This is because we know areas where the infection has been for a longer period of time would automatically have a higher total number, while areas with relatively new infection would have a lower total number of deaths but quite possible a higher percentage change since the virus is spreading more rapidly.

# Get the percentage change for each country, using one day lag. Would be interesting to play around with the lag time (periods) to see if this changes the analysis, my first thought would be to change this to 7 day (one week).

# In[ ]:


y_train = train.groupby(["Country/Region"]).Fatalities.pct_change(periods=1)


# There are issues with pct_change function returning NaN when doing percentage change from 0 to 0, so just change these to 0.

# In[ ]:


y_train = y_train.replace(np.nan, 0)


# There are also issues with pct_change function sometimes returning "inf" when going from 0 to 0, so just change these to 0

# In[ ]:


y_train = y_train.replace(np.inf, 0)


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# In[ ]:


reg.fit(X_train, y_train, verbose=True)


# In[ ]:


plot = plot_importance(reg, height=0.9, max_num_features=20)


# # Change y variable to Confirmed Cases
# 
# We will use the same train data as above but lets change the y variable to be Confirmed Cases and see if anything changes.

# In[ ]:


y_train = train["ConfirmedCases"]


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# In[ ]:


reg.fit(X_train, y_train, verbose=True)


# In[ ]:


plot = plot_importance(reg, height=0.9, max_num_features=20)


# In[ ]:


y_train = train.groupby(["Country/Region"]).ConfirmedCases.pct_change(periods=1)


# In[ ]:


y_train = y_train.replace(np.nan, 0)


# In[ ]:


y_train = y_train.replace(np.inf, 0)


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# In[ ]:


reg.fit(X_train, y_train, verbose=True)


# In[ ]:


plot = plot_importance(reg, height=0.9, max_num_features=20)


# # Try running on test data and submitting results

# One thing to note is that fatalities will always go up, so we will want to adjust any prediction that is less than the previous days prediction to be equal to the previous day. Also the same is true with confirmed cases.

# In[ ]:


y_train = train["ConfirmedCases"]
confirmed_reg = xgb.XGBRegressor(n_estimators=1000)
confirmed_reg.fit(X_train, y_train, verbose=True)
preds = confirmed_reg.predict(X_test)
preds = np.array(preds)
preds[preds < 0] = 0
preds = np.round(preds, 0)


# In[ ]:


preds = np.array(preds)


# In[ ]:


preds


# In[ ]:


submissionOrig = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")


# In[ ]:


submissionOrig["ConfirmedCases"]=pd.Series(preds)


# In[ ]:


submissionOrig


# So one further issue with the predictions is that XGBoost does not know that the predictions should be cumulative, so for each country we always want the predictions to be greater than or equal to the previous days. To fix this issue we should attached the predictions back to the original test dataframe so that we can grouby country/state and then for each grouping we need to make sure that the predicted variable is always equal the previous days prediction or greater.

# In[ ]:


test = test.join(submissionOrig["ConfirmedCases"])
test["Difference"] = test.groupby(["Country/Region"])["ConfirmedCases"].apply(lambda x: x.shift(1) - x)
for index, row in test.iterrows():
    if index>0:
        if row["Difference"] < 0:
            test.at[index,"ConfirmedCases"] = test.iloc[index-1]["ConfirmedCases"]


# In[ ]:


submissionOrig["ConfirmedCases"] = test["ConfirmedCases"]
test.drop("ConfirmedCases", axis=1, inplace=True)


# In[ ]:


y_train = train["Fatalities"]
confirmed_reg = xgb.XGBRegressor(n_estimators=1000)
confirmed_reg.fit(X_train, y_train, verbose=True)
preds = confirmed_reg.predict(X_test)
preds = np.array(preds)
preds[preds < 0] = 0
preds = np.round(preds, 0)
submissionOrig["Fatalities"]=pd.Series(preds)


# In[ ]:


test = test.join(submissionOrig["Fatalities"])
test["Difference"] = test.groupby(["Country/Region"])["Fatalities"].apply(lambda x: x.shift(1) - x)
for index, row in test.iterrows():
    if index>0:
        if row["Difference"] < 0:
            test.at[index,"Fatalities"] = test.iloc[index-1]["Fatalities"]


# In[ ]:


submissionOrig["Fatalities"] = test["Fatalities"]
test.drop("Fatalities", axis=1, inplace=True)


# In[ ]:


submissionOrig


# In[ ]:


submissionOrig.to_csv('submission.csv',index=False)

