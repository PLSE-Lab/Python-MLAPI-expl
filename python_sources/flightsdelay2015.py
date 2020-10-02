#!/usr/bin/env python
# coding: utf-8

# > # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import datetime, warnings, scipy
import matplotlib.pyplot as plt
import math
import category_encoders as ce
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# Import different libraries to avoid futures errors. 'filterwarnings' is used to avoid warnings messages.

# # Import Datasets

# In[ ]:


airlines = pd.read_csv('../input/airlines.csv')
airports = pd.read_csv('../input/airports.csv')
flights = pd.read_csv('../input/flights.csv')


# If you are running the code on your computer, must check before path direction to import datasets. In Kaggle kernels datasets are imported as done above.

# # Previsualization

# In[ ]:


# visualize airlines dataset
airlines.head(10)


# Not many airlines in dataset. There are just airlines flying inside the country.

# In[ ]:


len(airlines)
print('There are %d airlines in dataset' % len(airlines))


# In[ ]:


# info airlines dataset
airlines.info()


# Not NA values in airlines dataset

# In[ ]:


# describe airlines dataset
airlines.describe()


# In[ ]:


# visualize airports dataset
airports.head()


# In airports dataset there are geolocate information that might be usefull in future analysis.

# In[ ]:


len(airports)
print('There are %d airports in dataset' % len(airports))


# In[ ]:


# info airports dataset
airports.info()


# There are just 3 airports with NA values in latitude and longitude.

# In[ ]:


# describe airports dataset
airports.describe()


# checking latitude and longitude, we can check the map location using Tableau Software using the following link.
# 
# https://public.tableau.com/profile/federico.garcia.blanco#!/vizhome/AirportsLocation/Sheet1?publish=yes

# In[ ]:


# visualize flights dataset
flights.head()


# In[ ]:


flights.tail()


# In[ ]:


len(flights)
print('There are %d flights in dataset' % len(flights))


# In[ ]:


# info flights dataset
flights.info()


# In[ ]:


# describe flights dataset
flights.describe()


# In[ ]:


# lower case to columns headers

airlines.columns = airlines.columns.str.lower()
airports.columns = airports.columns.str.lower()
flights.columns = flights.columns.str.lower()


# Use lower case to make easier the code from scratch. Will be usefull for future.

# # NA Values

# In[ ]:


flights.isna().sum()


# Many NA values in certian columns. Could be a problem if we keep these lines.

# In[ ]:


# drop column big amount NA values
flights = flights[flights.columns.difference(['cancellation_reason', 'air_system_delay', 'security_delay',
                                              'airline_delay', 'late_aircraft_delay', 'weather_delay'])]


# Drop certian columns to avoid NA values.

# In[ ]:


# drop NA values
flights = flights.dropna()


# After dropping certian columns, we drop lines with NA values. There are a few lines, not representative.

# In[ ]:


len(flights)
print('There are %d flights in dataset' % len(flights))


# # Exclude Certian Columns

# In[ ]:


# just keep important columns
# order columns
flights = flights[['month', 'day_of_week', 'airline', 'origin_airport', 'destination_airport',
                   'scheduled_departure', 'departure_delay', 'scheduled_arrival', 'arrival_delay',
                   'scheduled_time', 'elapsed_time', 'distance']]


# There are some columns that cannot be use for prediction model beacuase there is unknown information at the moment of prediction. Thats why we drop the columns.

# # Create New Columns

# In[ ]:


# new columns time_delay to check on air time delay
flights['time_delay'] = flights.elapsed_time - flights.scheduled_time


# We create the variable to visualize, but will not be usefull for prediction.

# # Visualization

# In[ ]:


# correlation matrix
sns.heatmap(flights.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# Correlation matrix show relation between variables. Target correlate variables cannot be use for prediction because there are unknown at the moment of prediction.

# In[ ]:


# day of week amount of flights
plt.hist(flights['day_of_week'],bins=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5])
plt.title("Histogram Days Of Week")
plt.xlim()
plt.show()


# We can see that Saturday is the less crowded day to travel by flight. Job flights would be the reason of this kind of distribution.

# In[ ]:


# distance distribution
plt.hist(flights['distance'], bins='auto')
plt.title("Histogram distance")
plt.xlim((50, 2000))
plt.show()


# Short distatance flights are more frecuent than long distance flights.

# In[ ]:


# most crowded origin airports ranking
airport_origin_weight = flights.groupby(['origin_airport']).month.count().sort_values(ascending = False)
airport_origin_weight.head(10)


# Ranking of the most crowded airports. Atlanta, Chicago and Dallas are the podium.

# # Padding Time

# In[ ]:


# market shere total flights per airline
airline_weight = flights.groupby(['airline']).month.count().sort_values(ascending = False)/len(flights)
airline_weight


# This is the market share ranking of total flights.

# In[ ]:


# market share "padding time" flights per airline
df_filter = flights[(flights['departure_delay'] >= 1 ) & (flights['arrival_delay'] <=3 ) 
        & (flights['arrival_delay'] >=-3 ) & (flights['time_delay'] <=-1 )]
airline_weight_filter = df_filter.groupby(['airline']).month.count().sort_values(ascending = False)/len(df_filter)
airline_weight_filter


# This is the market share of 'padding time' flights.

# In[ ]:


# rate of padding times per airlines
airline_weight = pd.DataFrame(airline_weight)
airline_weight_filter = pd.DataFrame(airline_weight_filter)
df_padding = pd.merge(airline_weight,airline_weight_filter,on='airline', how='left')
df_padding['rate'] = df_padding.month_y/df_padding.month_x
df_padding.rate.sort_values(ascending = False)


# This is the ranking of 'padding time' flights. We can see that Southwest Airlines Co.(WN) and United Air Lines Inc.(UA) are the airlines that encourage most the 'padding time'. For example, WN get 50% more of market share in 'padding time' flights than in total flights.

# # Drop Unnamed Airports Info 

# In[ ]:


# check unnamed airports
flights.origin_airport.unique()

There are some airports that are unnamed. Should be drop of dataset.
# In[ ]:


# drop unnamed airports rows. from 4250000th row airport is unnamed
flights = flights[0:4250000]


# In[ ]:


# create new dataset flights2. not necessary
flights2 = flights[:]


# # Code Hours Time

# In[ ]:


# hour truncated
flights2['scheduled_departure_hour'] = flights2.scheduled_departure
flights2['scheduled_arrival_hour'] = flights2.scheduled_arrival
flights2['scheduled_departure_hour'] = flights2.scheduled_departure/100
flights2['scheduled_arrival_hour'] = flights2.scheduled_arrival/100
flights2['scheduled_departure_hour'] = np.fix(flights2.scheduled_departure_hour)
flights2['scheduled_arrival_hour'] = np.fix(flights2.scheduled_arrival_hour)


# Truncate hours to make easier the analysis. Taking just the entire hour.

# # Create Dummies Variables

# Dummy variables to make easier the prediction models.

# In[ ]:


# days_of_week rename values
flights2.day_of_week = flights2.day_of_week.replace([1, 2, 3, 4, 5, 6, 7], ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
flights2.day_of_week.unique()


# In[ ]:


# create dummy variable for days_of_week
dummy = pd.get_dummies(flights2.day_of_week)
flights2 = pd.concat([flights2, dummy], axis=1)


# In[ ]:


# month rename values
flights2.month = flights2.month.replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                      ['january', 'february', 'march', 'april', 'may', 'june', 
                                       'july', 'august', 'september', 'october', 'november', 'december'])
flights2.month.unique()


# In[ ]:


# create dummy variable for months
dummy = pd.get_dummies(flights2.month)
flights2 = pd.concat([flights2, dummy], axis=1)


# # Create Binary Encoding Categories

# There are some attributes that have many categories so we should not convert to dummies, but must be numerical. The process to get numerical values without converting to dummies is binarizing. Like computers work =)

# In[ ]:


# binary encoding to airlines column
flights2['airline2'] = flights2.airline
encoder_airlines = ce.BinaryEncoder(cols=['airline'])
encoder_airlines.fit(flights2)
flights2 = encoder_airlines.transform(flights2)


# In[ ]:


# binary encoding to origin_airport column
flights2['origin_airport2'] = flights2.origin_airport
encoder_origin_airport = ce.BinaryEncoder(cols=['origin_airport'])
flights2 = encoder_origin_airport.fit_transform(flights2)


# In[ ]:


# binary encoding to destination_airport column
flights2['destination_airport2'] = flights2.destination_airport
encoder_destination_airport = ce.BinaryEncoder(cols=['destination_airport'])
flights2 = encoder_destination_airport.fit_transform(flights2)


# In[ ]:


# binary encoding to scheduled_departure_hour column
flights2['scheduled_departure_hour2'] = flights2.scheduled_departure_hour
encoder_scheduled_departure_hour = ce.BinaryEncoder(cols=['scheduled_departure_hour'])
flights2 = encoder_scheduled_departure_hour.fit_transform(flights2)


# In[ ]:


# binary encoding to scheduled_arrival_hour column
flights2['scheduled_arrival_hour2'] = flights2.scheduled_arrival_hour
encoder_scheduled_arrival_hour = ce.BinaryEncoder(cols=['scheduled_arrival_hour'])
flights2 = encoder_scheduled_arrival_hour.fit_transform(flights2)


# # Drop Certian Columns

# Drop certian columns that are not useful to the model.

# In[ ]:


flights3 = flights2[flights2.columns.difference(['month', 'day_of_week', 'scheduled_departure', 
                                                'scheduled_arrival', 'elapsed_time', 'time_delay', 'airline2',
                                                'origin_airport2', 'destination_airport2', 'scheduled_departure_hour2',
                                                'scheduled_arrival_hour2', 'departure_delay'])] # departure_delay
# drop arrival_delay outliers
flights3 = flights3[flights3['arrival_delay']<500]


# # Error Metrics RMSE

# Define the metric error. Being a continuos variable problem, RMSE is a good method.

# In[ ]:


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


# # Baseline

# This is the baseline. Any model we can create, must be better than airlines model. The new model must get a lower RMSE.

# In[ ]:


rmse_baseline = rmse(flights3.arrival_delay,0)
print('The RSME BaseLine is',rmse_baseline)


# # Normalize Variables

# The process of normalizing variables is needed to compare variables.

# In[ ]:


# standar normalize arrival_delay, distance, schedule_time and departure_delay

std_arrival_delay = flights3.arrival_delay.std()
mean_arrival_delay = flights3.arrival_delay.mean()

flights3.arrival_delay=(flights3.arrival_delay-flights3.arrival_delay.mean())/flights3.arrival_delay.std()
flights3.distance=(flights3.distance-flights3.distance.mean())/flights3.distance.std()
flights3.scheduled_time=(flights3.scheduled_time-flights3.scheduled_time.mean())/flights3.scheduled_time.std()
#flights3.departure_delay=(flights3.departure_delay-flights3.departure_delay.mean())/flights3.departure_delay.std()


# # Train Test Split

# Make a train test splir datastet to avoid overfitting process. Test dataset will be 30% of the total data. Train dataset will be 70% of the total data.

# In[ ]:


# split 30% testing 70% training dataset
train,test=train_test_split(flights3,test_size=0.3,random_state=0)
train_X=train[train.columns.difference(['arrival_delay'])]
train_Y=train['arrival_delay']
test_X=test[test.columns.difference(['arrival_delay'])]
test_Y=test['arrival_delay']


# # Cross Validation

# Cross Validation process is usefull to avoid overfitting too. If there are an overfitting process, the algorithm will learn very well to predict the class. The algorithm must be useful working with new data, thats why avoiding overfitting is so important

# In[ ]:


#from sklearn.model_selection import KFold #for K-fold cross validation
#from sklearn.model_selection import cross_val_score #score evaluation
#from sklearn.model_selection import cross_val_predict #prediction
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.linear_model import LinearRegression

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[] # el % de aciertos en la matriz de confusion
std=[]
classifiers=['DecisionTree', 'LinearRegression']
models=[DecisionTreeRegressor(), LinearRegression()] # Decision Tree Model
for i in models:
    model = i
    cv_result = cross_val_score(model,train_X,train_Y, cv = kfold,scoring = "neg_mean_squared_error")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# # Apply Model

# Applying model to predict the class

# In[ ]:


final_model=LinearRegression()
final_model.fit(train_X,train_Y) #con la data train_X predice el valor de train_Y
prediction=final_model.predict(test_X)
rsme_denormalize = rmse(prediction*std_arrival_delay+mean_arrival_delay,
                        test_Y*std_arrival_delay+mean_arrival_delay)
print('The RSME of the Linear Regression is',rsme_denormalize)


# The results are not the best. The RMSE is very high and its almost the same as airlines preditcions. Adding some weather information woulb be usefull to improve the prediction. 

# In[ ]:




