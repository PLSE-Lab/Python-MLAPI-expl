#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Reading the DATA SET

# 
# In this notebook, I developed a model aimed at predicting flight delays at the Destination Airport. The purpose is to create a Data set which can be used for visualization and model Building so as to predict the Delays of Flights. I did the visualization so as to get better inferences about the data. For, model fitting I have seperated the Dataset into training Data and Testing Data so that prediction can be done on the testing Data. I also showed how to import Tableau and make visualization more crisp and clear.
# 
# Technical aspect Covered:
# 
# visualization: matplolib, seaborn, Tableau
# 
# data manipulation: pandas, numpy
# 
# modeling: sklearn
# 
# class definition: regression, Boosting, Bagging
# 
# For EDA I used some part of Python coding and Tableau Visulization so as to get a brief insight and inference from the data. Various Plots are created so as to get a great idea of whats happening in the Dataset and what is the most important variable affecting the dalays of the airlines. Feature scaling is a method used to normalize the range of independent variables or features of data and this concept is used.
# 
# 

# In[ ]:


data = pd.read_csv("../input/flight-delays/flights.csv")


# In[ ]:


airport = pd.read_csv('../input/flight-delays/airports.csv')
airlines = pd.read_csv('../input/flight-delays/airlines.csv')


# In[ ]:


data.info()


# In[ ]:


data.shape


# The Data Contains 31 columns and 1048575 Rows

# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data


# From the above table it is clear that data is not properly organised and date is given seperated and many columns have unnecessary data not useful for visualization for which it is required that we clean the data and take only those columns which is of our use.

# In[ ]:


airport.isnull().sum()


# In[ ]:


airport = airport.dropna(subset = ['LATITUDE','LONGITUDE'])


# In[ ]:


airport.isnull().sum()


# In[ ]:


airport.head(10)


# In[ ]:


airlines


# In[ ]:


Data_NULL = data.isnull().sum()*100/data.shape[0]
Data_NULL


# We can see that 96% of the values in Cancellation reason column are null for which it is of less use while predicting Delays. Some other columns include 78.2% in Air System Delay, Security Delay, Airline Delay, Weather Delay etc. So I am going to create two Dataset which is having no null values one is by removing all the null values irrespective of different types of Delays and other I am going to take the data set with respect to different types of delays. The first Dataset is named as Flights and the other one is named as Flight_Delays.

# In[ ]:


# Dropping of subset of null values
data1 = data.dropna(subset = ["TAIL_NUMBER",'DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME',
             'ELAPSED_TIME','AIR_TIME','WHEELS_ON','TAXI_IN','ARRIVAL_TIME','ARRIVAL_DELAY'])


# In[ ]:


data1.shape


# In[ ]:


data1.isnull().sum()


# In[ ]:


# Creting Dataset w.r.t different Types of Delays
data11 = data1.dropna(subset = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'])
data11 = data11.drop(['YEAR','MONTH','DAY','DAY_OF_WEEK','TAIL_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','SCHEDULED_TIME',
                     'SCHEDULED_ARRIVAL','ARRIVAL_TIME','DIVERTED','CANCELLED','CANCELLATION_REASON','FLIGHT_NUMBER','WHEELS_OFF',
                     'WHEELS_ON','AIR_TIME'],axis = 1)


# In[ ]:


data11.info()


# In[ ]:


# The other Dataset
Flight_Delays = data11


# In[ ]:


# Creating Dataset by removing null values by not focussing fully on different types of Delays
data2 = data1.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                    'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'],axis = 1)


# In[ ]:


data2.isnull().sum()


# In[ ]:


data2.shape


# In[ ]:


data2.info()


# In[ ]:


data2.DEPARTURE_TIME.isnull().sum()


# In[ ]:


data2.DEPARTURE_TIME.dtype


# In[ ]:


data2.DEPARTURE_TIME


# The departure time above is not very much informative so we are going to change it in the datetime format so that we get a better idea of the time.

# In[ ]:


# Creating a function to change the way of representation of time in the column
def Format_Hourmin(hours):
        if hours == 2400:
            hours = 0
        else:
            hours = "{0:04d}".format(int(hours))
            Hourmin = datetime.time(int(hours[0:2]), int(hours[2:4]))
            return Hourmin


# In[ ]:


data2['Actual_Departure'] =data1['DEPARTURE_TIME'].apply(Format_Hourmin)
data2['Actual_Departure']


# In[ ]:


data2.columns


# In[ ]:


# Creating Date in the Datetime format
data2['Date'] = pd.to_datetime(data2[['YEAR','MONTH','DAY']])
data2.Date


# In[ ]:


data2['Day'] = data2['Date'].dt.weekday_name


# In[ ]:


# Applying the function to required variables in the dataset
data2['Actual_Departure'] =data1['DEPARTURE_TIME'].apply(Format_Hourmin)
data2['Scheduled_Arrival'] =data1['SCHEDULED_ARRIVAL'].apply(Format_Hourmin)
data2['Scheduled_Departure'] =data1['SCHEDULED_DEPARTURE'].apply(Format_Hourmin)
data2['Actual_Arrival'] =data1['ARRIVAL_TIME'].apply(Format_Hourmin)


# # Merging of  3 data sets

# Since there are three dataset it is required to merge all the three data set so that we can use it during the visualization in a proper way.

# In[ ]:


# Merging on AIRLINE and IATA_CODE
data2 = data2.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')


# In[ ]:


data2 = data2.drop(['AIRLINE_x','IATA_CODE'], axis=1)


# In[ ]:


data2 = data2.rename(columns={"AIRLINE_y":"AIRLINE"})


# In[ ]:


data2 = data2.merge(airport, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='inner')
data2 = data2.merge(airport, left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='inner')


# In[ ]:


data2.columns


# In[ ]:


data2 = data2.drop(['LATITUDE_x', 'LONGITUDE_x',
       'STATE_y', 'COUNTRY_y', 'LATITUDE_y', 'LONGITUDE_y','STATE_x', 'COUNTRY_x'], axis=1)


# In[ ]:


data2 = data2.rename(columns={'IATA_CODE_x':'Org_Airport_Code','AIRPORT_x':'Org_Airport_Name','CITY_x':'Origin_city',
                             'IATA_CODE_y':'Dest_Airport_Code','AIRPORT_y':'Dest_Airport_Name','CITY_y':'Destination_city'})


# In[ ]:


data2


# In[ ]:


# we are taking the required data into Account for visualization and the Analysis
ReqdData = pd.DataFrame(data2[['AIRLINE','Org_Airport_Name','Origin_city',
                               'Dest_Airport_Name','Destination_city','ORIGIN_AIRPORT',
                               'DESTINATION_AIRPORT','DISTANCE','Actual_Departure','Date','Day',
                               'Scheduled_Departure','DEPARTURE_DELAY','Actual_Arrival','Scheduled_Arrival','ARRIVAL_DELAY',
                              'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','TAXI_IN','TAXI_OUT','DIVERTED',]])


# In[ ]:


data2.DEPARTURE_TIME.dtype


# In[ ]:


ReqdData = ReqdData.dropna(subset = ['Actual_Departure','Actual_Arrival'])


# In[ ]:


ReqdData.info()


# In[ ]:


# Cleaned Dataset for visualization and Analysis
Flights = ReqdData
Flights


# In[ ]:


plt.figure(figsize=(10, 10))
axis = sns.countplot(x=Flights['Origin_city'], data = Flights,
              order=Flights['Origin_city'].value_counts().iloc[:20].index)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# # The Figure shows that Atlanta has the highest count of flight from origin city 

# In[ ]:


plt.figure(figsize=(10, 10))
axis = sns.countplot(x=Flights['Org_Airport_Name'], data = Flights,
              order=Flights['Org_Airport_Name'].value_counts().iloc[:20].index)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:


axis = plt.subplots(figsize=(10,14))
Name = Flights["AIRLINE"].unique()
size = Flights["AIRLINE"].value_counts()
plt.pie(size,labels=Name,autopct='%5.0f%%')
plt.show()


# In[ ]:


axis = plt.subplots(figsize=(10,14))
sns.despine(bottom=True, left=True)
# Observations with Scatter Plot
sns.stripplot(x="ARRIVAL_DELAY", y="AIRLINE",
              data = Flights, dodge=True, jitter=True
            )
plt.show()


# American Airlines Inc has the highest Arrival Delay.

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1569936847474' style='position: relative'><noscript><a href='#'><img alt=' ' \nsrc='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;9B&#47;9BD44NRJ9&#47;1_rss.png' style='border: none'\n/></a></noscript><object class='tableauViz'  \nstyle='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' \n/> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;9BD44NRJ9' \n/> <param name='toolbar' value='yes' \n/><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;9B&#47;9BD44NRJ9&#47;1.png' \n/> <param name='animate_transition' value='yes'\n/><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' \n/><param name='display_overlay' value='yes' /><param name='display_count' value='yes' \n/><param name='filter' value='publish=yes' /></object></div>                \n<script type='text/javascript'>                    \nvar divElement = document.getElementById('viz1569936847474');                    \nvar vizElement = divElement.getElementsByTagName('object')[0];                    \nvizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    \nvar scriptElement = document.createElement('script');                    \nscriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    \nvizElement.parentNode.insertBefore(scriptElement, vizElement);               \n</script>")


# # Tableau Visualization to give more clear Insight and Inferences

# In[ ]:


# Plot to show the Taxi In and Taxi Out Time
axis = plt.subplots(figsize=(10,14))
sns.set_color_codes("pastel")
sns.set_context("notebook", font_scale=1.5)
axis = sns.barplot(x="TAXI_OUT", y="AIRLINE", data=Flights, color="g")
axis = sns.barplot(x="TAXI_IN", y="AIRLINE", data=Flights, color="r")
axis.set(xlabel="TAXI_TIME (TAXI_OUT: green, TAXI_IN: blue)")


# In[ ]:


axis = plt.subplots(figsize=(20,14))
sns.heatmap(Flights.corr(),annot = True)
plt.show()


# # Very High Correlation Between Arrival Delay and Departure Delay

# It shows that maximum of the Arrival Delays are due to the Departure Delays but some flights has still arrived on time even after departed late from the Origin Airport. Now we need to check why departure Delay is happening in the origin Airport Which may be due to security Delays, Air System Delays etc.

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1569938215526' style='position: relative'><noscript><a href='#'><img alt=' ' \nsrc='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;PH&#47;PHNKQPSZG&#47;1_rss.png' style='border: none'\n/></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' \n/> <param name='embed_code_version' value='3' \n/> <param name='path' value='shared&#47;PHNKQPSZG' \n/> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;PH&#47;PHNKQPSZG&#47;1.png' \n/> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes'\n/><param name='display_overlay' value='yes' \n/><param name='display_count' value='yes' /></object></div>                \n<script type='text/javascript'>                   \nvar divElement = document.getElementById('viz1569938215526');                   \nvar vizElement = divElement.getElementsByTagName('object')[0];                   \nvizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                   \nvar scriptElement = document.createElement('script');                    \nscriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                   \nvizElement.parentNode.insertBefore(scriptElement, vizElement);               \n</script>")


# # Tableau Visualization to give more clear Insight and Inferences

# # Prediction

# In[ ]:


Flights.head()


# In[ ]:


# Dropping unncecssary columns before prediction
Flights1 = Flights.drop(['Org_Airport_Name','Origin_city','Dest_Airport_Name','Destination_city'],axis = 1)


# In[ ]:


Flights1.columns


# In[ ]:


# To check the Distribution of Air Time
sns.distplot(Flights1['AIR_TIME'])
plt.show()


# In[ ]:


# To check the Distribution of Elapsed Time
sns.distplot(Flights1['ELAPSED_TIME'])
plt.show()


# In[ ]:


# To check the Distribution of Taxi IN
sns.distplot(Flights1['TAXI_IN'])
plt.show()


# In[ ]:


# To check the Distribution of Taxi out
sns.distplot(Flights1['TAXI_OUT'])
plt.show()


# In[ ]:


# importing Various regression algorithms 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso,LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


Las = Lasso()
LinR = LinearRegression()
Rid = Ridge()
Rfc = RandomForestRegressor(random_state=2)
Dtc = DecisionTreeRegressor(random_state = 2)
Boost_Lin = AdaBoostRegressor(base_estimator=LinR,random_state=2)
Boost_las = AdaBoostRegressor(base_estimator=Las,random_state=2)
Boost_rid = AdaBoostRegressor(base_estimator=Rid,random_state=2)
Bg_Lin = BaggingRegressor(base_estimator=LinR,random_state=2)
Bg_las = BaggingRegressor(base_estimator=Las,random_state=2)
Bg_rid = BaggingRegressor(base_estimator=Rid,random_state=2)


# In[ ]:


le = LabelEncoder()


# In[ ]:


# Label encoding features to change categorical variables into numerical one
Flights1['AIRLINE']= le.fit_transform(Flights1['AIRLINE'])
Flights1['ORIGIN_AIRPORT'] = le.fit_transform(Flights1['ORIGIN_AIRPORT'])
Flights1['DESTINATION_AIRPORT'] = le.fit_transform(Flights1['DESTINATION_AIRPORT'])
Flights1['Day'] = le.fit_transform(Flights1['Day'])


# In[ ]:


Flights1 = Flights1.drop(['Scheduled_Departure','Scheduled_Arrival','Actual_Arrival','Date','Actual_Departure'], axis = 1)


# In[ ]:


Flights1.info()


# In[ ]:


X = Flights1.drop('ARRIVAL_DELAY',axis = 1)
X.shape


# In[ ]:


y = Flights1['ARRIVAL_DELAY']
y.head()


# In[ ]:


# Splitting into train and test data set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 2)


# In[ ]:


sc1=StandardScaler()
X_train_sc=sc1.fit_transform(X_train)
X_test_sc=sc1.transform(X_test)


# # Model fitting and results

# In[ ]:


for model, name in zip([Las,LinR,Rid,Dtc,Rfc,Boost_Lin,Boost_las,Boost_rid,Bg_Lin,Bg_las,Bg_rid], 
     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Decision Tree Regressor','Boosted Linear',
      'Boosted Lasso','Boosted Ridge','Bagged Linear','Bagged Lasso','Bagged Ridge']):
    model1 = model.fit(X_train_sc,y_train)
    Y_predict=model1.predict(X_test_sc)
    print(name)
    print('Mean Absolute Error:', mean_absolute_error(y_test, Y_predict))  
    print('Mean Squared Error:', mean_squared_error(y_test, Y_predict))  
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, Y_predict)))
    print('R2 : ',r2_score(y_test, Y_predict))
    print()


# # Model Analysis

# In[ ]:


for model, name in zip([Las,LinR,Rid,Dtc,Rfc,Boost_Lin,Boost_las,Boost_rid,Bg_Lin,Bg_las,Bg_rid], 
     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Decision Tree Regressor','Boosted Linear',
      'Boosted Lasso','Boosted Ridge','Bagged Linear','Bagged Lasso','Bagged Ridge']):
    model1 = model.fit(X_train_sc,y_train)
    Y_predict=model1.predict(X_test_sc)
    print(name)
    plt.scatter(y_test, Y_predict)
    plt.title("Model Analysis")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()


# # When Flight_Delays is taken, where I took all the different Delays into Concern.

# I removed the null values present in all the different types of Delays and proceeded with prediction of the Arrival Delays. 

# In[ ]:


axis = plt.subplots(figsize=(10,14))
Name = Flight_Delays["AIRLINE"].unique()
size = Flight_Delays["AIRLINE"].value_counts()
plt.pie(size,labels=Name,autopct='%5.0f%%')
plt.show()


# In[ ]:


Flight_Delays


# In[ ]:


Flight_Delays['ORIGIN_AIRPORT'] = Flight_Delays['ORIGIN_AIRPORT'].astype(str)
Flight_Delays['DESTINATION_AIRPORT'] = Flight_Delays['DESTINATION_AIRPORT'].astype(str)


# In[ ]:


Flight_Delays['AIRLINE']= le.fit_transform(Flight_Delays['AIRLINE'])
Flight_Delays['ORIGIN_AIRPORT'] = le.fit_transform(Flight_Delays['ORIGIN_AIRPORT'])
Flight_Delays['DESTINATION_AIRPORT'] = le.fit_transform(Flight_Delays['DESTINATION_AIRPORT'])


# In[ ]:


X = Flight_Delays.drop('ARRIVAL_DELAY',axis = 1)
X.shape
y = Flight_Delays['ARRIVAL_DELAY']
y.head()


# In[ ]:


# Splitting the Data into Training and Testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 2)


# In[ ]:


# Scalling of the Data
sc1=StandardScaler()
X_train_sc=sc1.fit_transform(X_train)
X_test_sc=sc1.transform(X_test)


# # Model fitting and results

# In[ ]:


for model, name in zip([Las,LinR,Rid,Dtc,Rfc,Boost_Lin,Boost_las,Boost_rid,Bg_Lin,Bg_las,Bg_rid], 
     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Decision Tree Regressor','Boosted Linear',
      'Boosted Lasso','Boosted Ridge','Bagged Linear','Bagged Lasso','Bagged Ridge']):
    model1 = model.fit(X_train_sc,y_train)
    Y_predict=model1.predict(X_test_sc)
    print(name)
    print('Mean Absolute Error:', mean_absolute_error(y_test, Y_predict))  
    print('Mean Squared Error:', mean_squared_error(y_test, Y_predict))  
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, Y_predict)))
    print('R2 : ',r2_score(y_test, Y_predict))
    print()


# # Model Analysis

# In[ ]:


for model, name in zip([Las,LinR,Rid,Dtc,Rfc,Boost_Lin,Boost_las,Boost_rid,Bg_Lin,Bg_las,Bg_rid], 
     ['Lasso','Linear Regression','Ridge','Random forest Regressor','Decision Tree Regressor','Boosted Linear',
      'Boosted Lasso','Boosted Ridge','Bagged Linear','Bagged Lasso','Bagged Ridge']):
    model1 = model.fit(X_train_sc,y_train)
    Y_predict=model1.predict(X_test_sc)
    print(name)
    plt.scatter(y_test, Y_predict)
    plt.title("Model Analysis")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()


# # Conclusion

# We can see that maximum arrival Delays are dependent on the Departure Delays of the Airport. The first part is dealt with cleaning and exploration of the data set to get more insights and the second part dealt with the sitting of the model to predict the delays. For exploratory Analysis I used visualization tools like tableau and seaborn as well as matplotlib. The Second part dealt with the model fitting and predicting the Arrival delays of the airlines.
# 
# We can see that departure delay is the main problem which is creating Delay in the aviation industry. Departure Delays can be caused due Security Delay, Airline System Delays, Airlines Delay etc. The Delays affect the revenue of the company to a great extent so the delays has to be reduced as much as possible so as to increase the profitability in the Airline Industry. Customer Satisfaction will also be greatly enhanced if the delays can be brought down as low as possible.

# In[ ]:




