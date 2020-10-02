#!/usr/bin/env python
# coding: utf-8

# # This dataset contains the climbing statistics and weather reports for year 2014 and 2015 for Mt Rainier
# 
# Field Description
# 
# 1. Date : They day of the record
# 2. Battery Voltage AVG: The average voltage of the day captured
# 3. Temperate AVG:  The average temperature of the day in Farenheit
# 4. Relative Humidity: The value that depicts the average humidity of the day 
# 5. Wind Speed Daily Average: Average wind speed on that day in mph
# 6. Wind Directon Average: Average direction of the wind in deg
# 7. Solar Radiation AVG: The average solar radiation of the day in Watts/square metre
# 8. Route: There are several routes through which people attempt to climb Mt Rainier
# 9. Attempted: Total number of people who attempted the climb
# 10. Succeeded: Total number of people who have reach the summit
# 11. Success Percentage: The ration of Succeeded to attempted (target) 
# 

# In[ ]:


pip install sorted_months_weekdays


# In[ ]:


pip install sort_dataframeby_monthorweek


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import calendar as calendar

from sorted_months_weekdays import *

from sort_dataframeby_monthorweek import *

from scipy.stats import zscore
import matplotlib.pyplot as plt
import os


# In[ ]:


print(os.listdir('../input/mount-rainier-weather-and-climbing-data/'))


# In[ ]:


#Import the Weather data ( Available only between 2014 and 2015)
dbWeather=pd.read_csv('../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv')


# In[ ]:


dbWeather.info()


# In[ ]:


dbWeather.isnull().values.any()


# In[ ]:


dbWeather['Date'] = pd.to_datetime(dbWeather['Date'].str.strip(), format='%m/%d/%Y')


# In[ ]:


#Import the Climbing Statistics ( Available only between 2014 and 2015)
dbclimbs=pd.read_csv('../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv')


# In[ ]:


dbclimbs.isnull().values.any()


# In[ ]:


dbclimbs['Date'] = pd.to_datetime(dbclimbs['Date'].str.strip(), format='%m/%d/%Y')


# In[ ]:


dbclimbs.info()


# In[ ]:


df=pd.DataFrame(columns=['Date','Battery Voltage AVG','Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG','Wind Direction AVG','Solar Radiation AVG','Route','Attempted','Succeeded','Success Ratio']) 


# In[ ]:


for index,row in dbclimbs.iterrows():
    weatherRow=dbWeather.loc[dbWeather['Date'] == row['Date']]
    df=df.append({'Date':row['Date'],'Battery Voltage AVG':weatherRow['Battery Voltage AVG'].to_string(index=False,header=False),'Temperature AVG':weatherRow['Temperature AVG'].to_string(index=False,header=False),'Relative Humidity AVG':weatherRow['Relative Humidity AVG'].to_string(index=False,header=False),'Wind Speed Daily AVG':weatherRow['Wind Speed Daily AVG'].to_string(index=False,header=False),'Wind Direction AVG':weatherRow['Wind Direction AVG'].to_string(index=False,header=False),'Solar Radiation AVG':weatherRow['Solare Radiation AVG'].to_string(index=False,header=False),'Route':row['Route'],'Attempted':row['Attempted'],'Succeeded':row['Succeeded'],'Success Ratio':row['Success Percentage']}, ignore_index=True)
      


# In[ ]:


df.head(10)


# In[ ]:


df.info()


# In[ ]:


cols = df.columns.drop(['Date','Route'])


# In[ ]:


#Converting object type attributes to Float except for route
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')


# In[ ]:


#Extract Month
df['Month']=pd.to_datetime(df['Date']).dt.month.apply(lambda x: calendar.month_name[x])


# In[ ]:


df['Month_Val']=pd.to_datetime(df['Date']).dt.month


# In[ ]:


#Extract Year
df['Year']=pd.to_datetime(df['Date']).dt.year


# In[ ]:


df.isna().sum()


# In[ ]:


# After merging the Weather Data Set with the Climbing Dataset, it has been observed that there are many records of a day for whic the weather has not been captured
# Imputing an average weather of the month do not make sense, instead i would like to capture the climbing statistics only for those days where the weather information is available
df.dropna(inplace=True)


# In[ ]:


#Climbs by Year
dbRainierByYear=df.groupby(['Year'])["Attempted","Succeeded"].sum().reset_index()


# In[ ]:


dbRainierByYear['Success %']=(dbRainierByYear['Succeeded']/dbRainierByYear['Attempted'])*100


# In[ ]:


dbRainierByYear.head(20)


# In[ ]:


dbRainierByYear.plot(x='Year',y=["Success %"],kind='bar')


# In[ ]:


#Climbs by Month
dbRainierByMonth=df.groupby(['Month'])["Attempted","Succeeded"].sum().reset_index()


# In[ ]:


dbRainierByMonth['Success %']=(dbRainierByMonth['Succeeded']/dbRainierByMonth['Attempted'])*100


# In[ ]:


Sort_Dataframeby_Month(dbRainierByMonth,monthcolumnname='Month')


# In[ ]:


dbRainierByMonth.head(20)


# In[ ]:


Sort_Dataframeby_Month(dbRainierByMonth,monthcolumnname='Month').plot(x='Month',y=['Attempted','Succeeded'],kind='bar')


# ## Analysis
# 1. The Above chart show that most number of the climbing attempts and the successes are during the months, May, June, July , August and September

# In[ ]:


#Climbs by Route
dbRainierByRoute=df.groupby(['Route'])["Attempted","Succeeded"].sum().reset_index()


# In[ ]:


dbRainierByRoute.head(5)


# In[ ]:


dbRainierByRoute.plot(x='Route',y=["Attempted","Succeeded"],kind='bar')


# # Analysis
# 
# The above plot shows that most of the climbs were attempted through the disappointment cleaver route and a few considerable number through the Emmons and Kaultz Galcier

# In[ ]:


sns.pairplot(df, diag_kind='kde') 


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
X = df.iloc[:,0:20]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # Pairplot analysis and correlation
# 
# 1. Clearly there is a correlation between Battery Volate and The Temperature but their correlation does not have an effect on the summit success
# 2. The Number of attempts and success are clearly concentrated within a period of the year , during certain months, than on specific dates.
# 3. Higher the temperature, higher is the solar radiation and so is the number of attempts and successes
# 4. There seems to be a lot of Outliers in the data specifically in the attributed Solar Radiation
# 5. As the temperature has increased, the humidity has decreased creating favourable conditions for climbing.
# 6. As the temperateure has decreased the wind speed has decreased, indicating that during the colder weather the wind speed is usually high which is unfavourable for climbing
# 
# 
# 

# # Imputing variables

# In[ ]:


# removing the attempted, succeeded, Month, Year and Date columns from the dataset so that the target column success can be strictly used for analysis
# Removing Battery Voltage since it is redundant to Tempareture and Wind Direction since it doesnt seem to have a major impact on the overall summit attempts
# Solar Radiation is linear with Temperate - Has many null values and outliers. Hence removing the column

df_model=df.drop(['Attempted','Succeeded','Year','Month','Date', 'Battery Voltage AVG','Wind Direction AVG','Solar Radiation AVG'], axis=1)


# In[ ]:


# Wind Speed Average has 0 values in the data and need to be corrected. Windspeed can never be 0
# Observing the data has given some insights, that Wind Speed has been 0  when the Relative Humidity is above 90 in 98% of the cases
# Calculating the average windspeed when the relative humidity is over 89 but less than 99 gives - 21.17
# Replace the 0 values of the wind speed to 21.17
df_model['Wind Speed Daily AVG']=df_model['Wind Speed Daily AVG'].replace([0], 21.17)


# In[ ]:


df_model.info()


# In[ ]:


df_model.head(10)


# In[ ]:


df_model.isna().sum()


# In[ ]:


sns.pairplot(df_model, diag_kind='hist')


# In[ ]:


df_new=pd.get_dummies(df_model)
#df_new=df_model


# In[ ]:


df_new.info()


# In[ ]:


df_new.head(10)


# In[ ]:


df_new.isnull().sum()


# In[ ]:





# In[ ]:


# Converting the values to Z-Scores since the attributes values are in different ratios and types

df_new_z = df_new.apply(zscore)


# In[ ]:


df_new_z.describe()


# In[ ]:


#Split the data to train and test

y = df_new_z['Success Ratio'].values
X = df_new_z.drop('Success Ratio', axis=1).values


# In[ ]:


y


# In[ ]:


from sklearn import model_selection

test_size = 0.30 # taking 70:30 training and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)


# In[ ]:


y


# In[ ]:


X_train


# # Models

# In[ ]:


# Linear Regression

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression
from sklearn import metrics

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


model.score(X_test, y_test)
# Very bad score.


# In[ ]:


plt.scatter(X_test[:,0], y_test)
plt.plot(X_test[:,0], y_predict, color='red')
plt.show()


# In[ ]:


from sklearn.svm import SVR
model = SVR(gamma='scale', C=1.0, epsilon=0.2)
model.fit(X, y)
y_predict = model.predict(X_test)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


plt.scatter(X_test[:,0], y_test)
plt.plot(X_test[:,0], y_predict, color='red')
plt.show()


# In[ ]:


#Decision Tree and Random Forest
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 3)  

  
# fit the regressor with X and Y data 
regressor.fit(X, y) 


# In[ ]:


df1= pd.DataFrame()
#df1['feature'] = df.drop(['not.fully.paid'], axis=1).columns
df1['feature'] = df_new_z.drop('Success Ratio', axis=1).columns
df1['Importance Index']= regressor.feature_importances_
print(df1.sort_values(by='Importance Index', ascending=False))


# In[ ]:


#Choosing only the top 12 columns from the dataset as per the feature importance notatin above

df_Impftrs=df_new.drop(columns=['Route_Liberty RIngraham Directge','Route_Kautz Cleaver','Route_Ptarmigan RIngraham Directge','Route_Ingraham Direct','Route_Gibralter Ledges','Route_Mowich Face','Route_Success Cleaver','Route_Sunset RIngraham Directge','Route_Tahoma Cleaver',"Route_Fuhrer's Finger",'Route_Gibralter Chute','Route_Curtis RIngraham Directge','Route_Tahoma Glacier','Route_Unknown','Route_Wilson Headwall','Route_Nisqually Glacier'])


# In[ ]:


df_Impftrs.head(10)


# In[ ]:


df_Impftrs_z = df_Impftrs.apply(zscore)


# In[ ]:


#Split the data to train and test

y = df_Impftrs_z['Success Ratio'].values
X = df_Impftrs_z.drop('Success Ratio', axis=1).values


# In[ ]:


from sklearn import model_selection

test_size = 0.30 # taking 70:30 training and test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)


# In[ ]:


#Decision Tree and Random Forest
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  

  
# fit the regressor with X and Y data 
regressor.fit(X, y) 


# In[ ]:


y_predict = regressor.predict(X_test)


# In[ ]:


regressor.score(X_test, y_test)


# In[ ]:


# Bagging - Ensemble

from sklearn.ensemble import BaggingRegressor

bgcl = BaggingRegressor(n_estimators=35)

#bgcl = BaggingClassifier(n_estimators=50)
bgcl = bgcl.fit(X, y)


# In[ ]:


y_predict = bgcl.predict(X_test)

print(bgcl.score(X_test, y_test))


# In[ ]:


# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
gbcl = GradientBoostingRegressor(n_estimators = 50)
gbcl = gbcl.fit(X, y)
y_predict = bgcl.predict(X_test)
print(bgcl.score(X_test, y_test))


# In[ ]:


#RandomForest
from sklearn.ensemble import RandomForestRegressor
rfcl = RandomForestRegressor(n_estimators = 50)
rfcl = rfcl.fit(X, y)
y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))


# # Hyperparameter Tuning of Random Forests

# In[ ]:


print('Parameters currently in use:\n')
print(rfcl.get_params())
print("\n")

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num = 9)] ## play with start and stop

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 20, num = 5)] ## change 10,20 and 2
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,10]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator = rfcl, param_distributions = random_grid, n_iter = 200, cv = 3, 
                               verbose=2, random_state=50, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train,y_train)
print("Best Parameters are:",rf_random.best_params_)


# In[ ]:


best_random = rf_random.best_estimator_
best_random.fit(X_train,y_train)

predictions = best_random.predict(X_test)

print(best_random.score(X_test, y_test))

#from sklearn.metrics import classification_report,confusion_matrix
#print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=700,min_samples_split= 5,max_depth= 50, min_samples_leaf= 5) 


# In[ ]:


regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(regressor.score(X_test, y_test))

