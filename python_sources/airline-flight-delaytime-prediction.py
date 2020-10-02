#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Numerical libraries
import numpy as np   

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score

# to handle data in form of rows and columns 
import pandas as pd    

# importing ploting libraries
import matplotlib.pyplot as plt   

import statsmodels.formula.api as sm

#importing seaborn for statistical plots
import seaborn as sns

import datetime
import time
from time import strftime, gmtime

import statsmodels.formula.api as smf
#maschine learning libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC
from random import sample


# CSV-File Import

# In[ ]:


df_flights=pd.read_csv("/kaggle/input/flight-delays/flights.csv")


# In[ ]:


df_flights.head()


#     **Data Analysis and Preprocessing**
# The following is the first overview of all attributes:

# In[ ]:


df_flights.info()


# There is a need to convert them all to datetime. In addition, it seems to be helpful to write/use a function for this conversion. (Thanks to fabiendaniel and her great tutorial here ):

# In[ ]:


# converting input time value to datetime.
def conv_time(time_val):
    if pd.isnull(time_val):
        return np.nan
    else:
            # replace 24:00 o'clock with 00:00 o'clock:
        if time_val == 2400: time_val = 0
            # creating a 4 digit value out of input value:
        time_val = "{0:04d}".format(int(time_val))
            # creating a time datatype out of input value: 
        time_formatted = datetime.time(int(time_val[0:2]), int(time_val[2:4]))
    return time_formatted


# In[ ]:


df_flights['ARRIVAL_TIME'] = df_flights['ARRIVAL_TIME'].apply(conv_time)
df_flights['DEPARTURE_TIME'] = df_flights['DEPARTURE_TIME'].apply(conv_time)
df_flights['SCHEDULED_DEPARTURE'] = df_flights['SCHEDULED_DEPARTURE'].apply(conv_time)
df_flights['WHEELS_OFF'] = df_flights['WHEELS_OFF'].apply(conv_time)
df_flights['WHEELS_ON'] = df_flights['WHEELS_ON'].apply(conv_time)
df_flights['SCHEDULED_ARRIVAL'] = df_flights['SCHEDULED_ARRIVAL'].apply(conv_time)


# The required data has now the correct format.

#       **Handling the Null Values**
# After I converted the necessary time values to a DateTime datatype, I need to check our data according to its integrity. Null values or missing data are often occurring data states that need to be handled.
# 
# In addition to several other methods, I will focus on two or three methods in this notebook to deal with null value data or missing data.
# 
# One option is to delete the corresponding rows.
# 
# Another case of handling missing or null value data is to reconstruct the missing data according to information from other columns. Imagine there is a start and an end time and only the duration is missing. You could calculate the missing values simply by the difference between end time and start time. Accordingly, you do not have to delete the data column but you can continue to use the information contained in it.
# 
# One of the best ways to handle missing or null value data is the imputation. The imputation will fill the missing gaps with some numbers that are based on existing data columns. The numbers are not as accurate as the real data but fits the needs for the most prediction models and lead to a better resolution of the model.

# In[ ]:


df_flights.isnull().sum()


#     **Reconstruct Data Manually**
# Our null analysis above shows the following features with a lot of null values:
# 
# * CANCELLATION_REASON
# * AIR_SYSTEM_DELAY
# * SECURITY_DELAY
# * AIRLINE_DELAY
# * LATE_AIRCRAFT_DELAY
# * WEATHER_DELAY
# In this case here, I try to determine or "calculate" the data by deriving the situation. Look at the values that are mostly empty according to the coherent afford of an airline to not be the reason for a delay. Therefore the missing data (or Not-a-Number data) is not based on a bad data quality, it is more the fact that it didn't happen any action by these delay features. You can prove it by looking at a tuple of one of that features when there is at least one feature triggered, all the other features are "initialized" with "0.0":

# In[ ]:


df_flights['AIRLINE_DELAY'] = df_flights['AIRLINE_DELAY'].fillna(0)
df_flights['AIR_SYSTEM_DELAY'] = df_flights['AIR_SYSTEM_DELAY'].fillna(0)
df_flights['SECURITY_DELAY'] = df_flights['SECURITY_DELAY'].fillna(0)
df_flights['LATE_AIRCRAFT_DELAY'] = df_flights['LATE_AIRCRAFT_DELAY'].fillna(0)
df_flights['WEATHER_DELAY'] = df_flights['WEATHER_DELAY'].fillna(0)


# So it's ok to transform the NAN-data to the value "0.0" because there was no impact on the flight by these data that causes a delay.
# 
# Null values have now decreased significantly. There are only a few attributes left. Particular striking, however, is the CANCELLATION_REASON that hits the high mark with around 98% null values. We need to take a closer look at the cancellation data.

# **Dealing with Null Values in Categorical Data**
# 

# In[ ]:


df_flights.isnull().sum()


# The reason for cancellation of flights splits into the following occurrences:
# 
#     A - Airline/Carrier
#     B - Weather
#     C - National Air System
#     D - Security
#     ... and has the following ratio:

# In[ ]:


# group by CANCELLATION_REASON to see the ration
df_flights['CANCELLATION_REASON'].value_counts()


# As you can see the main reason for cancelation is B the weather. It is well known that the weather is often the cause of delays and cancelations. In the case of this attribute, we look at the weather as a cancelation reason, not a delay reason. Now there is the following question: If we want to predict delay times from departure flights, is it necessary to include flight cancellation reasons in our calculation? Don't we want to focus only on not canceled flights, on flights with a departure and a (late) arrival time? The answer is: No, we want them all! We don't want to lose data for our prediction. Every information, in this case, is important. For example cancellation reason "Weather" for a canceled flight. The flight themselves did not take place, that's right, but what about the consequences of the canceled flights? All the passengers need to get to their destinations, therefore they will be booked on the next flight or moreover the canceled flight will start in another timeshift and will probably block another's plains flight slot. That all leads to a knock-on effect on other flights.
# 
# "Manuell" Conversion of categories to numeric values Most models don't work pretty good with categorical values. They need to be converted into numeric values to use a prediction model. There is a way to convert all categorical data into numeric values, its called One-Hot Encoding. This approach will line in all categorical values in separate columns, creates a new column and matches every occurrence of the categorical value with 1 or 0 for non-occurrences. You should take a deeper look into it
# 
# Another approach that is similar to the One-Hot encoding approach is included in the Pandas library and is called get_dummies(). This function converts categorical values into dummy/indicator values as well. In this case, all null value data are also converted and filled with 0 values. Nevertheless, in this case and according to the small amount of four categorical values, I will convert the CANCELLATION_REASON manually. The final result will be the following:
# 
# NaN = 0 A = 1 B = 2 C = 3 D = 4

# In[ ]:


# -------------------------------------
# converting categoric value to numeric
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'A', 'CANCELLATION_REASON'] = 1
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'B', 'CANCELLATION_REASON'] = 2
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'C', 'CANCELLATION_REASON'] = 3
df_flights.loc[df_flights['CANCELLATION_REASON'] == 'D', 'CANCELLATION_REASON'] = 4

# -----------------------------------
# converting NaN data to numeric zero
df_flights['CANCELLATION_REASON'] = df_flights['CANCELLATION_REASON'].fillna(0)


# In[ ]:


df_flights.isnull().sum()


# Good to know how to calculate this values, bad is the fact that the values to calculate these times are also NaN - data. That is probably the reason for its initial NaN - data value. I have no choice but to declare the data as outliers.

# In[ ]:


# drop the last 1% of missing data rows.
df_flights = df_flights.dropna(axis=0)


# In[ ]:


df_flights.isnull().sum()


# **Analyzing Distribution after Cleansing, Conversion and Preprocessing**
# **Feature and Label Selection**
# For our prediction, I now need to identify the features that are most likely to impact on the flight delays.
# 
# First I want to include the airports and try to figure out whether there is an impact on the delay regarding the departure airport or not. For this, I will include the airports from another file in this evaluation. With the included information about the location of the airport, I could identify regions on the map that support a delay.
# 
# First, I will include the airlines in the evaluation to get a distribution of the delays per airline. Later I will add the airports and their location data to the evaluation to get a closer view of the map and some location-based delays.
# 
# Merging the Airline Codes (IATA-Codes) I am going to merge the IATA-Airline codes from the other .csv-file.

# In[ ]:


df_airlines = pd.read_csv('/kaggle/input/flightdelay/airlines.csv')
df_airlines


# In[ ]:


# joining airlines
df_flights = df_flights.merge(df_airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')


# In[ ]:


# dropping old column and rename new one
df_flights = df_flights.drop(['AIRLINE_x','IATA_CODE'], axis=1)
df_flights = df_flights.rename(columns={"AIRLINE_y":"AIRLINE"})


# **Analyzing the proportion of flights with respect to the companies.**

# In[ ]:


fig_dim = (14,18)
f, ax = plt.subplots(figsize=fig_dim)
quality=df_flights["AIRLINE"].unique()
size=df_flights["AIRLINE"].value_counts()

plt.pie(size,labels=quality,autopct='%1.0f%%')
plt.show()


#     **Analyzing the Delays by Airline**
# Getting an overview of delays by airlines companies.

# In[ ]:


sns.set(style="whitegrid")

# initialize the figure
fig_dim = (10,12)
f, ax = plt.subplots(figsize=fig_dim)
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot
sns.stripplot(x="ARRIVAL_DELAY", y="AIRLINE",
              data=df_flights, dodge=True, jitter=True
            )
plt.show()


# The distribution above shows the airlines in comparison to their ARRIVAL_DELAYs. It clearly shows that American Airlines has a wide spread of delays. By contrast, the airline with the most entries is Southwest Airlines and their delays look pretty low compared to the American Airlines delays. I will elaborate on this in the following:

# In[ ]:


# Group by airline and sum up / count the values
df_flights_grouped_sum = df_flights.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('sum').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_SUM"})
df_flights_grouped_cnt = df_flights.groupby('AIRLINE', as_index= False)['ARRIVAL_DELAY'].agg('count').rename(columns={"ARRIVAL_DELAY":"ARRIVAL_DELAY_CNT"})

# Merge the two groups together
df_flights_grouped_delay = df_flights_grouped_sum.merge(df_flights_grouped_cnt, left_on='AIRLINE', right_on='AIRLINE', how='inner')
# Calculate the average delay per airline
df_flights_grouped_delay.loc[:,'AVG_DELAY_AIRLINE'] = df_flights_grouped_delay['ARRIVAL_DELAY_SUM'] / df_flights_grouped_delay['ARRIVAL_DELAY_CNT']

df_flights_grouped_delay.sort_values('ARRIVAL_DELAY_SUM', ascending=False)


# In conclusion, Southwest Airlines has a lot of mostly smaller delays which are in total the high mark of delays in this evaluation. On the other side and with a hint on our distribution chart above, American Airlines has a lot of huge delays in single flights which effects the total delay of the airline. They are in the upper thirds of the delays but their mean delay per airline is one of the lowest of all airlines.

#     **Feature Correlation**
# So let us look at the correlation between each of the features ( and the label as well). This might be the first step into a closer feature selection. The main goal is to identify the features that affect the ARRIVAL_DELAY in a positive or negative way.

# In[ ]:


# Dataframe correlation
del_corr = df_flights.corr()

# Draw the figure
f, ax = plt.subplots(figsize=(14, 12))

# Draw the heatmap
sns.heatmap(del_corr,annot=True,cmap='inferno')
plt.show()


#     **Results from Correlation Matrix**
# I am dividing the different correlations into two parts, the positive correlations (higher than 0.6 ) and the less positive correlations (less than 0.6 but higher than 0.2). The results are listed in the list below:
# 
#     Positive correlations between:
#         DEPARTURE_DELAY and
#         ARRIVAL_DELAY
#         LATE_AIRCRAFT_DELAY
#         AIRLINE_DELAY
#         ARRIVAL_DELAY and
#         DEPARTURE_DELAY
#         LATE_AIRCRAFT_DELAY
#         AIRLINE_DELAY
#     Less positive correlations between:
#         ARRIVAL_DELAY and
#         AIR_SYSTEM_DELAY
#         WEATHER_DELAY
#         DEPARTURE_DELAY and
#         AIR_SYSTEM_DELAY
#         WEATHER_DELAY
#         TAXI_OUT and
#         AIR_SYSTEM_DELAY
#         ELAPSED_TIME

# In[ ]:


# Marking the delayed flights
df_flights['DELAYED'] = df_flights.loc[:,'ARRIVAL_DELAY'].values > 0


# In[ ]:


figsize=plt.subplots(figsize=(10,12))
sns.countplot(x='DELAYED',hue='AIRLINE',data=df_flights)
plt.show()


#     **Feature Selection with Machine Learning Algorithms**

# In[ ]:


# Label definition
y = df_flights.DELAYED

# Choosing the predictors
feature_list_s = [
    'LATE_AIRCRAFT_DELAY'
    ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY'
    ,'WEATHER_DELAY'
    ,'ELAPSED_TIME']

# New dataframe based on a small feature list
X_small = df_flights[feature_list_s]


# In[ ]:


# RandomForestClassifier with 10 trees and fitted on the small feature set 
clf = RandomForestClassifier(n_estimators = 10, random_state=32) 
clf.fit(X_small, y)


# In[ ]:


importances=clf.feature_importances_
importances=pd.DataFrame([X_small.columns,importances]).transpose()
importances.columns=[['Variables','Importance']]
importances


# A little bit has changed. Now the AIR_SYSTEM__DELAY has got the most influences on a flight that has been delayed. This feature had a less positive correlation in our correlation resume above. All the other features have remained in the same order of importance as we have found out. Let us try a wider range with the same model. And please keep in mind that we use a classification here. We have classified the data into delayed and not delayed data and want to find out now which of these features effects a delay of a flight the most. There could be and there probably will be different features for a flight that arrives just in time, but this will be part of a later section where we try to determine the actual arrival at an airport.
# 
# In following, I want to proof the above written down feature correlation count with a machine learning algorithms. Do they really correlate as good as I think with the ARRIVAL_DELAY? I will classify the data into delayed and not delayed data and define a label (DELAYED) for that in the dataframe. Afterward I will show the feature importance for the given attributes.
# 
# In the beginning, I need to reduce computation time by reducing the data ,so choose a Random Sample of 50000 records. Otherwise, this whole prediction will execute too long.

# In[ ]:


# choosing the predictors
feature_list = [
    'YEAR'
    ,'MONTH'
    ,'DAY'
    ,'LATE_AIRCRAFT_DELAY'
    ,'AIRLINE_DELAY'
    ,'AIR_SYSTEM_DELAY'
    ,'WEATHER_DELAY'
    ,'ELAPSED_TIME'
    ,'DEPARTURE_DELAY'
    ,'SCHEDULED_TIME'
    ,'AIR_TIME'
    ,'DISTANCE'
    ,'TAXI_IN'
    ,'TAXI_OUT'
    ,'DAY_OF_WEEK'
    ,'SECURITY_DELAY'
]
# Any number can be used in place of '0'. 
import random
random.seed(0)
    
df_flights_1=df_flights.sample(n=50000)
X = df_flights_1[feature_list]


# In[ ]:


X.info()


# In[ ]:


y = df_flights_1.DELAYED


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import scale
X_train=scale(X_train)
X_test=scale(X_test)


# In[ ]:


model=LinearRegression()
model=model.fit(X_train,y_train)
slope=model.coef_
coef=model.intercept_
print(slope.flatten())
print(coef)


# In[ ]:


y_pred=model.predict(X_train)


# In[ ]:


r2_score(y_train,y_pred)


# Stepwise Regression  is where we select the features to be used in a model on the basis of variable importance.
# 
# Variables are sequentially selected and a model is built starting with one variable. In the next model another variable is added and the adjusted R2 for both the models are compared. If the adjusted R2 increases the next variable is added. This process is repeated till there is a decrease in the adjusted R2 The above process is known as forward selection.
# 
# For this purpose we can use the mlextend library

# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression


# In[ ]:


lr = LinearRegression()
sfs = SFS(lr, k_features='best', forward=True, floating=False, 
          scoring='neg_mean_squared_error', cv=10)
model = sfs.fit(X_train, y_train)

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[ ]:


print('Selected features:', sfs.k_feature_idx_)


# Similarly backward elimination can also be performed on the same data

# In[ ]:


lr = LinearRegression()
sfs2 = SFS(lr, k_features='best', forward=False, floating=False, 
          scoring='neg_mean_squared_error', cv=10)
model = sfs2.fit(X_train, y_train)

fig = plot_sfs(sfs2.get_metric_dict(), kind='std_err')

plt.title('Backward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[ ]:


print('Selected features:', sfs2.k_feature_idx_)


# We can see almost all the features are importants, as they are interlinked.
# 
# We then applied the train on all the models to check which model is giving us the best accuracy.

# In[ ]:


from sklearn import ensemble,gaussian_process,linear_model,naive_bayes,neighbors,svm,tree


# In[ ]:


MLA = [
    #Ensemble Methods
    ensemble.AdaBoostRegressor(),
    ensemble.BaggingRegressor(),
    ensemble.ExtraTreesRegressor(),
    ensemble.GradientBoostingRegressor(),
    ensemble.RandomForestRegressor(),
    #Nearest Neighbor
    neighbors.KNeighborsRegressor(),
    #Trees    
    tree.DecisionTreeRegressor(),
    tree.ExtraTreeRegressor()
    ]


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score,precision_score,recall_score,auc


# In[ ]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)
results=[]

row_index = 0
for alg in MLA:
    
    cv_results = cross_val_score(alg, X_train, y_train, cv=10)
    results.append(cv_results)
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
    
    
    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
MLA_compare


# We can see that Bagging Regressor and Random Forest Regressor models are giving the accuracy. So we choose Random Forest Model to predict the tune is more.

# In[ ]:


plt.subplots(figsize=(15,6))
sns.lineplot(x="MLA Name", y="MLA Train Accuracy",data=MLA_compare,palette='hot',label='Train Accuracy')
sns.lineplot(x="MLA Name", y="MLA Test Accuracy",data=MLA_compare,palette='hot',label='Test Accuracy')
plt.xticks(rotation=90)
plt.title('MLA Accuracy Comparison')
plt.legend()
plt.show()


# The Above plot shows the test train accuracy with respect to each models.

# In[ ]:


plt.subplots(figsize=(15,6))
sns.lineplot(x="MLA Name", y="MLA AUC",data=MLA_compare,palette='hot',label='Accuracy')

plt.xticks(rotation=90)
plt.title('MLA Accuracy Comparison')
plt.legend()
plt.show()


# The Above plot shows the Accuracy score of each model, where clearly Random Forest and Bagging Models are giving the best accuracy.

# In[ ]:


#sns.boxplot(MLA_compare["MLA AUC"])# boxplot algorithm comparison
fig = plt.figure(figsize=(10,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results,labels=MLA_compare['MLA Name'])
plt.xticks(rotation=45)
plt.show()


# So this is basiccally the Test accuracy result of each models, where we can see Random Forest is giving us the best results,whose data is not varying much as well as it is not having any outliyers like Extra Tree ,Extra Trees and Ada Boost models.
# 
# We can see, Bagging Regressor and Randon Forrest Regressor are the two models,which gives approximately same accuracy and we can see, if we can imporve those accuracy.

#     **Data Prediction**
#     Preparing the Prediction
#     Building the Model First
#     I am building the model first. Here I am choosing 100 trees for the model to not overexert the computation time in later purpose.

# In[ ]:


# RandomForest with 100 trees
forest_model = RandomForestRegressor(n_estimators = 100, random_state=42)


#     **Choosing the Prediction Target**
# This time I choose the ARRIVAL_DELAY as the target and change the model to the Random Forest Regressor (seen above) to predict the exact minutes delayed or arrived in time.

# In[ ]:


y = df_flights_1.ARRIVAL_DELAY
y = np.array(y)


#     **Choosing the Predictors**
# To predict our prediction target (ARRIVAL_DELAY), we need some features. I will select the same features as in the chapter before.

# In[ ]:


X = np.array(X)


#     **Separating into Test and Train Datasets**
# It is necessary to separate the data into train and test dataset.

# In[ ]:


# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.30, random_state = 42)


# In[ ]:


#The Shape of Train- and Testdata
print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:', val_X.shape)
print('Testing Labels Shape:', val_y.shape)


#     **Model Training and Prediction**
# Establish Baseline

# In[ ]:


# Average arrival delay for our dataset
baseline_preds = df_flights['ARRIVAL_DELAY'].agg('sum') / df_flights['ARRIVAL_DELAY'].agg('count') 

# Baseline error by average arrival delay 
baseline_errors = abs(baseline_preds - val_y)
print('Average baseline error: ', round(np.mean(baseline_errors),2))


# This is our average baseline error of 21.39 minutes of delays we want to beat with our regression model.

# **Train Model**

# In[ ]:


# Fit the model
forest_model.fit(train_X, train_y)


# **Predict and Validate the Result**

# In[ ]:


# Predict the target based on testdata 
flightdelay_pred= forest_model.predict(val_X)


# In[ ]:


#Calculate the absolute errors
errors_random1 = abs(flightdelay_pred - val_y)


# **Return the Absolute Error**

# In[ ]:


print('Mean Absolute Error: ', round(np.mean(errors_random1),3), 'minutes.')


# This looks quite after overfitting. The mean absolute error is pretty small which means the model predicts the arrival delay nearly accurate or over accurate. I will validate and visualize the model in the next chapter.

#     **Validate and Visualize the Model**
# In this chapter, I will validate and visualize the prediction model. The previous mentioned mean absolute error of 0.857 minute seems to be a quite good prediction of the arrival delay. The predictions are on average around 0.857 minutes away from the real value. This is a really exact prediction. It is mandatory to check the model whether it is an overfitted one or not.
# 
# The previously shown feature importance of the model looks like this:

# In[ ]:


X=pd.DataFrame(X)


# In[ ]:


importances=forest_model.feature_importances_
importances=pd.DataFrame([X.columns,importances]).transpose()
importances.columns=[['Variables','Importance']]
importances


# Nearly 88% of the feature importance is based by the DEPARTURE_DELAY feature which is a lot. The next eight features not even have individually more than 10% of importance, they are all lower. AIR_SYSTEM_DELAY, SCHEDULED_TIME and ELAPSED_TIME have at least values that are greater than 1.0%. The remaining features all having an importance that is lower 1.0%.
# 
# I will take a closer look into the single features in the next section.

# In[ ]:


# Count of DEPARTURE_DELAYs that are not zero and could influence our prediction.
print("DEPARTURE_DELAY count: ")
print(df_flights_1[df_flights_1['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count())
print("-------------------------------")
print("All datarow count:")
print((df_flights_1)['DEPARTURE_DELAY'].count())
print("-------------------------------")
print("-------------------------------")
print("Percentag of DEPARTURE_DELAY that is not zero:")
print(df_flights_1[df_flights_1['DEPARTURE_DELAY'] != 0]['DEPARTURE_DELAY'].count() / df_flights_1['DEPARTURE_DELAY'].count())


# Nearly 94% of the values from DEPARTURE_DELAY are set with a value that is not zero. The nearly 100% fulfillment and the effects from the DEPARTURE_DELAY on the ARRIVAL_DELAY leads to that feature importance for the built model. So it seems to be not unusual to have such an accuracy in that case. Still, this seems too accurate, we are talking here about a minute difference to the real arrival delay of a flight. There is a need to check the accuracy of the model in a much better way.
# 
# In the next chapter, I will analyze the coefficient of determination to get a better overview of how good the model fits the dataset.

#     **The Coefficient of Determination - The Model Fitness**
# In this chapter, I will calculate the coefficient of determination or "R-squared" for the model. It will show how good the inputs fit the output of the model, or how good the model represents the underlying data. That means if the regressions of our features have an R-squared close to 1, it means that the independent variables (the features) are well-suited to predict the dependent variable (our target, the ARRIVAL_DELAY).
# 
# I will now calculate the R-squared for the built model based on the training and test dataset:

# In[ ]:


print("----------------- TRAINING ------------------------")
print("r-squared score: ",forest_model.score(train_X, train_y))
print("------------------- TEST --------------------------")
print("r-squared score: ", forest_model.score(val_X, val_y))


# This here seems to be as well pretty accurate. The training dataset is a known dataset by the model why the test dataset is used as well here. As we know due to the previous analysis, the model is highly based on the DEPARTURE_DELAY feature. All the model's decision is based on what the DEPARTURE_DELAY does, which afterward leads to that accuracy.
# 
# I will test the model with another new dataset and calculate the necessary key figures.

#     **Test with Unknown Data Again**
# I will use data from February now, to test the model against total new, unknown data. After all the necessary model preparations I will print out the Mean Absolute Error as well as the r-squared score of the new test data.

# In[ ]:


random.seed(1)
df_flights__2=df_flights.sample(n=50000)
X2 = df_flights__2[feature_list]
y2 = df_flights__2.ARRIVAL_DELAY


# In[ ]:


# Predict the new data based on the old model (forest_model)
flightdelay_pred_ = forest_model.predict(X2)

#Calculate the absolute errors
errors_random_2 = abs(flightdelay_pred_ - y2)


# In[ ]:


# Mean Absolute Error im comparison
print('Mean Absolute Error Random Sample 1: ', round(np.mean(errors_random1),3), 'minutes.')
print('---------------------------------------------------------------')
print('Mean Absolute Error Random Sample 1: ', round(np.mean(errors_random_2),3), 'minutes.')


# The difference between the two datasets (Random Sample 1 and Random Sample 2 ) is not that big, it's even very small. The model even fits on total new data. What about the R-squared calculation?

# In[ ]:


print("r-squared score Random Sample 1: ",forest_model.score(val_X, val_y))
print("------------------- TEST --------------------------")
print("r-squared score Random Sample 2: ", forest_model.score(X2, y2))


# As I already mentioned, the mean absolute error, as well as the r-squared equation both look that the model would not fit that well, because they seem too accurate. The model is highly based on the DEPARTURE_DELAY feature and makes its decisions by that. If there is a flight that has been delayed but not according to the DEPARTURE_DELAY, the model would probably don't give a prediction that has that accuracy.
# 
# I will test this in the following.

#     **Model Check without DEPARTURE_DELAY Impact**
# For this test, I will search for a special flight that is not delayed by the DEPARTURE_DELAY and is at least a delayed flight of 60 minutes (ARRIVAL_DELAY > 60).

# In[ ]:


# Searching for a flight that fits our needs
a=df_flights__2[(df_flights__2.loc[:,'DEPARTURE_DELAY'] < 0) & (df_flights__2.loc[:,'ARRIVAL_DELAY'] > 60)].head(10)


# These delayed flights seems to be a good one. It has the following properties:

# In[ ]:


a


# In[ ]:


# Look into the flight with Arrival Delay but no Departure Delay
a.iloc[1]


# We clearly see that the DEPARTURE_DELAY is not the reason for the delay this time, moreover the airplane departed early than scheduled. So let's use this flight for the model check. Preparations in the following step:

# In[ ]:


# Retrieving the flight with index 3221210 (delayed flight without departure delay).
X3 = a.loc[:,feature_list]
X3 = X3.iloc[0]
# Setting the target for our flight index 3221210
y3 =a.iloc[0]['ARRIVAL_DELAY']
print(y3)
X3


#     **Flight Delay Prediction without DEPARTURE_DELAY**
# Next step will be the prediction and the validation of the result. Therefore I will use the already trained model and give them the information from the special flight above.

# In[ ]:


# Printing the important stuff
flight_pred_s = forest_model.predict([X3])
print("Predicted Delay of the Flight (Minutes): ", flight_pred_s)
print("-------------------------------------------------")
print("Original Delay of the Flight (Minutes):  ", y3)
print("_________________________________________________")
print("_________________________________________________")
print("Difference (Minutes)                   : ",  y3-flight_pred_s)


#     **Conclusion:**
# The gap between the predicted and the original delay is 0.26 minutes. Here we can see how the model behavior changes according to the missing main feature impact (the DEPARTURE_DELAY). The original delay is much lower than the mean absolute error of 2.118 minutes from the previous calculations. The conjecture about the risk of one high rated feature has confirmed. Nevertheless, this difference is in a range that has not be bad at all. It seems this model has a good accuracy to predict the flight delay.
# 
# Some kind of pruning for the DEPARTURE_DELAY would definitely improve the model more. I will keep that in mind for a later version of this model, this notebook, right now I'm happy with the result of the model and will leave it as it is.

#     **Insights:**
# We can see that departure delay is the main problem which is creating Delay in the aviation industry. Departure Delay can be caused by many circumstances, that is airline haven't reached the origin,its still on its way from other location,because of some weather delay,crew delay. These delays are basically causing the flight to be on air for more time,thus more fuel is being consumed.feul Consumption negatively effects the revene of the airline company. So in order to increse there revenue, we can think of reducing the delay such that fue consumption is reduced.
# 
# Another important insight is customer satisfaction, people flight boarders generally get irritated if flights are delayed for long hours, so complimentary foods should be given in order no to churn the customers.

#     **Thank You!**

# In[ ]:




