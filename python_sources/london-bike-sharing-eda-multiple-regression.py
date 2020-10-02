#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# This notebook is based on the London Bike Sharing dataset. 
# 
# There are various weather, season, and time related data associated with bike sharing counts and we are looking to discover a pattern that can be used for prediction. This is useful because given a set of inputs, a bike sharing business can predict future demand and determine the necessary inventory levels to sustain that demand. This information can also be used to derive revenue and expense forecasts which is useful for business planning and forecasting. 

# # Setup Environment
# 
# We need to import that various packages we will be using for preparing the dataframe, creating visualizations, and creating a multiple linear regression model. 

# In[ ]:


# importing packages
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
import pandas as pd

# import ML packages
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # Loading Data
# 
# First we import the dataset from the csv file into our DataFrame and inspect the data. 

# In[ ]:


# save filepath to variable for easier access
bike_file_path = '../input/london-bike-sharing-dataset/london_merged.csv' 

# read the data and store data in DataFrame titled bike_data
bike_data = pd.read_csv(bike_file_path)

# inspect data
bike_data.head()


# In[ ]:


# inspect variables
bike_data.info()


# The dataset has 10 initial columns and 17,414 records. Most columns are of floats, including several categorical variables interestingly, and there is a timestamp object. 

# In[ ]:


# check the sum of null records
bike_data.isnull().sum()


# No null exists in the data, contributing to the high usability score in Kaggle. No further cleanup of the dataset is required, as there are no rows to drop or null values to fill in. 

# # Exploratory Data Analysis
# 
# First we want to view a description of the Metadata as provided by the London Bike Sharing Data Set:
# 
# *timestamp* - timestamp field for grouping the data  
# *cnt* - the count of new bike shares  
# *t1* - real temperature in C  
# *t2* - temperature in C "feels like"  
# *hum* - humidity in percentage  
# *windspeed* - wind speed in km/h  
# *weathercode* - category of the weather  
# *isholiday* - boolean field - 1 holiday / 0 non holiday  
# *isweekend* - boolean field - 1 if the day is weekend  
# *season* - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.  
# 
# *weather_code* category description:  
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 2 = scattered clouds / few clouds 3 = Broken clouds 4 = Cloudy 7 = Rain/ light Rain shower/ Light rain 10 = rain with thunderstorm 26 = snowfall 94 = Freezing Fog
# 
# <br/>
# 
# Let's take a look at the distribution of the cnt variable which will be our dependent variable we will be using in our model. We will also take a look at the description of the data which contains several standard statistical measures. 

# In[ ]:


# plot distribution of cnt target variable
sns.distplot(bike_data['cnt'])
plt.show()


# The cnt is not normally distributed and contains a positive skew.

# In[ ]:


# inspect description of variables
bike_data.describe()


# From the **cnt** we can again confirm that there are no missing values for any of the variables. We can also see the **mean**, **std** (standard deviation), **min** (mininum), **max** (maximum), and the various quartiles (**25%**, **50%**, **75%**). 

# Let's take a look at the pearson correlation coefficients. This helps us understand the extent to which two variables are correlated. We will be able to see both the strength of the correlation as well as the direction and use that to make a decision on the exclusion of predictive variables that display multicollinearity.

# In[ ]:


# create correlation matrix displaying pearson correlation coefficients for all variables
corr_matrix = bike_data.corr()
corr_matrix


# We can also visually examine the relationships between the measurable variables via a scatterplot using a randomly selected sample size of 1,000. We also see the histograms for each of the measurable variables displayed across the diagonal. 

# In[ ]:


# plot pair grid with histograms and scatterplots
bike_data_sample = bike_data.sample(1000)
p = sns.PairGrid(data=bike_data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'cnt'])
p.map_diag(plt.hist)
p.map_offdiag(plt.scatter)


# We can observe a roughly normal distribution for **t1**, **t2**, **hum**, and **wind_speed**. Collinearity is observed between **t1** (real temperature in C) and **t2** (temperature in C "feels like") which is not suprising given that **t2** uses **t1** as a starting point and is potentially modified by other conditions such as **wind_speed** and **humidity** which can be observed in the scatterplots. A decision will need to be made as to which should be used when we create the linear regression model.
# 
# <br/><br/>
# 
# We also examined histograms and scatterplots comparing cnt, t1, hum, and wind_speed, color coded by season. A couple of observations that stand out are a moderate negative correlation between t1 and hum and a mild negative correlation between t1 and wind_speed.

# In[ ]:


# plot pair grid with histograms and scatterplots using season as hue
bike_data_sample = bike_data.sample(1000)
p = sns.PairGrid(data=bike_data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'cnt'], hue='season')
p.map_diag(plt.hist)
p.map_offdiag(plt.scatter)
plt.legend(title='Season', loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1, labels=['Spring', 'Summer', 'Fall', 'Winter'])


# # Adding Time and Date Variables
# 
# The **timestamp** data is not very useful to us in it's current format. In order to incorporate it into our linear regression model, we will need to extract the hour, day, and month from the **timestamp**.

# In[ ]:


# convert timestamp string to datetime format for entire timestamp column
bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp']) 
# retrieving timestamp column by iloc method
type(bike_data['timestamp'].iloc[0]) 

# create hour, month, and day of week variables from timestamp data
bike_data['hour']=bike_data['timestamp'].apply(lambda time: time.hour) 
bike_data['month']=bike_data['timestamp'].apply(lambda time: time.month)
bike_data['day_of_week']=bike_data['timestamp'].apply(lambda time: time.dayofweek)

# creating mapping variable for day of week labels
date_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'} 
bike_data['day_of_week'] = bike_data['day_of_week'].map(date_names)

bike_data.head()


# We can now look for patterns in the hour, month, and day of week data and see if there are periods of higher usage. A hypothesis I will be testing is the assumption that the **cnt** will be higher during daylight hours, weekdays, and summer months. 
# 
# We will now visualize the **Hour**, **Month**, and **Day of Week** data using boxplot graphs.

# In[ ]:


# create box plots for time related variables
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(24, 8)

sns.boxplot(data=bike_data, x='month', y='cnt', ax=ax1)
sns.boxplot(data=bike_data, x='hour', y='cnt', ax=ax2)
sns.boxplot(data=bike_data, x='day_of_week', y='cnt', ax=ax3)


# Reviewing the month box plot, the peak is during the summer, particularly in July with the months of April through October (when the weather is presumably warmer) having higher counts than the rest of the year. The hour box plot demonstrates the highest traffic during commuter hours in the morning (7am-9am) and afternoon (4pm-7pm), and a steep decrease during the night when its dark.  The Day of Week boxplot shows a modest decrease on Saturday and Sunday due to the weekend. 
# 
# <br/>
# 
# Next we will visualize the effect of **is_weekend** and **is_holiday**, **season**, and **weather_code** on the **cnt** by **Hour** using a point plot. A hue will be used to see the effect of these variables.

# In[ ]:


# create point plot comparing cnt by hour for is_holiday variable
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18,5)
sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='is_holiday', ax=ax1, palette='YlGnBu')


# In[ ]:


# create point plot comparing cnt by hour for is_weekend variable
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18,5)
sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='is_weekend', ax=ax1, palette='YlGnBu')


# As you can see, both the weekend and holiday variable have a similar effect. One thing to note is that a non holiday can still also be a weekend, which is why you see a slightly weaker contrast on the is_holiday graph.

# In[ ]:


# creating mapping variable for season labels
season_names = {0:'Spring',1:'Summer',2:'Fall',3:'Winter'} 
bike_data['season'] = bike_data['season'].map(season_names) 

# create point plot comparing cnt by hour for season variable
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18,5)
sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='season', ax=ax1, palette='YlGnBu')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# As expected, the season has an effect on count but mainly from 8am until midnight, with the biggest effects found in the afternoon during what would be considered peak commute times. 

# In[ ]:


# creating mapping variable for weather labels
weather_names = {1:'Clear',2:'Scattered Clouds',3:'Broken Clouds',4:'Cloudy',7:'Light Rain',10:'Thunderstorm',26:'Snowing',94:'Freezing Fog'}
bike_data['weather_code'] = bike_data['weather_code'].map(weather_names)

# create point plot comparing cnt by hour for weather variable
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18,5)
sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='weather_code',ax=ax1, palette='YlGnBu')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)


# Weather, however, had a much more pronounced difference than season, with severe weather conditions drastically decreasing ridership as one would expect. 

# # Preparing Variables for Regression Model
# 
# First we will prepare the data and inspect it to quickly verify things look good. This step is needed because values for season and weather_code have been rewritten as categorical names for use as labels within visualizations for the exploratory data analysis to be more useful. We will start fresh here by pulling in the raw data again and starting fresh.

# In[ ]:


# reset data to prepare for building regression model
bike_data = pd.read_csv(bike_file_path)

# convert float variables to int
bike_data.weather_code = bike_data.weather_code.astype(int)
bike_data.is_holiday = bike_data.is_holiday.astype(int)
bike_data.is_weekend = bike_data.is_weekend.astype(int)
bike_data.season = bike_data.season.astype(int)

# convert timestamp string to datetime format for entire timestamp column
bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp']) 

# retrieving timestamp column by iloc method
type(bike_data['timestamp'].iloc[0]) 

# create new variables from timestamp data
bike_data['hour']=bike_data['timestamp'].apply(lambda time: time.hour) 
bike_data['month']=bike_data['timestamp'].apply(lambda time: time.month)

# inspect data
bike_data.head()


# We have several categorical variables and will need to transform them into dummy variables with binary values in order to incorporate them into our model. This is done because despite having int values, they are not ordinal variables. In order to avoid multicollinearity we will also have to drop one of the dummy variables from each set. For example, if we create four season variables season_1, season_2, season_3, season_4, the first variable will be dropped. In our regression, it will be assumed that if variables for seasons 2 through 4 have values of 0, then our missing dummy variable of season_1 is being represented. The same will apply to weather_code, hour, and month.

# In[ ]:


# create binary dummy variables from categorical variables and drop first column to avoid multicollinearity
bike_data = pd.get_dummies(bike_data, columns = ['weather_code', 'season','hour','month'],drop_first = True)

# drop timestamp
bike_data.drop('timestamp', axis=1, inplace=True)

# inspect bike_data df with added dummy variables
bike_data.head(5)


# After creating the dummy variables and dropping the timestamp variables that is no longer needed, we want to inspect the new full list of variables.

# In[ ]:


# inspect variables
bike_data.info()


# Let's examine the pearson correlation coefficients again given that we have added a significant amount of variables. We will be dropping any that display multicolinearity. We have segtt a correlation coefficient threshold and will automatically calculate the coefficients and drop from the DataFrame any variables that exceed our threshold. 

# In[ ]:


# set limit for correlation coefficient
drop_corr = .95

# select only upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# drop any variables with correlation coefficient greater than drop_corr value
to_drop = [column for column in upper.columns if any(upper[column] > drop_corr)]
bike_data = bike_data.drop(to_drop,axis=1)
print("Dropping: " + str(to_drop) + " variable(s) for exceeding correlation of " + str(drop_corr))
      
# display remaining variables represented as dataframe columns
bike_data.columns


# # Splitting Data into Training and Test Sets
# 
# We will be creating a multiple regression model which is a supervised machine learning parametric method. To do that we are going to split our data into a training set (60%) and a test set(40%). We will be fiting the multiple linear regression to the variables and data from the training set and then testing the performance of the model against the test set.

# In[ ]:


# set the target variable
y = bike_data['cnt']

# set the independent predictor variables
X = bike_data.drop('cnt', axis=1)

# split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
X_train = sm.add_constant(X_train)
X_train.head()


# # Creating the Multiple Regression and Evaluating the Model
# 
# There are many methods for selecting predictors in the process of creating a multiple regeression model.
# 
# * Backward Selection
# * Forward Selection
# * Stepwsie Regression
# * Best Subsets
# 
# 
# To start we will create a multiple regression using all of the variables to get a baseline. 
# 
# We will be using backward selection where we start with all of the predictor variables in the model, and after evaluating the variables, remove less useful predictors one at a time or in batches for categorical variables. We will take a look at the strength of the model (adj r2) and the statistical significance (P-value) of the indepenent variables and determine how to proceed in revising the model. 
# 
# R Squared, also known as the coefficient of determination, displays the variation in the depdendent target variable y as explained by the independent predictor x variables. In other words, the percentage of the prediction outcome that can be attributed to the predictor variables of the model. Another way to look at it is R2 = explained variation / total variation. The higher the number the better the data fits the model in question. 
# 
# Adjusted R Squared is used when multiple independent predictor variables exist and includes a penalty for adding additional predictors. This allows you to compare the effectiveness of different models with differing numbers of predictor variables. This is required because as you add predictors, R2 will always continue to increase, even if there is just a chance correlation between variables. In general a parsimonious model is preferred given that it meets reasonable Adjusted R Squared and statistical significance criteria. 
# 
# P-Values measure the statistical significance of each variable in the model within the context of all variables in the model. It is essentially a measure of the liklihood of achieving results as extreme as were observed given a null hypothesis. In other words, the likeliness of the results being explained by random chance. A very low P-value is desired with .05 or .01 often being used as the standard depending on the context and several factors.
# 
# We begin by creating our first model.

# In[ ]:


# fit data to linear regression
mlr1 = sm.OLS(y_train, X_train).fit()

# view OLS regression results
print(mlr1.summary())


# **Model 1**
# 
# For our first model, we have already eliminated **t2** due to multicollinearity at an earlier step. Here we take a look at the strength of the model (adj r2) and the statistical significance (P-value) of the indepenent variables.
# 
# * Adj R2: 0.713
# 
# There are many variables that have an unacceptably high P-value, particularly several of the **month** variables. Given that they are binary categorical dummy variables, it doesn't make much sense to keep some of them and drop only the problematic ones.  For our second model we will be dropping all of the month variables. 

# In[ ]:


X_train2 = X_train.drop(['month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12'], axis=1)
mlr2 = sm.OLS(y_train, X_train2).fit()
print(mlr2.summary())


# **Model 2**
# 
# For our second model, we see that the adjusted R2 has only decreased  slightly after dropping the month variables.
# 
# * Adj R2: 0.712
# 
# There are still several variables that have an unacceptably high P-value, particularly the **season** variables. We will be dropping all of the season variables as well for our third model and observing the change.  

# In[ ]:


X_train3 = X_train2.drop(['season_1','season_2','season_3'], axis=1)
mlr3 = sm.OLS(y_train, X_train3).fit()
print(mlr3.summary())


# **Model 3**
# 
# For our third model, we see that the adjusted R2 has stayed the same after dropping the season variables.
# 
# * Adj R2: 0.712
# 
# The **weather_code_3** variable has a P-value over the .05 threshold, so we will remove all of the **weather_code** variables to see how our model is affected.

# In[ ]:


X_train4 = X_train3.drop(['weather_code_2','weather_code_3','weather_code_4','weather_code_7','weather_code_10','weather_code_26'], axis=1)
mlr4 = sm.OLS(y_train, X_train4).fit()
print(mlr4.summary())


# **Model 4**
# 
# For our fourth, we see that the adjusted R2 has only decreased slightly after dropping the weather variables. We can see that all remaining p-values are below 0.05 and thus we have satisfied our desired criteria. We will keep this as our final model for making predictions. 
# 
# * Adj R2: 0.706

# # Variance Inflation Factors (VIF)
# 
# Variance inflation factors range from a value of 1.0 and upwards. The VIF helps you quantify the severity of multicollinearity in an OLS regression. The VIF value tells you how much larger the standard error increases compared to if that variable had 0 correlation to other independent predictor variables in your model. 

# In[ ]:


# create dataframe to calculate and display VIF for each variable
vif = pd.DataFrame()
vif['Features'] = X_train4.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# All of the variables have relatively low VIF values with the exception of t1 which is still below our cuttoff of 5.0 which is acceptable. 

# # Making Predictions With Final Model
# 
# We are going to create our final model for prediction using the final set of variables that we had in our final training model. 

# In[ ]:


# drop predictive variables to prepare final model
X_test = sm.add_constant(X_test)
X_test_1 = X_test[X_train4.columns] 

# fit data to linear regression for final model using test data
mlr_test = sm.OLS(y_test, X_test_1).fit()

# inspect X_test data
X_test_1.head()


# Now we will make predictions of the y variable which is **cnt** using our final rest model.

# In[ ]:


# Making predictions using the final model
y_pred = mlr_test.predict(X_test_1)


# # Evaluating Predictions of Final Model

# To evaluate our final model we will look a number of data points and visualizations. First we start off with a distribution plot of predicted y values subtracted from the actual y values from the test dataset. This will help us visualize the distribution of errors. We also take a look at a scatter plot of predicted y values vs actuals from the test data set. 

# In[ ]:


# distribution plot of predicted y values vs test y values
sns.distplot((y_test - y_pred), bins=50);


# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)   
plt.xlabel('y_test ', fontsize=18)                       
plt.ylabel('y_pred', fontsize=16) 


# Lastly, we will look at the R Squared (R2), Adjusted R Squared (Adj R2), and Root Mean Square Error (RMSE) values for the predictions. 
# 
# The RMSE is the standard deviation of the prediction errors also known as residuals. This helps us understand how well the actual data fits our model. 

# In[ ]:


# create R2 score
r2 = r2_score(y_test, y_pred)

# create Adjusted R2 Score
p = len(X_test_1.columns)
n = y_test.shape[0]
adj_r2 = 1 - (1 - r2) * ((n - 1)/(n-p-1))

# create RMSE score
rmse = mean_squared_error(y_test, y_pred, squared = False)


# print final model performance stats
print(str(p) + " Predictors in Test Set")
print(str(n) + " Records in Test Set")
print("R2: " + str(r2))
print("Adj R2: " + str(adj_r2))
print("RMSE: " + str(rmse))


# Let's compare the R2 and Adj R2 values from the training dataset vs the test dataset.
# 
# **Training Dataset**
# * R2 - 0.707
# * Adj R2 - 0.706
# 
# **Test Dataset**
# * R2 - 0.717
# * Adj R2 - 0.715

# # Final Model Conclusion
# 
# **Here is the equation for the final model:**
# 
# Count = 1228.6733 + (41.1958 * t1) + (-14.9461 * hum) + (-10.3165 * wind_speed) + (-310.1950 * is_holiday) + (-227.6981 * is_weekend) + (-80.2197 * hour_1) + (-121.5762 * hour_2) + (-151.5846 * hour_3) + (-151.7127 * hour_4) + (-110.8856 * hour_5) + (233.1219 * hour_6) + (1208.3575 * hour_7) + (2438.8380 * hour_8) + (1246.4129 * hour_9) + (574.6209 * hour_10) + (628.7589 * hour_11) + (862.8595 * hour_12) + (841.0589 * hour_13) + (833.7588 * hour_14) + (900.3832 * hour_15) + (1258.4986 * hour_16) + (2199.0846 * hour_17) + (2110.0713 * hour_18) + (1123.3729 * hour_19) + (593.3988 * hour_20) + (316.1232 * hour_21) + (218.6951 * hour_22) + (94.2308 * hour_23)
# 
# Let's make a final future prediction using the model with the following values:
# 
# * t1: 18.0
# * hum: 70.0
# * wind_speed: 6.0
# * is_holiday: 0
# * is_weekend: 0
# * hour_1: 0
# * hour_2: 0
# * hour_3: 0
# * hour_4: 0
# * hour_5: 0
# * hour_6: 0
# * hour_7: 0
# * hour_8: 0
# * hour_9: 0
# * hour_10: 1
# * hour_11: 0
# * hour_12: 0
# * hour_13: 0
# * hour_14: 0
# * hour_15: 0
# * hour_16: 0
# * hour_17: 0
# * hour_18: 0
# * hour_19: 0
# * hour_20: 0
# * hour_21: 0
# * hour_22: 0
# * hour_23: 0
# 
# We plug in the above values into the formula:
# 
# Count = 1228.6733 + (41.1958 * 18.0) + (-14.9461 * 70.0) + (-10.3165 * 6.0) + (-310.1950 * 0) + (-227.6981 * 0) + (-80.2197 * 0) + (-121.5762 * 0) + (-151.5846 * 0) + (-151.7127 * 0) + (-110.8856 * 0) + (233.1219 * 0) + (1208.3575 * 0) + (2438.8380 * 0) + (1246.4129 * 0) + (574.6209 * 1) + (628.7589 * 0) + (862.8595 * 0) + (841.0589 * 0) + (833.7588 * 0) + (900.3832 * 0) + (1258.4986 * 0) + (2199.0846 * 0) + (2110.0713 * 0) + (1123.3729 * 0) + (593.3988 * 0) + (316.1232 * 0) + (218.6951 * 0) + (94.2308 * 0)
# 
# Count = 1228.6733 + (41.1958 * 18.0) + (-14.9461 * 70.0) + (-10.3165 * 6.0) + (574.6209 * 1)
# 
# Count = 1228.6733 + 741.5244 -1046.227 -61.899 + 574.6209
# 
# **Count = 1,436.6926**
# 
# 
# # Thank you!
# 
# Thank you for reading and please leave a comment below if you have a question or suggestion for improvement!
# 
