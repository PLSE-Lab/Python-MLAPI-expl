#!/usr/bin/env python
# coding: utf-8

# # What is the relationship between pre-existing health conditions and COVID-19?
# 
# **Considering:** Number of positive test cases and number of deaths associated with each condition
# 
# **Health Conditions:** Smoking, diabetes, cardiovascular disease, stroke

# # **Data Setup and Cleaning:**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# creating the dataframes to work with

counties = pd.read_csv('../input/covid19-preexisting-conditions/abridged_couties.csv')
confirmed = pd.read_csv('../input/covid19-preexisting-conditions/time_series_covid19_confirmed_US.csv')
deaths = pd.read_csv('../input/covid19-preexisting-conditions/time_series_covid19_deaths_US.csv')
states_april = pd.read_csv('../input/covid19/4.18states.csv')


# In[ ]:


#cleaning data sets
states_april = states_april[['Province_State', 'Confirmed', 'Deaths', 'Active','People_Tested']]
states_april = states_april.fillna(0)
states_april.head()


# In[ ]:


counties = counties[['CountyName', 'State', 'PopulationEstimate2018', 'DiabetesPercentage', 'HeartDiseaseMortality', 'StrokeMortality', 'Smokers_Percentage']]
counties = counties[counties["State"].isnull() == False]
counties = counties.fillna(0)
counties.head()


# Find more information pertaining to the counties dataframe here: https://github.com/Yu-Group/covid19-severity-prediction/blob/master/data/list_of_columns.md

# In[ ]:


#Cleaning confirmed dataframe and grouping by state

confirmed = confirmed.dropna()
confirmed = confirmed.groupby(by="Province_State", as_index=False).mean()
confirmed = confirmed.drop(columns={"UID", "code3", "FIPS", "Lat", "Long_"})
confirmed.head()


# In[ ]:


#Cleaning deaths dataframe and grouping by state

deaths = deaths.dropna()
deaths = deaths.groupby(by="Province_State", as_index=False).mean()
deaths = deaths.drop(columns={"UID", "code3", "FIPS", "Lat", "Long_"})
deaths.head()


# # **Exploratory Data Analysis (EDA):**

# ## **Exploring death rates and positive test rates in different states.**

# In[ ]:


#Finding the death rates and positive rates for each state in the US

explore = counties[["CountyName", "State"]]
explore = explore.dropna()
explore = explore.groupby(by="State", as_index=False).count()
explore = explore.merge(states_april, left_on='State', right_on='Province_State')
explore = explore[["State", "Confirmed", "Deaths", "People_Tested"]]
explore["Death_Rate"] = (explore["Deaths"] / explore["Confirmed"]) * 100
explore["Positive_Rate"] = (explore["Confirmed"] / explore["People_Tested"]) * 100
explore = explore.drop(columns=["Confirmed", "Deaths", "People_Tested"])
explore.head()


# In[ ]:


#Let's visualize the death and positive rates according to states

import plotly.express as px

fig = px.scatter(explore, x="Death_Rate", y="Positive_Rate", text=explore["State"])

fig.update_traces(textposition='top center')

fig.update_layout(height=800, title_text='US States and Death/Positive Rates')

fig.show()


# The plot aligns with the early spread of the virus and high number of cases in the tri-state area, i.e. New York, New Jersey, and Connecticut, and also in the state of Michigan.

# In[ ]:


highest_death_rate = max(explore["Death_Rate"])
highest_death_rate_state = explore["State"][explore["State"].index == np.argmax(explore["Death_Rate"])].to_list()
print("The highest Death Rate is", highest_death_rate, "in the state of", *highest_death_rate_state)


# In[ ]:


lowest_death_rate = min(explore["Death_Rate"])
lowest_death_rate_state = explore["State"][explore["State"].index == np.argmin(explore["Death_Rate"])].to_list()
print("The lowest Death Rate is", lowest_death_rate, "in the state of", *lowest_death_rate_state)


# In[ ]:


highest_positive_rate = max(explore["Positive_Rate"])
highest_positive_rate_state = explore["State"][explore["State"].index == np.argmax(explore["Positive_Rate"])].to_list()
print("The highest Positive Test Rate is", highest_positive_rate, "in the state of", *highest_positive_rate_state)


# In[ ]:


lowest_positive_rate = min(explore["Positive_Rate"])
lowest_positive_rate_state = explore["State"][explore["State"].index == np.argmin(explore["Positive_Rate"])].to_list()
print("The lowest Positive Test Rate is", lowest_positive_rate, "in the state of", *lowest_positive_rate_state)


# ## **1. The impact of smoking on the death rate and the positive test rate.**

# In[ ]:


impact_smoking = counties[["CountyName", "State", "Smokers_Percentage"]]
impact_smoking = impact_smoking.dropna()
impact_smoking = impact_smoking[impact_smoking["Smokers_Percentage"] > 0]
impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()


# In[ ]:


#We are grouping by state because the county data is not available in the other spreadsheet

impact_smoking = impact_smoking.groupby(by="State", as_index=False).mean()
impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()


# In[ ]:


impact_smoking = impact_smoking.merge(states_april, left_on='State', right_on='Province_State')
impact_smoking = impact_smoking[["State", "Smokers_Percentage", "Confirmed", "Deaths", "People_Tested"]]
impact_smoking["Death_Rate"] = (impact_smoking["Deaths"] / impact_smoking["Confirmed"]) * 100
impact_smoking["Positive_Rate"] = (impact_smoking["Confirmed"] / impact_smoking["People_Tested"]) * 100
impact_smoking = impact_smoking.drop(columns=["Confirmed", "Deaths", "People_Tested"])
impact_smoking.sort_values(by=["Smokers_Percentage"], ascending=False).head()


# In[ ]:


#Finding correlation between state's smoking percentage vs their death rate

smoking_correlation_death = impact_smoking["Smokers_Percentage"].corr(impact_smoking["Death_Rate"])
print("The correlation is", smoking_correlation_death)


# In[ ]:


plt.scatter(impact_smoking["Smokers_Percentage"], impact_smoking["Death_Rate"])
plt.xlabel('Smokers Percentage')
plt.ylabel('Death Rate')
plt.show()


# In[ ]:


#Finding correlation between state's smoking percentage vs their positive rate

smoking_correlation_positve = impact_smoking["Smokers_Percentage"].corr(impact_smoking["Positive_Rate"])
print("The correlation is", smoking_correlation_positve)


# In[ ]:


plt.scatter(impact_smoking["Smokers_Percentage"], impact_smoking["Positive_Rate"])
plt.xlabel('Smokers Percentage')
plt.ylabel('Positive Rate')
plt.show()


# Conclusion:
# 
# The correlation between smoking percentage and the death rate came out to be positive but the number was very small. However, the state with the lowest death rate is South Dakota, which has 2 counties in the top 4 counties with the highest smoking percentages. 
# 
# The correlation between smoking percentage and the positive rate came out to be negative and small. However, this might suggest that smoking does in fact bring down the number of positive cases.
# 
# This suggests that if we had the data for the death rate according to counties, we could potentially find a negative correlation between smoking and falling sick with the coronavirus.

# ## **2. The impact of diabetes on the death rate and the positive test rate.**
# 

# In[ ]:


impact_diabetes = counties[["CountyName", "State", "DiabetesPercentage"]]
impact_diabetes = impact_diabetes.dropna()
impact_diabetes = impact_diabetes[impact_diabetes["DiabetesPercentage"] > 0]
impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()


# In[ ]:


#We are grouping by state because the county data is not available in the other spreadsheet
#Observe the sharp decrease in % when doing this; Tippah County of MS is 33% but the entire state averages to be <15%

impact_diabetes = impact_diabetes.groupby(by="State", as_index=False).mean()
impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()


# In[ ]:


impact_diabetes = impact_diabetes.merge(states_april, left_on='State', right_on='Province_State')
impact_diabetes = impact_diabetes[["State", "DiabetesPercentage", "Confirmed", "Deaths", "People_Tested"]]
impact_diabetes["Death_Rate"] = (impact_diabetes["Deaths"] / impact_diabetes["Confirmed"]) * 100
impact_diabetes["Positive_Rate"] = (impact_diabetes["Confirmed"] / impact_diabetes["People_Tested"]) * 100
impact_diabetes = impact_diabetes.drop(columns=["Confirmed", "Deaths", "People_Tested"])
impact_diabetes.sort_values(by=["DiabetesPercentage"], ascending=False).head()


# In[ ]:


#Finding correlation between state's diabetes percentage vs their death rate

diabetes_correlation_death = impact_diabetes["DiabetesPercentage"].corr(impact_diabetes["Death_Rate"])
print("The correlation is:", diabetes_correlation_death)


# In[ ]:


plt.scatter(impact_diabetes["DiabetesPercentage"], impact_diabetes["Death_Rate"])
plt.xlabel('Diabetes Percentage')
plt.ylabel('Death Rate')
plt.show()


# In[ ]:


#Finding correlation between state's diabetes percentage vs their positive rate

smoking_correlation_positve = impact_diabetes["DiabetesPercentage"].corr(impact_diabetes["Positive_Rate"])
print("The correlation is:", smoking_correlation_positve)


# In[ ]:


plt.scatter(impact_diabetes["DiabetesPercentage"], impact_diabetes["Positive_Rate"])
plt.xlabel('Diabetes Percentage')
plt.ylabel('Positive Rate')
plt.show()


# Conclusion:
# 
# The correlation between diabetes percentage and the death rate came out to be negative but the number was very small.
# 
# The correlation between diabetes percentage and the positive rate came out to be negative and small. This result is kind of surprising but at the same time not significant enough to make a conclusion.

# ## **3. The impact of heart disease on the death rate and the positive test rate.**

# In[ ]:


impact_heart = counties[["CountyName", "State", "HeartDiseaseMortality"]]
impact_heart = impact_heart.dropna()
impact_heart = impact_heart[impact_heart["HeartDiseaseMortality"] > 0]
impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()


# In[ ]:


#We are grouping by state because the county data is not available in the other spreadsheet

impact_heart = impact_heart.groupby(by="State", as_index=False).mean()
impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()


# In[ ]:


impact_heart = impact_heart.merge(states_april, left_on='State', right_on='Province_State')
impact_heart = impact_heart[["State", "HeartDiseaseMortality", "Confirmed", "Deaths", "People_Tested"]]
impact_heart["Death_Rate"] = (impact_heart["Deaths"] / impact_heart["Confirmed"]) * 100
impact_heart["Positive_Rate"] = (impact_heart["Confirmed"] / impact_heart["People_Tested"]) * 100
impact_heart = impact_heart.drop(columns=["Confirmed", "Deaths", "People_Tested"])
impact_heart.sort_values(by=["HeartDiseaseMortality"], ascending=False).head()


# In[ ]:


#Finding correlation between state's heart disease mortality vs their death rate

heart_correlation_death = impact_heart["HeartDiseaseMortality"].corr(impact_heart["Death_Rate"])
print("The correlation is:", heart_correlation_death)


# In[ ]:


plt.scatter(impact_heart["HeartDiseaseMortality"], impact_heart["Death_Rate"])
plt.xlabel('Heart Disease Mortality')
plt.ylabel('Death Rate')
plt.show()


# In[ ]:


#Finding correlation between state's heart rate mortality vs their positive rate

heart_correlation_positve = impact_heart["HeartDiseaseMortality"].corr(impact_heart["Positive_Rate"])
print("The correlation is:", heart_correlation_positve)


# In[ ]:


plt.scatter(impact_heart["HeartDiseaseMortality"], impact_heart["Positive_Rate"])
plt.xlabel('Heart Disease Mortality')
plt.ylabel('Positive Rate')
plt.show()


# Conclusion:
# 
# The correlation between heart rate mortality and the death rate came out to be positive, which is not surprising as people with heart disease are more suceptible to the virus.
# 
# The correlation between heart rate mortality and the positive rate came out to be positive but small and insignificant.

# ## **4. The impact of stroke mortality on the death rate and the positive test rate.**

# In[ ]:


impact_stroke = counties[["CountyName", "State", "StrokeMortality"]]
impact_stroke = impact_stroke.dropna()
impact_stroke = impact_stroke[impact_stroke["StrokeMortality"] > 0]
impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()


# In[ ]:


#We are grouping by state because the county data is not available in the other spreadsheet

impact_stroke = impact_stroke.groupby(by="State", as_index=False).mean()
impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()


# In[ ]:


impact_stroke = impact_stroke.merge(states_april, left_on='State', right_on='Province_State')
impact_stroke = impact_stroke[["State", "StrokeMortality", "Confirmed", "Deaths", "People_Tested"]]
impact_stroke["Death_Rate"] = (impact_stroke["Deaths"] / impact_stroke["Confirmed"]) * 100
impact_stroke["Positive_Rate"] = (impact_stroke["Confirmed"] / impact_stroke["People_Tested"]) * 100
impact_stroke = impact_stroke.drop(columns=["Confirmed", "Deaths", "People_Tested"])
impact_stroke.sort_values(by=["StrokeMortality"], ascending=False).head()


# In[ ]:


#Finding correlation between state's stroke mortality vs their death rate

stroke_correlation_death = impact_stroke["StrokeMortality"].corr(impact_stroke["Death_Rate"])
print("The correlation is:", stroke_correlation_death)


# In[ ]:


plt.scatter(impact_stroke["StrokeMortality"], impact_stroke["Death_Rate"])
plt.xlabel('Stroke Mortality')
plt.ylabel('Death Rate')
plt.show()


# In[ ]:


#Finding correlation between state's stroke mortality vs their positive test rate

stroke_correlation_positve = impact_stroke["StrokeMortality"].corr(impact_stroke["Positive_Rate"])
print("The correlation is:", stroke_correlation_positve)


# In[ ]:


plt.scatter(impact_stroke["StrokeMortality"], impact_stroke["Positive_Rate"])
plt.xlabel('Stroke Mortality')
plt.ylabel('Positive Rate')
plt.show()


# Conclusion:
# 
# The correlation between stroke mortality and the death rate came out to be negative surprisingly.
# 
# The correlation between stroke mortality and the positive rate came out to be negative surprisingly.
# 
# This surprise result could be because of the fact that stroke mortality rates are from 2014-2016 and not live data as young COVID-19 patients are suffering and dying from stroke.

# # **Modeling:**

# Observing the EDA above, we can see that there are clearly some certain health conditions that potentially demonstrate a higher likelihood of contracting COVID-19. This is demonstrated by the greater absolute value of correlation.

# ## **Modeling the Relationship between Stroke and Smoking with Testing Positive for COVID-19**

# Two conditions that clearly stick out considering the Positive Rate value are Stroke and Smoking. Below we construct a model to try to predict whether an individual has a higher likelihood of testing positive COVID-19. We first begin by merging the tables created in the EDA section for the Impact of Stroke and the Impact of Smoking.

# In[ ]:


impact_stroke_smoking = impact_stroke.merge(impact_smoking, on='State')[['State', 'StrokeMortality', 'Smokers_Percentage', 'Positive_Rate_x']]
impact_stroke_smoking.head()


# We then extract the two covariates we hope to use to build our model, which in this case are simply the StrokeMortality and Smokers_Percentage. We assign this reduced dataframe to the value X.
# We then assign Y to an array representing the corresponding Positive Rate values for each state.

# In[ ]:


X_positive = impact_stroke_smoking[['StrokeMortality', 'Smokers_Percentage']]
X_positive.head()


# In[ ]:


Y_positive = np.array(impact_stroke_smoking['Positive_Rate_x'])
Y_positive


# We perform a train-test split below.

# In[ ]:


from sklearn.model_selection import train_test_split

np.random.seed(41)
X_positive_train, X_positive_test, Y_positive_train, Y_positive_test = train_test_split(X_positive, Y_positive, test_size = 0.10)


# We can now start building our model. We use a linear model because the values of data we are using here are continuous in nature, not categorical.

# In[ ]:


from sklearn import linear_model as lm

linear_model_positive = lm.LinearRegression(fit_intercept=True)
linear_model_positive.fit(X_positive_train, Y_positive_train)


# Let's then compute both the fitted and the predicted values of the training subset of data.

# In[ ]:


linear_model_positive.predict(X_positive_train)


# Compare the model with the actual data using the RMSE function.

# In[ ]:


def rmse(predicted, actual):
    return np.sqrt(np.mean((actual - predicted)**2))


# In[ ]:


rmse(linear_model_positive.predict(X_positive_train), Y_positive_train)


# The RMSE value is apporixmately 8.72 which we will later interpret and compare against the RMSE value of the other model to follow.

# Below, we create a regression plot visualization using seaborn. If the result of the plot is a horizontal line of points at 0, this means that the prediction is perfectly correlated. However, the plot is clearly not close to this perfect situation.

# In[ ]:


import seaborn as sns

#We would hope to see a horizontal line of points at 0.

residuals = Y_positive_test - linear_model_positive.predict(X_positive_test)
ax = sns.regplot(Y_positive_test, residuals)
ax.set_xlabel('Death Rate (Test Data)')
ax.set_ylabel('Residuals (Actual Rate - Predicted Rate)')
ax.set_title("Residuals vs. Death Rate on Test Data");


# Conclusion: This model seems to demonstrate a poor prediction of correlation. Let us then observe the relationship between Heart Disease and Stroke with the chance of dying from COVID-19.

# ## **Modeling the Relationship between Heart Disease and Stroke with Dying from COVID-19**

# We will now build a model using the two health conditions with the most impactful correlation value to the death rate of COVID-19. These conditions are Heart Disease and Stroke. We again begin by merging the tables created in the EDA section for the Impact of Stroke and the Impact of Heart Disease, focusing now on the Death Rate instead of the Positive Test percentage.

# In[ ]:


impact_stroke_heart = impact_stroke.merge(impact_heart, on='State')[['State', 'StrokeMortality', 'HeartDiseaseMortality', 'Death_Rate_x']]
impact_stroke_heart.head()


# We then extract the two covariates we hope to use to build our model, which in this case are the StrokeMortality and HeartDiseaseMortality. We assign this reduced dataframe to the value X. We then assign Y to an array representing the corresponding Date Rate values for each state.

# In[ ]:


X_death = impact_stroke_heart[['StrokeMortality', 'HeartDiseaseMortality']]
X_death.head()


# In[ ]:


Y_death = np.array(impact_stroke_heart['Death_Rate_x'])
Y_death


# We perform a train-test split below.

# In[ ]:


np.random.seed(41)
X_death_train, X_death_test, Y_death_train, Y_death_test = train_test_split(X_death, Y_death, test_size = 0.10)


# Let's take another look at X_death.

# In[ ]:


X_death.head()


# There is a key issue here: the range of HeartDiseaseMorality values is considerably greater than that of the StrokeMortality column. Therefore, we will normalize the data using the function written below.

# In[ ]:


def normalize(data):

    output = (data - np.mean(data))/np.std(data)
    return output.replace(np.nan, 0)
    
X_normalized = normalize(X_death_train)
X_normalized.head()


# We can now build a linear model because the values of data are continuous, not categorical.

# In[ ]:


linear_model_death = lm.LinearRegression(fit_intercept=True)
linear_model_death.fit(normalize(X_death_train), Y_death_train)


# Let's then compute both the fitted and the predicted values of the training subset of data.

# In[ ]:


linear_model_death.predict(normalize(X_death_train))


# In[ ]:


rmse(linear_model_death.predict(normalize(X_death_train)), Y_death_train)


# Again, we will create a seaborn regression plot visualization to analyze the relationship between the residuals and the death rate based on test data for the death rate subset.

# In[ ]:


#Ideally, we would see a horizontal line of points at 0.

residuals = Y_death_test - linear_model_death.predict(normalize(X_death_test))
ax = sns.regplot(Y_death_test, residuals)
ax.set_xlabel('Death Rate (Test Data)')
ax.set_ylabel('Residuals (Actual Rate - Predicted Rate)')
ax.set_title("Residuals vs. Death Rate on Test Data");


# Again, because the line is not horizontal at 0, this shows that the model does not do a great job predicting the death rate of those who have COVID-19 given the two chosen pre-existing health conditions.

# The resulting rmse for the death rate linear model is approximately 1.33.
# This is considerably lower than the rmse value for the positive rate linear model which is 8.71. However, note that these two models are predicting two very different things. Now, let's look at the test rmse values for each model.

# In[ ]:


Positive_Rate_Test_RMSE = rmse(linear_model_positive.predict(X_positive_test), Y_positive_test)
Death_Rate_Test_RMSE = rmse(linear_model_death.predict(normalize(X_death_test)), Y_death_test)

print("The Positive Rate Test RMSE is {}".format(Positive_Rate_Test_RMSE))
print("The Death Rate Test RMSE is {}".format(Death_Rate_Test_RMSE))


# As shown, the Death Rate RMSE on the test set of data for the two chosen health condition covariates is significantly lower than that of the Positive Rate RMSE. To observe this difference further, let's compare the R^2 scores for both models.

# Below, compare the R^2 score for each model we built. This will help us observe the goodness of fit of the model on a scale from 0-1.

# In[ ]:


#Best possible value is 1.0 as r^2 measures goodness of fit

from sklearn.metrics import r2_score

Positive_Rate_Test_R2 = r2_score(Y_positive_test, linear_model_positive.predict(X_positive_test))
Death_Rate_Test_R2 = r2_score(Y_death_test, linear_model_death.predict(normalize(X_death_test)))

print("The Positive Rate Test R^2 is {}".format(Positive_Rate_Test_R2))
print("The Death Rate Test R^2 is {}".format(Death_Rate_Test_R2))


# Conclusion: The models we built above are far from perfect in their ability to predict whether an individual has a higher likelihood of testing positive or dying from COVID-19 given selected pre-existing health conditions. However, these models are not entirely imperfect either. For instance, the R^2 value of 0.18 demonstrates that some part of the death rate can be explained using pre-existing conditions.
# 

# In[ ]:




