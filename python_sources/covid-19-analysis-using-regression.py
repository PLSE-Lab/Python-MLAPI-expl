#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. What is Covid-19?
# 2. Dataset
# 3. World statistics
# 4. Linear models
# 5. Correlation of spread between countries
# 6. Conclusion

# # 1. What is Covid-19?
# Covid-19 or Coronavirus disease is an infectious disease caused by a newly discovered coronavirus. It was first identified in December 2019 in Wuhan, China. As of May 15 2020 more than 4 million confirmed cases and more than 300K deaths over 188 countries. The person infected with this virus will experience mild to moderate respiratory illness. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
# 
# The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes. It is most contagious during the first three days after the onset of symptoms, although spread is possible before symptoms appear, and from people who do not show symptoms. The standard method of diagnosis is by real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab. At this time, there are no specific vaccines or treatments for COVID-19. There are many ongoing trials and researches evaluating potential treatment.
# 
# 
# 

# # 2. Dataset
# This dataset has daily level information on the number of affected cases, deaths and recovery from 2019 novel coronavirus. Please note that this is a time series data and so the number of cases on any given day is the cumulative number.
# 
# The data is available from 22 Jan, 2020.
# 
# ## Column Description
# Main file in this dataset is covid_19_data.csv and the detailed descriptions are below.
# 
# covid_19_data.csv
# 1. Sno - Serial number
# 2. ObservationDate - Date of the observation in MM/DD/YYYY
# 3. Province/State - Province or state of the observation (Could be empty when missing)
# 4. Country/Region - Country of observation
# 5. Last Update - Time in UTC at which the row is updated for the given province or country. (Not standardised and so please clean before using it)
# 6. Confirmed - Cumulative number of confirmed cases till that date
# 7. Deaths - Cumulative number of of deaths till that date
# 8. Recovered - Cumulative number of recovered cases till that date
# 
# This dataset contains 8 csv files containing number of cases confirmed, deaths, recovered etc.,.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For Viz
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
import datetime as dt
'''
sklearn.linear_model.LinearRegression

'''


get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 3. World Statistics
# 
# 

# The below graph shows how the number of confirmed cases rose after 22/01/2020. The number of daily cases 20/03/2020 was very low compared to later stages. This is when the virus started spreading more in the US and other european countries.

# In[ ]:


df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
confirmed_dates = df_confirmed.columns[4:]
cumulative_confirmed_data = []

for date in confirmed_dates:
    cumulative_confirmed_data.append(df_confirmed[date].sum())

confirmed_cases_per_day = [cumulative_confirmed_data[0]]
for i in range(1, len(cumulative_confirmed_data)):
    confirmed_cases_per_day.append(cumulative_confirmed_data[i] - cumulative_confirmed_data[i - 1])

days = len(confirmed_dates)

fig = make_subplots(rows=1, cols=2)
fig.append_trace(go.Scatter(x=confirmed_dates, y=cumulative_confirmed_data, text="Confirmed"), row=1, col=1)
fig.append_trace(go.Bar(x=confirmed_dates, y=confirmed_cases_per_day, text="Confirmed this day"), row=1, col=2)

fig.update_layout(height=600, width=1500, title_text =  "Number of cases confirmed total vs Each day")
fig.show()
df_confirmed.head()


# The above table shows various columns expalaining about the number of cases in each countries cumulatively.

# Let's now interpret the graph of death cases. The graph more resembles the number of cases confirmed shown above. 

# In[ ]:


df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
deaths_dates = df_deaths.columns[4:]
cumulative_deaths_data = []

for date in deaths_dates:
    cumulative_deaths_data.append(df_deaths[date].sum())

deaths_cases_per_day = [cumulative_deaths_data[0]]
for i in range(1, len(cumulative_deaths_data)):
    deaths_cases_per_day.append(cumulative_deaths_data[i] - cumulative_deaths_data[i - 1])

fig = make_subplots(rows=1, cols=2)
fig.append_trace(go.Scatter(x=deaths_dates, y=cumulative_deaths_data, text="Deaths"), row=1, col=1)
fig.append_trace(go.Bar(x=deaths_dates, y=deaths_cases_per_day, text="Deaths this day"), row=1, col=2)

fig.update_layout(height=600, width=1500, title_text="Deaths Cumulative vs Each Day")
fig.show()


# Since the patients usually recover after 2 weeks, this graph also synchronized with the confirmed cases grpah. The number of recovered cases at 30/04/2020 is very high which is about 60k.

# In[ ]:


df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
recovered_dates = df_recovered.columns[4:]
cumulative_recovered_data = []

for date in recovered_dates:
    cumulative_recovered_data.append(df_recovered[date].sum())

recovered_cases_per_day = [cumulative_recovered_data[0]]
for i in range(1, len(cumulative_recovered_data)):
    recovered_cases_per_day.append(cumulative_recovered_data[i] - cumulative_recovered_data[i - 1])

fig = make_subplots(rows=1, cols=2)
fig.append_trace(go.Scatter(x=recovered_dates, y=cumulative_recovered_data, text="Recovered"), row=1, col=1)
fig.append_trace(go.Bar(x=recovered_dates, y=recovered_cases_per_day, text="Recovered this day"), row=1, col=2)

fig.update_layout(height=600, width=1500, title_text="Recovered Cumulative vs Each Day")
fig.show()


# # 4. Regression models
# The regression techniques are employed for investigating or examining the relationship between the dependent and independent set of variables. To establish the possible relationship among different variables, various modes of statistical approaches are implemented, known as regression analysis. In order to understand how the variation in an independent variable can impact the dependent variable, regression analysis is specially molded out.
# 
# ## Types of regression techniques
# 1. Linear Regression
# 2. Polynomial Regression
# 3. Ridge Regression
# 4. Lasso Regression
# 5. ElasticNet Regression
# 

# Let's see how each of these models work with the above data. 

# In[ ]:


X_train = [[day] for day in range(1, days + 1)]

DAYS_TO_PREDICT = 20
DATE_FORMAT = "%m/%d/%Y"

X_test = [[day] for day in range(1, days + DAYS_TO_PREDICT)]

dates = []
for i in confirmed_dates:
    dates.append(i)
current_date = dt.datetime.strptime(dates[-1] + '20', DATE_FORMAT).date()

for i in range(DAYS_TO_PREDICT):
    date = current_date + dt.timedelta(days=i + 1)
    dates.append([date.strftime(DATE_FORMAT)[:-2]])

Y_confirmed = cumulative_confirmed_data
Y_recovered = cumulative_recovered_data
Y_deaths = cumulative_deaths_data

def Linear(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

def Poly(X, degree=2):
    poly = PolynomialFeatures(degree)
    new_X = poly.fit_transform(X)
    return new_X

def RidgeRegression(X, Y, _alpha=1):
    model = Ridge(alpha=_alpha)
    model.fit(X, Y)
    return model

def LassoRegression(X, Y, _alpha=1):
    model = Lasso(alpha=_alpha)
    model.fit(X, Y)
    return model

def Elastic(X, Y):
    model = ElasticNet(random_state=0)
    model.fit(X, Y)
    return model

def Predict(model, X):
    return model.predict(X)


# Comparison between regression models predicting number of future confirmed cases. The below graph shows how each of the five models fit the number of confirmed cases worldwide. The fit of each model except linear regression is more or less fits with the actual data. But the predictions given for later dates by those models are bit different. Increasing the polynomial transform degree also overfits the data which is of no use. We can see that ridge regression after transforming with polynomial of degree 2 gives a good fit and prediction.
# 

# # PREDICTIONS
# Various linear models are used to predict the future cases globally. The prediction is done for 20 days. It can be changed by setting the DAYS_TO_PREDICT variable in the above code.

# In[ ]:


linear = Linear(X_train, Y_confirmed)
linear_prediction = Predict(linear, X_test)

POLYNOMIAL_DEGREE = 4

poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_confirmed)
poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))

RIDGE_DEGREE = 2
RIDGE_ALPHA = 1

ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_confirmed, RIDGE_ALPHA)
ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))

LASSO_DEGREE = 7
LASSO_ALPHA = 1

lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_confirmed, LASSO_ALPHA)
lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))

fig = go.Figure()

fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Confirmed cases"))
fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))
fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))
fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))
fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))


fig.update_layout(
    title={
        'text': "Comparision of future cases using various linear models",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()


# None of the regression models does well when fitting with the number of deaths. Since the number of deaths in first half of the data is very small when compared to the second half. These models tried their best to fit the first half of the model and failed to do well in the second half. This is when the disease spread across european countries like Italy and Germany which led to numerous deaths.

# In[ ]:


linear = Linear(X_train, Y_deaths)
linear_prediction = Predict(linear, X_test)

POLYNOMIAL_DEGREE = 5

poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_deaths)
poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))

RIDGE_DEGREE = 2
RIDGE_ALPHA = 1

ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_deaths, RIDGE_ALPHA)
ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))

LASSO_DEGREE = 7
LASSO_ALPHA = 1

lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_deaths, LASSO_ALPHA)
lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))

fig = go.Figure()

fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Deaths"))
fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))
fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))
fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))
fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))

fig.update_layout(
    title={
        'text': "Comparision of future deaths using various linear models",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()


# This regression analysis doesnt do well with the number of deaths. Initially the total number of people died were less. This is when the disease spread only across China. Still China have only around 4000 deaths for 80,000 cases. When the disease spread to some other European countries and USA total number of deaths increased exponentially. This recent change is not captured by the regression algorithms. More data analysis is to be done to set the right parameters for the graph. The parameter can be changed accordingly in the above code. Various parameters like RIDGE_DEGREE and ALPHA values can be altered. The data transformation used for this data is polynomial transform. The Degree of the transformation can be changed accordingly. 
# 
# The average age in italy is around 47 and in other countries it is much lower. In India it is around 27. This explains why italy has more deaths than any other countries and number of deaths in India is lesser eventhough the number of cases is higher.

# In[ ]:


linear = Linear(X_train, Y_recovered)
linear_prediction = Predict(linear, X_test)

POLYNOMIAL_DEGREE = 10

poly = Linear(Poly(X_train, POLYNOMIAL_DEGREE), Y_recovered)
poly_prediction = Predict(poly, Poly(X_test, POLYNOMIAL_DEGREE))

RIDGE_DEGREE = 2
RIDGE_ALPHA = 1

ridge = RidgeRegression(Poly(X_train, RIDGE_DEGREE), Y_recovered, RIDGE_ALPHA)
ridge_prediction = Predict(ridge, Poly(X_test, RIDGE_DEGREE))

LASSO_DEGREE = 2
LASSO_ALPHA = 1

lasso = LassoRegression(Poly(X_train, LASSO_DEGREE), Y_recovered, LASSO_ALPHA)
lasso_prediction = Predict(lasso, Poly(X_test, LASSO_DEGREE))

fig = go.Figure()

fig.add_trace(go.Scatter(x=recovered_dates, y=cumulative_confirmed_data, name="Actual Recorvered cases"))
fig.add_trace(go.Scatter(x=dates, y=linear_prediction, name="Linear Regression"))
fig.add_trace(go.Scatter(x=dates, y=poly_prediction, name="Polynomial Regression"))
fig.add_trace(go.Scatter(x=dates, y=ridge_prediction, name="Ridge Regression"))
fig.add_trace(go.Scatter(x=dates, y=lasso_prediction, name="Lasso Regression"))


fig.update_layout(
    title={
        'text': "Comparision of future recovery using various linear models",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()


# Regression does quite well predicting recovery cases compared to total deaths. This algorithm doesnt overestimate the recovery but changing the degree of the polynomial further may overfit the data or produce unpredicted predictions. The parameters value can be altered similar to above methods.

# # 5. Top 10 countries
#   

# In[ ]:


TOP_COUNTRY_COUNT = 10

country_wise_total = []
top_countries = []
for i in range(len(df_confirmed)):
    row = df_confirmed.iloc[i]
    country = row['Country/Region']
    confirmed_cases = row[-1]
    country_wise_total.append((country, confirmed_cases))

country_wise_total.sort(key=lambda x: x[-1], reverse=True)
df_country_wise = pd.DataFrame(country_wise_total, columns=['Country', 'Cases'])
# p = plt.pie(df_country_wise['Cases'][:10], labels=df_country_wise['Country'][:10])
px.pie(df_country_wise, names=df_country_wise['Country'][:TOP_COUNTRY_COUNT], values=df_country_wise['Cases'][:TOP_COUNTRY_COUNT], title='Top 10 contires contribution so far')


# The pie chart explains the distributions of the disease for the 10 contries. US contributes 40% of the total count. Brazil although with very few tests and more positive results stands in second contributing 20% approximately.  India and Russia is slowly catching up in the race. If lockdown hasnt been announced then the disease could have spread exponentially in India due to higher population density. Still Mumbai, Chennai are at peak due to its population density per square kilometers.

# In[ ]:


store = {}

for i in range(TOP_COUNTRY_COUNT):
    top_countries.append(country_wise_total[i][0])
    
for i in range(len(df_confirmed)):
    row = df_confirmed.iloc[i]
    if not (row[1] in top_countries):
        continue
    series = list(row[4:])
    country = row[1]
    store[country] = list(series)

df_top_countries = pd.DataFrame(store)
corr = df_top_countries.corr()
corr


# The below heatmap shows how the rate of increase between top 10 countries are similar. The yellow part indicates that these countries are highly correlated meaning that the spread is similar.

# In[ ]:


px.imshow(corr, labels=dict(x="Countries", y="Countries", color="Correlation"), x = corr.columns, y = corr.columns, title="Similar spread comparision between contries")


# A correlation coefficient of 1 means that these two contries have similar rate of increase in the count of confirmed cases. As you can see India Brazil and US have the color yellow which have correlation coefficient of 1. This means that if no actions are taken then the disease could spread more leading to economic breakdown of the country and people.

# # Conclusion
# Social distancing is to be followed to stop the transmission of coronavirus in large crowds in meetings, movie halls, weddings and pubic transport. To enforce the importance of social distancing, schools, colleges, malls and movie halls are now closed across states. People are being advised to work remotely and reduce interaction with other people to a minimum. It is important for people to have a certain level of social interaction for good mental health. Therefore, according to the spread of the disease, different levels of social distancing can be followed. As per advisories, it is safe to have up to 10 people in a closed environment. The development of effective vaccines is still in progress until then social distancing, wearing masks are mandotory. Stay safe and stay hygenic all the time.
