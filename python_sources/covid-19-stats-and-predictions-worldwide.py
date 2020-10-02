#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 stats and predictions with LSTM neural networks
# ---
# 
# This noteboook makes comparative analysis and future predictions of covid-19 active cases and deaths for 18 countries: 
# - 9 European Union countries: Spain, Belgium, France, Germany, Italy, Netherlands, Portugal, Sweden, Switzerland
# - 9 world countries: Brazil, Canada, China, India, Iran, Mexico, Russia, United Kingdom, United States (US)
# 
# From version 68 (update on 14/08/2020) the code has been modified to read the data files directly from the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University: 
# 
#     https://github.com/CSSEGISandData/COVID-19
# 
# These files are updated daily and hence you can get an up-to-date, fresh execution any time you run the notebook. Also these files contain confirmed cases, recovered and dates cumulative numbers for 188 countries in the world, so whilst I use a subset of 18 countries, you can easily fork the notebook and taylor it to your needs.
# 
# The primary (measured) variables I will work with, as obtained from the repository files, are:
# 
#     - number of confirmed cases
#     - number of recovered cases
#     - number of deaths
# 
# Then I will calculate the number of active cases at any time as:
# 
#     Active Cases = Confirmed cases - Recovered - Deaths
# 
# In addition to the above, I will also calculate mortality and rates of growth (daily difference).
# 
# Finally, I will implement LSTM recurring neural networks (RNN) with input data of all the selected countries, some of which are in different stages of the pandemic evolution, with the expectation that the RNN will enhance its predictive power of those curves lagging behind by using its knowledge of more evolved curves.
# 
# I hope you will find it interesting. Please upvote me if you do!!!

# In[ ]:


import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn')
              
# Set precision to two decimals
pd.set_option("display.precision", 4)

# Define date format for charts like Apr 16 or Mar 8
my_date_fmt = mdates.DateFormatter('%b %e')


# ## Data load and pre-processing
# ---

# First I will download the three files (confirmed cases, recovered and deaths) from their github repository and do the required pre-processing.

# ### Read files and tidy up

# In[ ]:


# Download files from github
cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df_cases = pd.read_csv(cases_url, error_bad_lines=False)

deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_deaths = pd.read_csv(deaths_url, error_bad_lines=False)

recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
df_recovered = pd.read_csv(recovered_url, error_bad_lines=False)


# In[ ]:


# Drop Province/State, Lat and Long
df_cases.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_deaths.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
df_recovered.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)

# Rename Country/Region as Country
df_cases.rename(columns={'Country/Region' : 'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region' : 'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region' : 'Country'}, inplace=True)

# Some countries (Australia, Canada...) report data by province so we need to aggregate it
df_cases = df_cases.groupby(by='Country').sum()
df_deaths = df_deaths.groupby(by='Country').sum()
df_recovered = df_recovered.groupby(by='Country').sum()

# Transpose dataframes and make the date column index of datetime type
df_cases = df_cases.T
df_cases.index = pd.to_datetime(df_cases.index)
df_deaths = df_deaths.T
df_deaths.index = pd.to_datetime(df_deaths.index)
df_recovered = df_recovered.T
df_recovered.index = pd.to_datetime(df_recovered.index)


# In[ ]:


# Get last date in the set
last_date = df_cases.tail(1).index[0]
print('Last date in the set: ' + str(datetime.date(last_date)))


# In[ ]:


# List of countries for this work
country_list = ['Spain', 'Belgium', 'France', 'Germany', 'Italy', 'Netherlands', 'Portugal', 'Sweden', 'Switzerland',  
                 'Brazil', 'Canada', 'China', 'India', 'Iran',  'Mexico', 'Russia', 'United Kingdom', 'US']
clist1 = ['Spain', 'Belgium', 'France', 'Germany', 'Italy', 'Netherlands', 'Portugal', 'Sweden', 'Switzerland']
clist2 = ['Brazil', 'Canada', 'China', 'India', 'Iran', 'Mexico', 'Russia', 'United Kingdom', 'US']


# In[ ]:


# Extract selection of countries
df_cases = df_cases[country_list]
df_recovered = df_recovered[country_list]
df_deaths = df_deaths[country_list]


# ### Compute active cases

# In[ ]:


# Create new dataframe for active cases
# Active = Cases - Deaths - Recovered
df_active = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
df_active = df_cases - df_deaths - df_recovered


# ### Compute mortality

# In[ ]:


# Mortality(%) = Deaths / Cases
df_mortality = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
for x in country_list:
    df_mortality[x] = 100 * df_deaths[x] / df_cases[x] 


# ### Compute growth rates

# In[ ]:


# Compute daily variation of confirmed and active cases, and deaths
df_cases_diff = pd.DataFrame(columns=df_cases.columns, index=df_cases.index)
df_active_diff = pd.DataFrame(columns=df_active.columns, index=df_active.index)
df_deaths_diff = pd.DataFrame(columns=df_deaths.columns, index=df_deaths.index)

for x in country_list:
    df_cases_diff[x] = df_cases[x].diff()
    df_active_diff[x] = df_active[x].diff()
    df_deaths_diff[x] = df_deaths[x].diff()
    
df_cases_diff.fillna(value=0, inplace=True)
df_active_diff.fillna(value=0, inplace=True)
df_deaths_diff.fillna(value=0, inplace=True)


# ### Remove outliers

# In[ ]:


# Confirmed cases and deaths are always growing, hence their derivatives must be positive or zero
df_cases_diff[df_cases_diff < 0] = 0
df_deaths_diff[df_deaths_diff < 0] = 0


# ## Descriptive statistics
# ---
# 
# Let's have a look at where each country is in its specific pandemic expansion. The following variables will be displayed for each of the 18 countries:
# - Confirmed cases, active cases, recovered and deaths
# - Mortality(%)
# - Growth rates: confirmed cases, active cases and deaths

# ### Evolution of covid-19 cases

# In[ ]:


# First batch of 9 countries: EVOLUTION of CASES (1 of 2)
fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0
for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(14)) 
  ax1[i,j].plot(df_cases.index, df_cases[x], color='navy', linewidth=2, label='Confirmed cases')
  ax1[i,j].plot(df_active.index, df_active[x], color='skyblue', linewidth=2, label='Active cases')
  ax1[i,j].plot(df_recovered.index, df_recovered[x], color='lime', linewidth=2, label='Recovered cases')
  ax1[i,j].plot(df_deaths.index, df_deaths[x], color='coral', linewidth=2, label='Deaths')  
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Evolution of covid-19 cases by country (1 of 2)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()


# In[ ]:


# Second batch of 9 countries: EVOLUTION of CASES (2 of 2)
fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0
for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(14)) 
  ax2[i,j].plot(df_cases.index, df_cases[x], color='navy', linewidth=2, label='Confirmed cases')
  ax2[i,j].plot(df_active.index, df_active[x], color='skyblue', linewidth=2, label='Active cases')
  ax2[i,j].plot(df_recovered.index, df_recovered[x], color='lime', linewidth=2, label='Recovered cases')
  ax2[i,j].plot(df_deaths.index, df_deaths[x], color='coral', linewidth=2, label='Deaths')  
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Evolution of covid-19 cases by country (2 of 2)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Mortality
# 
# Mortality is calculated as the number of deaths divided by the number of confirmed cases, and expressed as %. As both the number of deaths and confirmed cases vary with time, we can plot an "instant mortality". However, the mortality will only be known once the pandemic has been erradicated, so the total number of cases and deaths are known. We can assume then the true mortality value to be the last one in the series.

# In[ ]:


# Plot mortality curves

fig, ax = plt.subplots(1,2, figsize=(32,8))

# First set of countries
ax[0].set_title('Covid-19 mortality(%)  (7-day MA)', fontsize='x-large')
ax[0].xaxis.set_major_formatter(my_date_fmt)
ax[0].xaxis.set_major_locator(plt.MultipleLocator(14))

for x in clist1:
    ax[0].plot(df_mortality.index, df_mortality[x].rolling(window=7).mean(), linewidth=1.5, label=x)
    
ax[0].legend(loc='upper left', fontsize='large')

# Second set of countries
ax[1].set_title('Covid-19 mortality(%)  (7-day MA)', fontsize='x-large')
ax[1].xaxis.set_major_formatter(my_date_fmt)
ax[1].xaxis.set_major_locator(plt.MultipleLocator(14))

for x in clist2:
    ax[1].plot(df_mortality.index, df_mortality[x].rolling(window=7).mean(), linewidth=1.5, label=x)
    
ax[1].legend(loc='upper left', fontsize='large')
               
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Confirmed cases growth rate
# 
# The daily change of the number of confirmed cases is calculated by substracting the value of confirmed cases on day t-1 from the value on day t. It represents the rate of growth, that is, how quickly (or slowly) the number of detected cases is changing.

# In[ ]:


# First batch of 9 countries: DAILY VARIATION of CONFIRMED CASES (1 of 2)

fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax1[i,j].bar(df_cases_diff.index,  df_cases_diff[x], color='grey', alpha=0.2, label='Cases growth rate')
  ax1[i,j].plot(df_cases_diff.index,  df_cases_diff[x].rolling(window=7).mean(), color='indigo', linewidth=1.5, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Daily variation of covid-19 confirmed cases by country (1 of 2)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()


# In[ ]:


# Second batch of 9 countries: DAILY VARIATION of CONFIRMED CASES (2 of 2)

fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax2[i,j].bar(df_cases_diff.index,  df_cases_diff[x], color='grey', alpha=0.2, label='Cases growth rate')
  ax2[i,j].plot(df_cases_diff.index,  df_cases_diff[x].rolling(window=7).mean(), color='navy', linewidth=1.5, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Daily variation of covid-19 confirmed cases by country (2 of 2)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Active cases growth rate

# In[ ]:


# First batch of 9 countries: DAILY VARIATION of CONFIRMED CASES (1 of 2)

fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax1[i,j].bar(df_active_diff.index,  df_active_diff[x], color='grey', alpha=0.2, label='Active cases growth rate')
  ax1[i,j].plot(df_active_diff.index,  df_active_diff[x].rolling(window=7).mean(), color='skyblue', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='lower left', fontsize='large')    
fig1.suptitle('Daily variation of covid-19 active cases by country (1 of 2)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()


# In[ ]:


# Second batch of 9 countries: DAILY VARIATION of ACTIVE CASES (2 of 2)

fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax2[i,j].bar(df_active_diff.index,  df_active_diff[x], color='grey', alpha=0.2, label='Active cases growth rate')
  ax2[i,j].plot(df_active_diff.index,  df_active_diff[x].rolling(window=7).mean(), color='skyblue', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1
    
ax2[0,0].legend(loc='lower left', fontsize='large')
fig2.suptitle('Daily variation of covid-19 active cases by country (2 of 2)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Deaths growth rate
# 
# The daily variation of deaths (or growth rate) tells us how quickly the number of deaths due to covid-19 is increasing (or decreasing) 

# In[ ]:


# First batch of 9 countries: DAILY VARIATION of DEATHS (1 of 2)

fig1, ax1 = plt.subplots(3,3, figsize=(36,15))
fig1.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist1:
  ax1[i,j].set_title(x, fontsize='x-large')
  ax1[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax1[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax1[i,j].bar(df_deaths_diff.index,  df_deaths_diff[x], color='grey', alpha=0.2, label='Deaths growth rate')
  ax1[i,j].plot(df_deaths_diff.index,  df_deaths_diff[x].rolling(window=7).mean(), color='coral', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax1[0,0].legend(loc='upper left', fontsize='large')
fig1.suptitle('Daily variation of covid-19 deaths by country (1 of 2)', fontsize='xx-large')  
fig1.autofmt_xdate(rotation=45, ha='right')
plt.show()


# In[ ]:


# Second batch of 9 countries : DAILY VARIATION of DEATHS (2 of 2)

fig2, ax2 = plt.subplots(3,3, figsize=(36,15))
fig2.subplots_adjust(top=0.93)
i = 0
j = 0

for x in clist2:
  ax2[i,j].set_title(x, fontsize='x-large')
  ax2[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax2[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))  
  ax2[i,j].bar(df_deaths_diff.index,  df_deaths_diff[x], color='grey', alpha=0.2, label='Deaths growth rate')
  ax2[i,j].plot(df_deaths.index,  df_deaths_diff[x].rolling(window=7).mean(), color='coral', linewidth=2, label='7-day MA')
  if j<2:
    j = j + 1
  else:
    j = 0
    i = i + 1

ax2[0,0].legend(loc='upper left', fontsize='large')
fig2.suptitle('Daily variation of covid-19 deaths by country (2 of 2)', fontsize='xx-large')  
fig2.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ## Predictive models of confirmed cases and deaths
# ---
# 
# I will now build, train and evaluate two LSTM-based predictive models for the main cumulative variables.

# In[ ]:


#########################################################################
# Prediction model parameters (Confirmed cases and deaths)
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases)
NBR_TRAIN_SAMPLES = NBR_SAMPLES - NBR_PREDICTIONS
NBR_TEST_SAMPLES = NBR_SAMPLES - NBR_TRAIN_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 30

print('Prediction model parameters for confirmed cases and deaths')
print('..........................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_TEST_SAMPLES: ', NBR_TEST_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('NBR_FEATURES: ', NBR_FEATURES)
print('TIME_STEPS:', TIME_STEPS)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('..........................................................')


# ### Build, train and evaluate RNN for confirmed cases

# In[ ]:


# Process of CONFIRMED CASES 

# Split dataset into test and train subsets 
df_train_1 = df_cases.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_1 = df_cases.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc1 = MinMaxScaler(feature_range = (0, 1))
sc1.fit(df_train_1)
sc_df_train_1 = sc1.transform(df_train_1)
# sc_df_test = sc.transform(df_test)

# Prepare training sequences
X_train_1 = []
y_train_1 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_1.append(sc_df_train_1[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_1.append(sc_df_train_1[i, 0:NBR_FEATURES])
   
X_train_1, y_train_1 = np.array(X_train_1), np.array(y_train_1)
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], X_train_1.shape[1], NBR_FEATURES))


# In[ ]:


# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN1 = Sequential()

# Build layers: 3 LSTM layers with dropout
RNN1.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train_1.shape[1], NBR_FEATURES)))
RNN1.add(Dropout(0.25))
RNN1.add(LSTM(units = 256))
RNN1.add(Dropout(0.25))
# NBR_FEATURES output dense layer
RNN1.add(Dense(units = NBR_FEATURES, activation='relu'))

RNN1.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Compile the RNN\nRNN1.compile(optimizer = 'adam', loss = 'mean_squared_error')\n\n# Train the RNN\nRNN1.fit(X_train_1, y_train_1, epochs = EPOCHS, batch_size = BATCH_SIZE)")


# In[ ]:


# Use now the full dataframe to predict / evaluate the model
df_full_1 = df_cases.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_1 = sc1.transform(df_full_1)

X_test_1 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_1.append(df_full_1[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_1 = np.array(X_test_1)
X_test_1 = np.reshape(X_test_1, (X_test_1.shape[0], X_test_1.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_1 = RNN1.predict(X_test_1)
predicted_values_1 = sc1.inverse_transform(predicted_values_1)

i = 0
for x in country_list:
  df_test_1[x + '_Predicted'] = predicted_values_1[:,i]
  i = i+1


# In[ ]:


fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_1.index, df_train_1[x], color='navy', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_1.index, df_test_1[x], color='grey', linewidth=1.5, alpha=0.5, label='Test')
  ax[i,j].plot(df_test_1.index, df_test_1[x + '_Predicted'], color='navy', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of covid-19 cases vs. training and validation data', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Build, train and evaluate RNN for deaths

# In[ ]:


# Process of DEATHS 
# Split dataset into test and train subsets 
df_train_2 = df_deaths.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_2 = df_deaths.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc2 = MinMaxScaler(feature_range = (0, 1))
sc2.fit(df_train_2)
sc_df_train_2 = sc2.transform(df_train_2)

# Prepare training sequences
X_train_2 = []
y_train_2 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_2.append(sc_df_train_2[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_2.append(sc_df_train_2[i, 0:NBR_FEATURES])
   
X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], NBR_FEATURES))


# In[ ]:


# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN2 = Sequential()

# Build layers: 3 LSTM layers with dropout
RNN2.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train_2.shape[1], NBR_FEATURES)))
RNN2.add(Dropout(0.25))
RNN2.add(LSTM(units = 256))
RNN2.add(Dropout(0.25))
# NBR_FEATURES output dense layer
RNN2.add(Dense(units = NBR_FEATURES, activation='relu'))

RNN2.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Compile the RNN\nRNN2.compile(optimizer = 'adam', loss = 'mean_squared_error')\n\n# Train the RNN\nRNN2.fit(X_train_2, y_train_2, epochs = EPOCHS, batch_size = BATCH_SIZE)")


# In[ ]:


# Use now the full dataframe to predict / evaluate the model
df_full_2 = df_deaths.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_2 = sc2.transform(df_full_2)

X_test_2 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_2.append(df_full_2[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_2 = np.array(X_test_2)
X_test_2 = np.reshape(X_test_2, (X_test_2.shape[0], X_test_2.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_2 = RNN2.predict(X_test_2)
predicted_values_2 = sc2.inverse_transform(predicted_values_2)

i = 0
for x in country_list:
  df_test_2[x + '_Predicted'] = predicted_values_2[:,i]
  i = i+1


# In[ ]:


fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_2.index, df_train_2[x], color='coral', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_2.index, df_test_2[x], color='grey', linewidth=1.5, alpha=0.5, label='Test')
  ax[i,j].plot(df_test_2.index, df_test_2[x + '_Predicted'], color='coral', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of covid-19 deaths vs. training and validation data', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ## Future predictions of confirmed cases and deaths
# ---
# 
# The above predictons look quite impressive. However, it is neccessary to explain here that the predicted values are single-step predictions. 
# 
# The LSTM neural network works in such way that it predicts y(t) from X(t-1), X(t-2), .... X(t-TIMESTEPS). So I have trained it to do so with batches formed with data up to NBR_TRAIN_SAMPLES. But then the validation is done by using such batches with data after NBR_TRAIN_SAMPLES and up to NBR_PREDICTIONS - 1. 
# 
# Which is not bad. But what I really want is to make multi-step predictions. More precisely, I would like to predict 30 days ahead starting the day after today. I can do that by training the LSTM network with all data available (up to today) and then make recurring predictions retrofitting each new single-step prediction as an input for the next prediction.
# 
# Let's see this at work.

# In[ ]:


#########################################################################
# Future prediction model parameters for confirmed cases and deaths
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases)
NBR_TRAIN_SAMPLES = NBR_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 30

print('Future prediction model parameters for confirmed cases and deaths')
print('.................................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('TIME_STEPS:', TIME_STEPS)
print('NBR_FEATURES: ', NBR_FEATURES)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('.................................................................')


# ### Retrain confirmed cases RNN and make future predictions

# In[ ]:


# Use full dataset as train data - CONFIRMED CASES
df_train_1 = df_cases.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_1.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_1 = pd.DataFrame(index=ind, columns=df_train_1.columns)
df_pred_1.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc1 = MinMaxScaler(feature_range = (0, 1))
sc1.fit(df_train_1)
sc_df_train_1 = sc1.transform(df_train_1)

# Prepare training sequences
X_train_1 = []
y_train_1 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_1.append(sc_df_train_1[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_1.append(sc_df_train_1[i, 0:NBR_FEATURES])

X_train_1, y_train_1 = np.array(X_train_1), np.array(y_train_1)
X_train_1 = np.reshape(X_train_1, (X_train_1.shape[0], X_train_1.shape[1], NBR_FEATURES))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Will reuse RNN1 already defined and validated earlier\nRNN1.summary()\n\n# Retrain the RNN with all available data\nRNN1.fit(X_train_1, y_train_1, epochs = EPOCHS, batch_size = BATCH_SIZE)')


# In[ ]:


# Make predictions 
LSTM_predictions_scaled_1 = list()
batch = sc_df_train_1[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_1)):   
    LSTM_pred_1 = RNN1.predict(current_batch)[0]
    LSTM_predictions_scaled_1.append(LSTM_pred_1) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_1]],axis=1)
    
# Reverse downscaling
LSTM_predictions_1 = sc1.inverse_transform(LSTM_predictions_scaled_1)
df_pred_1 = pd.DataFrame(data=LSTM_predictions_1, index=df_pred_1.index, columns=df_pred_1.columns)


# In[ ]:


fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_1.index, df_train_1[x], color='navy', linewidth=2, label='Actual data')
  ax[i,j].plot(df_pred_1.index, df_pred_1[x], color='navy', linewidth=3, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of covid-19 confirmed cases by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ### Retrain deaths RNN and make future predictions

# In[ ]:


# Use full dataset as train data - DEATHS
df_train_2 = df_deaths.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_2.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_2 = pd.DataFrame(index=ind, columns=df_train_2.columns)
df_pred_2.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc2 = MinMaxScaler(feature_range = (0, 1))
sc2.fit(df_train_2)
sc_df_train_2 = sc2.transform(df_train_2)

# Prepare training sequences
X_train_2 = []
y_train_2 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_2.append(sc_df_train_2[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_2.append(sc_df_train_2[i, 0:NBR_FEATURES])

X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], NBR_FEATURES))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Will reuse RNN2 already defined and validated earlier\nRNN2.summary()\n\n# Retrain the RNN with all available data\nRNN2.fit(X_train_2, y_train_2, epochs = EPOCHS, batch_size = BATCH_SIZE)')


# In[ ]:


# Make predictions 
LSTM_predictions_scaled_2 = list()
batch = sc_df_train_2[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_2)):   
    LSTM_pred_2 = RNN2.predict(current_batch)[0]
    LSTM_predictions_scaled_2.append(LSTM_pred_2) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_2]],axis=1)
    
# Reverse downscaling
LSTM_predictions_2 = sc2.inverse_transform(LSTM_predictions_scaled_2)
df_pred_2 = pd.DataFrame(data=LSTM_predictions_2, index=df_pred_2.index, columns=df_pred_2.columns)


# In[ ]:


fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_2.index, df_train_2[x], color='coral', linewidth=2, label='Actual data')
  ax[i,j].plot(df_pred_2.index, df_pred_2[x], color='coral', linewidth=3, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='large')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0

fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of covid-19 deaths by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ## Predictive model of confirmed cases growth rate
# ---
# 
# Finally, I will implement and train a LSTM-based predicive model of the cases growth rate. 
# 
# I initially wanted to implement a future prediction model of active cases growth rate (in fact, earlier versions of this notebook did so) but time and experience has taught me that is a bad variable to predict. Countries like Spain, UK or Sweden do not report covid-19 recovered figures, which is a component of active cases, so any further work on the latter will carry any errors of the former.
# 
# So I decided to stick to the basics and do a predictive model of the number of cases growth rate instead. This is actually the very first variable reported and spoken about. When someone says in the TV news "there were 2500 new cases registered today" this is the value of the cases growth rate today. 

# In[ ]:


#########################################################################
# Prediction model parameters (Confirmed cases growth rate)
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases_diff)
NBR_TRAIN_SAMPLES = NBR_SAMPLES - NBR_PREDICTIONS
NBR_TEST_SAMPLES = NBR_SAMPLES - NBR_TRAIN_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 50

print('Prediction model parameters for confirmed cases growth rates')
print('............................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_TEST_SAMPLES: ', NBR_TEST_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('NBR_FEATURES: ', NBR_FEATURES)
print('TIME_STEPS:', TIME_STEPS)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('............................................................')


# ### Build, train and evaluate RNN for confirmed cases growth rate

# In[ ]:


# Process of CONFIRMED CASES GROWTH RATE data

# Split dataset into test and train subsets 
df_train_3 = df_cases_diff.iloc[0:NBR_TRAIN_SAMPLES, 0:NBR_FEATURES] 
df_test_3 = df_cases_diff.iloc[NBR_TRAIN_SAMPLES:, 0:NBR_FEATURES]

# Normalize test and train data (range: 0 - 1)
sc3 = MinMaxScaler(feature_range = (0, 1))
sc3.fit(df_train_3)
sc_df_train_3 = sc3.transform(df_train_3)

# Prepare training sequences
X_train_3 = []
y_train_3 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_3.append(sc_df_train_3[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_3.append(sc_df_train_3[i, 0:NBR_FEATURES])
   
X_train_3, y_train_3 = np.array(X_train_3), np.array(y_train_3)
X_train_3 = np.reshape(X_train_3, (X_train_3.shape[0], X_train_3.shape[1], NBR_FEATURES))


# In[ ]:


# Build the RNN, dropout helps prevent overfitting

# Initialize structure
RNN3 = Sequential()

# Build layers: 2 LSTM layers with dropout
RNN3.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train_3.shape[1], NBR_FEATURES)))
RNN3.add(Dropout(0.25))
RNN3.add(LSTM(units = 512))
RNN3.add(Dropout(0.25))
# NBR_FEATURES output dense layer
RNN3.add(Dense(units = NBR_FEATURES, activation='relu'))

RNN3.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Compile the RNN\nRNN3.compile(optimizer = 'adam', loss = 'mean_squared_error')\n\n# Retrain the RNN with all available data\nRNN3.fit(X_train_3, y_train_3, epochs = EPOCHS, batch_size = BATCH_SIZE)")


# In[ ]:


# Use now the full dataframe to predict / evaluate the model
df_full_3 = df_cases_diff.copy()

# Scale full dataset (use same scaler fitted with train data earlier)
df_full_3 = sc3.transform(df_full_3)

X_test_3 = []
for i in range(NBR_TRAIN_SAMPLES, NBR_SAMPLES):
    X_test_3.append(df_full_3[i-TIME_STEPS:i, 0:NBR_FEATURES])

X_test_3 = np.array(X_test_3)
X_test_3 = np.reshape(X_test_3, (X_test_3.shape[0], X_test_3.shape[1], NBR_FEATURES))

# Make predictions
predicted_values_3 = RNN3.predict(X_test_3)
predicted_values_3 = sc3.inverse_transform(predicted_values_3)

i = 0
for x in country_list:
  df_test_3[x + '_Predicted'] = predicted_values_3[:,i]
  i = i + 1


# In[ ]:


# Plot future predictions of the cases growth rate
fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_3.index, df_train_3[x], color='indigo', linewidth=1.5, label='Train')
  ax[i,j].plot(df_test_3.index, df_test_3[x], color='grey', linewidth=1.5, alpha=0.5, label='Test')
  ax[i,j].plot(df_test_3.index, df_test_3[x + '_Predicted'], color='indigo', linestyle=':', linewidth=2.5, label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='medium')
  if j<2: 
    j = j + 1
  else:
    i = i + 1
    j = 0


fig.suptitle(str(NBR_PREDICTIONS) + '-day prediction of the covid-19 cases growth rate by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# ## Future predictions of confirmed cases growth rate

# In[ ]:


#########################################################################
# Future prediction model parameters for confirmed cases growth rate
#########################################################################

# Number of features Xi (Countries)
NBR_FEATURES = len(country_list)

# Number of predictions (days)
NBR_PREDICTIONS = 30

# Size ot TRAIN and TEST samples
NBR_SAMPLES = len(df_cases_diff)
NBR_TRAIN_SAMPLES = NBR_SAMPLES

# Number of input steps [x(t-1), x(t-2), x(t-3)...] to predict an output y(t)
TIME_STEPS = 8

# Number of overlapping training sequences of TIME_STEPS
BATCH_SIZE = 8

# Number of training cycles
EPOCHS = 50

print('Future prediction model parameters for confirmed cases growth rate')
print('..................................................................')
print('NBR_SAMPLES: ', NBR_SAMPLES)
print('NBR_TRAIN_SAMPLES: ', NBR_TRAIN_SAMPLES)
print('NBR_PREDICTIONS: ', NBR_PREDICTIONS)
print()
print('TIME_STEPS:', TIME_STEPS)
print('NBR_FEATURES: ', NBR_FEATURES)
print('BATCH_SIZE: ', BATCH_SIZE)
print('EPOCHS: ', EPOCHS)
print('..................................................................')


# ### Retrain cases growth rate RNN and make future predictions

# In[ ]:


# Use full dataset as train data - CONFIRMED CASES GROWTH RATE
df_train_3 = df_cases_diff.copy()

# Create empty dataframe with NBR_PREDICTIONS samples
start_date = df_train_3.index[-1] + timedelta(days=1)
ind = pd.date_range(start_date, periods=NBR_PREDICTIONS, freq='D')
df_pred_3 = pd.DataFrame(index=ind, columns=df_train_3.columns)
df_pred_3.fillna(value=0, inplace=True)

# Normalize train data (range: 0 - 1)
sc3 = MinMaxScaler(feature_range = (0, 1))
sc3.fit(df_train_3)
sc_df_train_3 = sc3.transform(df_train_3)

# Prepare training sequences
X_train_3 = []
y_train_3 = []
for i in range(TIME_STEPS, NBR_TRAIN_SAMPLES):
    X_train_3.append(sc_df_train_3[i-TIME_STEPS:i, 0:NBR_FEATURES])
    y_train_3.append(sc_df_train_3[i, 0:NBR_FEATURES])

X_train_3, y_train_3 = np.array(X_train_3), np.array(y_train_3)
X_train_3 = np.reshape(X_train_3, (X_train_3.shape[0], X_train_3.shape[1], NBR_FEATURES))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Will reuse RNN3 already defined and validated earlier\nRNN3.summary()\n\n# Retrain the RNN with all available data\nRNN3.fit(X_train_3, y_train_3, epochs = EPOCHS, batch_size = BATCH_SIZE)')


# In[ ]:


# Make predictions 
LSTM_predictions_scaled_3 = list()
batch = sc_df_train_3[-TIME_STEPS:]
current_batch = batch.reshape((1, TIME_STEPS, NBR_FEATURES))

for i in range(len(df_pred_3)):   
    LSTM_pred_3 = RNN3.predict(current_batch)[0]
    LSTM_predictions_scaled_3.append(LSTM_pred_3) 
    current_batch = np.append(current_batch[:,1:,:],[[LSTM_pred_3]],axis=1)
    
# Reverse downscaling
LSTM_predictions_3 = sc3.inverse_transform(LSTM_predictions_scaled_3)
df_pred_3 = pd.DataFrame(data=LSTM_predictions_3, index=df_pred_3.index, columns=df_pred_3.columns)


# In[ ]:


fig, ax = plt.subplots(6,3, figsize=(36,30))
fig.subplots_adjust(top=0.95)
i = 0
j = 0

for x in country_list:
  ax[i,j].set_title(x, fontsize='x-large')
  ax[i,j].xaxis.set_major_formatter(my_date_fmt)
  ax[i,j].xaxis.set_major_locator(plt.MultipleLocator(14))
  ax[i,j].plot(df_train_3.index, df_train_3[x], color='indigo', linewidth=1.5, label='Actual data')
  ax[i,j].plot(df_pred_3.index, df_pred_3[x], color='indigo', linewidth=2.5, linestyle=':', label='Prediction')
  ax[i,j].legend(loc='upper left', fontsize='medium')
  if j<2:
    j = j + 1
  else:
    i = i + 1
    j = 0


fig.suptitle(str(NBR_PREDICTIONS) + '-day future prediction of the confirmed cases growth rate by country', fontsize='xx-large')  
fig.autofmt_xdate(rotation=45, ha='right')
plt.show()


# So this is it. 
# 
# Definitely, the LSTM neural network works pretty well for predicting timeseries, as we have seen in the charts. 
# 
# It is also true the coding effort is also significant, specially when compared to other libraries available like Facebook Prophet, but again the results are worth the effort. 
# 
# Infinite thanks to Kaggle GPU without which this would have been seriously painful.
# 
# From the pandemic perspective.... well, this is far from over. But at the same time I somewhat understand a bit better how the enemy moves..... and that gives me an edge to fight back.    
# 
# (End of noteboook)
