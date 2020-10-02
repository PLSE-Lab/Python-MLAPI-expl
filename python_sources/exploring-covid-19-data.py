#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load data
covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")


# In[ ]:


# A look at the dataset
covid_19_data.head()


# In[ ]:


# A concise summary of dataset
covid_19_data.info()


# ### Understanding different features
# There are 8 features:
# * **SNo:** Serial number
# * **ObservationDate:** Date of observation of the cases (format: MM/DD/YYYY)
# * **Province/State:** Province or State of the country where cases were observed
# * **Country/Region:** Country where cases were observed
# * **Last Update:** Time in UTC at which the row is updated for the given province or country. (It is not in a standard format)
# * **Confirmed:** Cumulative number of confirmed cases till the date
# * **Deaths:** Cumulative number of deats till the date
# * **Recovered:** Cumulative number of recovered cases till date

# In[ ]:


# Set the SNo as index
covid_19_data.set_index('SNo', inplace = True)


# In[ ]:


# Checking the shape 
covid_19_data.shape


# Let's rename some of the columns:
# * Province/State ==> State
# * Country/Region ==> Country
# * Last Update ==> Last_Update

# In[ ]:


# Renaming columns
covid_19_data.rename(columns={'Province/State': 'State', 'Country/Region': 'Country', 'Last Update': 'Last_Update'}, 
                     inplace=True)


# In[ ]:


# Check the columns
covid_19_data.columns


# In[ ]:


# Converting 'ObservationDate' and 'Last_Update' to datetime
covid_19_data.ObservationDate = pd.DatetimeIndex(covid_19_data.ObservationDate)
covid_19_data.Last_Update = pd.DatetimeIndex(covid_19_data.Last_Update)


# In[ ]:


# Renaming 'Mainland China' to 'Chine'
covid_19_data.Country = covid_19_data.Country.apply(lambda x: 'China' if x == 'Mainland China' else x)


# In[ ]:


# List of all the countries infected
countries = covid_19_data.Country.unique()
print(countries)
print()
print('Total number of countries infected: ', len(countries))


# Total of 223 countries infected with corana virus.

# In[ ]:


covid_19_data.Last_Update.quantile(1)


# In[ ]:


# The data was updated last on 13 June 2020.
updated_data = covid_19_data[covid_19_data.Last_Update == covid_19_data.Last_Update.quantile(1)]


# In[ ]:


# Visualizing the confirmed, deaths and recovered cases per country for top 10 countries with confirmed cases
fig = updated_data.groupby('Country').sum().sort_values(by = 'Confirmed', ascending = False)[:10].plot(
                                                                                            kind = 'bar',
                                                                                            figsize = (16, 8))

plt.ylabel('Number of cases')
plt.title("Top 10 countries with confirmed cases")

# To diplay the the count on top of the bar
for p in fig.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    fig.annotate('{:.0}'.format(height), (x, y + height + 0.01))


# In[ ]:


# percentage confirmed cases 
conf_cases = updated_data.groupby('Country')['Confirmed'].sum().sort_values(ascending = False) / updated_data['Confirmed'].sum()


# In[ ]:


# Plotting the percentage of confirmed cases worldwide
def value(val):
    per = np.round(val , 2)
    return per
conf_cases.plot(kind = 'pie', figsize = (16, 12), autopct = value)
plt.title('Percentage of confirmed cases worldwide')
plt.show()


# **USA is leading with 26.8% of confirmed cases worlwide and then Brazil at second position with 10.8% confirmed cases worldwide.**
# 

# In[ ]:


# Getting top10 countries with highest number of confirmed cases
top_10 = updated_data[['Country', 'Confirmed','Recovered','Deaths']].groupby('Country').sum().sort_values(by = 'Confirmed', ascending = False)[:10]

# Recovery and Deaths percentage
top_10['Recovered_percentage'] = top_10['Recovered'] / top_10['Confirmed'] * 100
top_10['Deaths_percentage'] = top_10['Deaths'] / top_10['Confirmed'] * 100


# In[ ]:


top_10


# In[ ]:


top_10.Recovered_percentage.sort_values(ascending = False)


# * Germany is leading in the recovery percentage of confirmed corona cases.
# * The situation in UK is very poor with only 0.43 percentage of recovered corona cases

# In[ ]:


top_10.Deaths_percentage.sort_values(ascending=False)


# * Russia is at 3rd postion in terms of confirmed cases and at 5th position in terms of recovered percentage, however, the death percentage in Russia is very low i.e. only 1.3% which is a very good sign.
# * However, India is in top 5 in terms of confirmed cases and at 6th position in terms of rocovered percentage, the death percentage in india seems to be low (i.e. 2.8%) as compared to other countries.
# * UK has more deaths than recovered cases. This shows a very poor condition in UK.

# In[ ]:


top_10[['Recovered_percentage','Deaths_percentage']].plot(kind = 'bar', figsize = (16, 8))
plt.title("Recovered percentage and Deaths percentage of Top 10 Countries in terms of Confirmed Cases")
plt.ylabel("Percentage")
plt.show()


# ## USA With Highest Confirmed Cases Worldwide

# In[ ]:


# USA data
USA_data = updated_data[updated_data.Country == 'US']
USA_data_by_state = USA_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(
                                                                                by = 'Confirmed',
                                                                                ascending = False)

USA_day_wise = covid_19_data[covid_19_data.Country == 'US'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(
                                                                                by = 'ObservationDate',
                                                                                ascending = True)


# In[ ]:


USA_total_recovered = USA_data_by_state.loc['Recovered']
USA_data_by_state = USA_data_by_state.drop('Recovered', axis = 0)


# In[ ]:


USA_total_recovered


# In[ ]:


# For USA overall recovered is given. Statewise recovered cases are not given in the dataset.
USA_data_by_state.plot(kind = 'bar', figsize = (16, 4))
plt.title("Statewise Confirmed and Deaths cases of USA")
plt.ylabel('Number of Cases')
plt.show()


# In[ ]:


USA_day_wise[['Confirmed', 'Deaths']].plot(figsize = (8,4))
plt.title('Confirmed and Deaths rate of USA')
plt.ylabel('Number of Cases')
plt.show()


# * Confirmed cases in USA is increasing with high slope and this needs to be flatened.

# ## Germany With Highest Recovery Percentage

# In[ ]:


# Germany data
Germany_data = updated_data[updated_data.Country == 'Germany']
Germany_data_by_state = Germany_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(
                                                                                by = 'Confirmed',
                                                                                ascending = False)

Germany_day_wise = covid_19_data[covid_19_data.Country == 'Germany'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(
                                                                                by = 'ObservationDate',
                                                                                ascending = True)


# In[ ]:


Germany_data_by_state.plot(kind = 'bar', figsize = (12,6))
plt.title("Covid data cases of Germany statewise")
plt.ylabel('Number of Cases')
plt.show()


# In[ ]:


Germany_day_wise.plot(figsize = (16,6))
plt.title("Confirmed, Recovered and Deaths rates of Germany per day")
plt.ylabel("Number of cases")
plt.show()


# * It seems like the Confirmed case rate will be soon constant for Germany.

# ## Russia With Lowest Deaths Percentage Among 10 Countries

# In[ ]:


Russia_data = updated_data[updated_data.Country == 'Russia']
Russia_data_by_state = Russia_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(
                                                                                by = 'Confirmed',
                                                                                ascending = False)

Russia_day_wise = covid_19_data[covid_19_data.Country == 'Russia'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(
                                                                                by = 'ObservationDate',
                                                                                ascending = True)


# In[ ]:


Russia_data_by_state.plot(kind = 'bar', figsize = (15,6))
plt.title("Statewise Confirmed, Recovered and Deaths percentage of Russia")
plt.ylabel("Number of Cases")
plt.show()


# In[ ]:


Russia_day_wise.plot(figsize = (16, 6))
plt.title("Daywise Confirmed, Recovered and Deaths rates of Russia")
plt.ylabel('Number of Cases')
plt.show()


# * The death rate curve for Russia seems to be nearly constant to 0.

# ## India With Second Lowest Deaths Percentage Among 10 Countries

# In[ ]:


India_data = updated_data[updated_data.Country == 'India']
India_data_by_state = India_data[['State', 'Confirmed', 'Recovered', 'Deaths']].groupby('State').sum().sort_values(
                                                                                by = 'Confirmed',
                                                                                ascending = False)

India_day_wise = covid_19_data[covid_19_data.Country == 'India'][['ObservationDate', 'Confirmed', 'Recovered', 'Deaths']].groupby('ObservationDate').sum().sort_values(
                                                                                by = 'ObservationDate',
                                                                                ascending = True)


# In[ ]:


India_data_by_state.plot(kind = 'bar', figsize = (16, 6))
plt.title("Statewise Confirmed, Recovered and Deaths cases in India")
plt.ylabel("Number of Cases")
plt.show()


# * 50% cases in Maharashtra seems to be recovered.

# In[ ]:


India_day_wise.plot(figsize = (16, 6))
plt.title("Confirmed, Recovered and Deaths Rate in India per Day")
plt.ylabel("Number of Cases")
plt.show()


# In[ ]:




