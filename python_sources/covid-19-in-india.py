#!/usr/bin/env python
# coding: utf-8

# # COVID-19 in INDIA

# COVID-19 is an infectious disease caused by the Corona Virus, biologically known as severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The disease was first identified in Wuhan, the capital of China's Hubei province in December 2019 and has spread all over the world since then. As of writing this, on 26th April 2020, 21:57 IST, there are 2.92 million confirmed cases throughout the world and has resulted in 204,000 deaths according to Google.
# 
# In this notebook, I will take a look at the current situation in India. We will take a look at the regions which are most hampered by the outbreak and how numbers have steadily climbed in the country
# 
# To start off, we will import the necessary libraries which I will be using in my analysis and the different data tables from where I sourced my information.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


covid19_df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
individuals_df = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")


# In this analysis, I will use two of the datasets submitted by the collaborators. Much thank you to all those who helped in putting up this dataset. Let us have a look at some of the records of the covid19 dataset.

# In[ ]:


covid19_df.head()


# In[ ]:


covid19_df.tail()


# So we can see that the dataset provides a day by day record of the number of cases found in a specific state in the country. On further inspecting, we find that this dataset contains 1318 entries and contains 9 features. These are as seen contains some vital data like the number of confirmed cases, deaths, cured people till a specific day in a specific state. The Confirmed Cases are further broken down into Indian Nationals and Foreigners. The level of detail in this dataset is something which I loved the most !

# In[ ]:


covid19_df.shape


# In[ ]:


covid19_df.isna().sum()


# Here, we see that there are no missing values in this dataset which makes my job more easier. Let us now have a look at the most recent records for each state to gain an idea about where we stand currently. From the last set of records, we can see that we have data till 26th April 2020.

# In[ ]:


covid19_df_latest = covid19_df[covid19_df['Date']=="26/04/20"]
covid19_df_latest.head()


# In[ ]:


covid19_df_latest['Confirmed'].sum()


# So now we have filtered the dataset of 1318 records on the basis of the most recent data for every state. On inspecting this data, we see that India has a total of 26,605 cases till 26th April 2020.

# <h1> STATEWISE FIGURES </h1>

# In[ ]:


covid19_df_latest = covid19_df_latest.sort_values(by=['Confirmed'], ascending = False)
plt.figure(figsize=(12,8), dpi=80)
plt.bar(covid19_df_latest['State/UnionTerritory'][:5], covid19_df_latest['Confirmed'][:5],
        align='center',color='lightgrey')
plt.ylabel('Number of Confirmed Cases', size = 12)
plt.title('States with maximum confirmed cases', size = 16)
plt.show()


# On inspecting the above visualization, we see that Maharashtra has the most number of inspected cases as of now. Maharashtra is almost touching 8000 cases. The situation in Maharashtra is so grave that no other state in India has crossed even half that mark as per the data we have. Gujarat and Delhi are about to touch the 3000 mark whereas Rajasthan and Madhya Pradesh have crossed over 2000 cases.

# In[ ]:


covid19_df_latest['Deaths'].sum()


# As per the data in the dataset, India has had 826 deaths across all states. We will now see which states have the most deaths.

# In[ ]:


covid19_df_latest = covid19_df_latest.sort_values(by=['Deaths'], ascending = False)
plt.figure(figsize=(12,8), dpi=80)
plt.bar(covid19_df_latest['State/UnionTerritory'][:5], covid19_df_latest['Deaths'][:5], align='center',color='lightgrey')
plt.ylabel('Number of Deaths', size = 12)
plt.title('States with maximum deaths', size = 16)
plt.show()


# It is hardly surprising that all of the five states which figured in the former graph appear in this graph as well. Maharashtra currently account for almost 40% of the deaths in India due to COVID-19. Second placed Gujarat has not reached the halfway mark here as well. Madhya Pradesh is almost about to reach the three-figure mark with Delhi and Andhra Pradesh following on.
# 
# Next up, I wanted to look at the number of deaths per confirmed cases in different Indian states to gain a better idea about the healthcare facilities available.

# In[ ]:


covid19_df_latest['Deaths/Confirmed Cases'] = (covid19_df_latest['Confirmed']/covid19_df_latest['Deaths']).round(2)
covid19_df_latest['Deaths/Confirmed Cases'] = [np.nan if x == float("inf") else x for x in covid19_df_latest['Deaths/Confirmed Cases']]
covid19_df_latest = covid19_df_latest.sort_values(by=['Deaths/Confirmed Cases'], ascending=True, na_position='last')
covid19_df_latest.iloc[:10]


# So after creating this new measure and sorting the states based on this figure, I look at the ten worst states in this regard. We see that there are some states like Meghalaya, Jharkhand and Assam where the number of cases and deaths are pretty low as of now and it appears things are in control. But other states like Punjab, Karnataka look well hit by the condition. We leave West Bengal out of the entire equation since there has been news emerging from the state regarding mispublishing of numbers. Madhya Pradesh, Gujarat and Maharashtra also find themselves here in this list.
# 

# <h1> INDIVIDUAL DATA </h1>
# Next up, we have a look at the individual case data which we have. On initial inspection of this dataset, we see that there are a huge number of missing data in this dataset which we must take into consideration as we move forward with our analysis.

# In[ ]:


individuals_df.isna().sum()


# In[ ]:


individuals_df.iloc[0]


# The first case in India due to COVID-19 was noticed on 30th January 2020. It was detected in the city of Thrissur in Kerala. The individual had a travel history in Wuhan.

# In[ ]:


individuals_grouped_district = individuals_df.groupby('detected_district')
individuals_grouped_district = individuals_grouped_district['id']
individuals_grouped_district.columns = ['count']
individuals_grouped_district.count().sort_values(ascending=False).head()


# Next up, I decided to group the individual data in terms of district where the case was found. I had to be extra careful while doing this since there were some missing data in this column. From the data which was available, Mumbai is the worst hit district in the country. It has more than 2000 cases followed by Ahmedabad. Pune is another district in Maharashtra which figures in this list. All these districts belong to states which we had seen in the earlier graphs as well.

# In[ ]:


individuals_grouped_gender = individuals_df.groupby('gender')
individuals_grouped_gender = pd.DataFrame(individuals_grouped_gender.size().reset_index(name = "count"))
individuals_grouped_gender.head()

plt.figure(figsize=(10,6), dpi=80)
barlist = plt.bar(individuals_grouped_gender['gender'], individuals_grouped_gender['count'], align = 'center', color='grey', alpha=0.3)
barlist[1].set_color('r')
plt.ylabel('Count', size=12)
plt.title('Count on the basis of gender', size=16)
plt.show()


# Continuing our analysis, I thought about looking at how the case count is distributed according to gender. We see that there is no parity in this distribution. From the data, it seems that the virus is affecting males more than females in India. This is also validated by news article. You can read <a href = 'https://www.theguardian.com/commentisfree/2020/apr/07/coronavirus-hits-men-harder-evidence-risk'> this</a>.

# <h1> PROGRESSION OF CASE COUNT IN INDIA </h1>
# In this section, we will have a look at how the number of cases increased in India. Afterwards, we will inspect this curve and find similarities with the state-level curves.
# 
# For doing this analysis, I had to modify the dataset a bit. I grouped the data on the basis of the diagnosed data feature so that I had a count of number of cases detected each day throughout India. I followed this up by doing a cumulative sum of this feature and adding it to a new column.

# In[ ]:


individuals_grouped_date = individuals_df.groupby('diagnosed_date')
individuals_grouped_date = pd.DataFrame(individuals_grouped_date.size().reset_index(name = "count"))
individuals_grouped_date[['Day','Month','Year']] = individuals_grouped_date.diagnosed_date.apply( 
   lambda x: pd.Series(str(x).split("/")))
individuals_grouped_date.sort_values(by=['Year','Month','Day'], inplace = True, ascending = True)
individuals_grouped_date.reset_index(inplace = True)
individuals_grouped_date['Cumulative Count'] = individuals_grouped_date['count'].cumsum()
individuals_grouped_date = individuals_grouped_date.drop(['index', 'Day', 'Month', 'Year'], axis = 1)
individuals_grouped_date.head()


# In[ ]:


individuals_grouped_date.tail()


# This dataset contained data till the 20th of April. On that day, India had a total of 18,032 confirmed cases. We notice that the dataset contains data from 30th January but does not contain data in between since no cases were detected in that period. For the sake of continuity, I decided to assume 2nd March 2020 as Day 1 since we have data for every day since then.

# In[ ]:


individuals_grouped_date = individuals_grouped_date.iloc[3:]
individuals_grouped_date.reset_index(inplace = True)
individuals_grouped_date.columns = ['Day Number', 'diagnosed_date', 'count', 'Cumulative Count']
individuals_grouped_date['Day Number'] = individuals_grouped_date['Day Number'] - 2
individuals_grouped_date

plt.figure(figsize=(12,8), dpi=80)
plt.plot(individuals_grouped_date['Day Number'], individuals_grouped_date['Cumulative Count'], color="grey", alpha = 0.5)
plt.xlabel('Number of Days', size = 12)
plt.ylabel('Number of Cases', size = 12)
plt.title('How the case count increased in India', size=16)
plt.show()


# In the above curve, we see that the rise was more or less steady till the 20th day mark. In the interval between 20-30, the curve inclined a little. This inclination gradually incremented and we see a steady and steep slope after 30-day mark with no signs of flattening. These are ominous indications.
# 
# In the next few code elements, I prepare and process the dataset to group the data in terms of different states. I used the following five states for this next analysis:
# <li> Maharashtra </li>
# <li> Kerala </li>
# <li> Delhi </li>
# <li> Rajasthan </li>
# <li> Gujarat </li>

# In[ ]:


covid19_maharashtra = covid19_df[covid19_df['State/UnionTerritory'] == "Maharashtra"]
covid19_maharashtra.head()
covid19_maharashtra.reset_index(inplace = True)
covid19_maharashtra = covid19_maharashtra.drop(['index', 'Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'],  axis = 1)
covid19_maharashtra.reset_index(inplace = True)
covid19_maharashtra.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
covid19_maharashtra['Day Count'] = covid19_maharashtra['Day Count'] + 8
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,8)],
                           "Date": ["0"+str(x)+"/03/20" for x in range(2,9)],
                           "State/UnionTerritory": ["Maharashtra"]*7,
                           "Deaths": [0]*7,
                           "Confirmed": [0]*7})
covid19_maharashtra = covid19_maharashtra.append(missing_values, ignore_index = True)
covid19_maharashtra = covid19_maharashtra.sort_values(by="Day Count", ascending = True)
covid19_maharashtra.reset_index(drop=True, inplace=True)
print(covid19_maharashtra.shape)
covid19_maharashtra.head()


# In[ ]:


covid19_kerala = covid19_df[covid19_df['State/UnionTerritory'] == "Kerala"]
covid19_kerala = covid19_kerala.iloc[32:]
covid19_kerala.reset_index(inplace = True)
covid19_kerala = covid19_kerala.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)
covid19_kerala.reset_index(inplace = True)
covid19_kerala.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
covid19_kerala['Day Count'] = covid19_kerala['Day Count'] + 1
print(covid19_kerala.shape)
covid19_kerala.head()


# In[ ]:


covid19_delhi = covid19_df[covid19_df['State/UnionTerritory'] == "Delhi"]
covid19_delhi.reset_index(inplace = True)
covid19_delhi = covid19_delhi.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)
covid19_delhi.reset_index(inplace = True)
covid19_delhi.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
covid19_delhi['Day Count'] = covid19_delhi['Day Count'] + 1
print(covid19_delhi.shape)
covid19_delhi.head()


# In[ ]:


covid19_rajasthan = covid19_df[covid19_df['State/UnionTerritory'] == "Rajasthan"]
covid19_rajasthan.reset_index(inplace = True)
covid19_rajasthan = covid19_rajasthan.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)
covid19_rajasthan.reset_index(inplace = True)
covid19_rajasthan.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
covid19_rajasthan['Day Count'] = covid19_rajasthan['Day Count'] + 2
missing_values = pd.DataFrame({"Day Count": [1],
                           "Date": ["02/03/20"],
                           "State/UnionTerritory": ["Rajasthan"],
                           "Deaths": [0],
                           "Confirmed": [0]})
covid19_rajasthan = covid19_rajasthan.append(missing_values, ignore_index = True)
covid19_rajasthan = covid19_rajasthan.sort_values(by="Day Count", ascending = True)
covid19_rajasthan.reset_index(drop=True, inplace=True)
print(covid19_rajasthan.shape)
covid19_rajasthan.head()


# In[ ]:


covid19_gujarat = covid19_df[covid19_df['State/UnionTerritory'] == "Gujarat"]
covid19_gujarat.reset_index(inplace = True)
covid19_gujarat = covid19_gujarat.drop(['index','Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational','Cured'], axis = 1)
covid19_gujarat.reset_index(inplace = True)
covid19_gujarat.columns = ['Day Count', 'Date', 'State/UnionTerritory', 'Deaths', 'Confirmed']
covid19_gujarat['Day Count'] = covid19_gujarat['Day Count'] + 19
missing_values = pd.DataFrame({"Day Count": [x for x in range(1,19)],
                           "Date": [("0" + str(x) if x < 10 else str(x))+"/03/20" for x in range(2,20)],
                           "State/UnionTerritory": ["Gujarat"]*18,
                           "Deaths": [0]*18,
                           "Confirmed": [0]*18})
covid19_gujarat = covid19_gujarat.append(missing_values, ignore_index = True)
covid19_gujarat = covid19_gujarat.sort_values(by="Day Count", ascending = True)
covid19_gujarat.reset_index(drop=True, inplace=True)
print(covid19_gujarat.shape)
covid19_gujarat.head()


# All of the five states have 56 records with 5 features. Each record stands for every day. In this analysis as well, I decided to use 2nd March 2020 as Day 1 for the sake of consistency.
# 
# Let us now have a look at the visualization.

# In[ ]:


plt.figure(figsize=(12,8), dpi=80)
plt.plot(covid19_kerala['Day Count'], covid19_kerala['Confirmed'])
plt.plot(covid19_maharashtra['Day Count'], covid19_maharashtra['Confirmed'])
plt.plot(covid19_delhi['Day Count'], covid19_delhi['Confirmed'])
plt.plot(covid19_rajasthan['Day Count'], covid19_rajasthan['Confirmed'])
plt.plot(covid19_gujarat['Day Count'], covid19_gujarat['Confirmed'])
plt.legend(['Kerala', 'Maharashtra', 'Delhi', 'Rajasthan', 'Gujarat'], loc='upper left')
plt.xlabel('Day Count', size=12)
plt.ylabel('Confirmed Cases Count', size=12)
plt.title('Which states are flattening the curve ?', size = 16)
plt.show()


# We see almost all the curves follow the curve which is displayed by the nation as a whole. The only anomaly is that of Kerala. Kerala's curve saw the gradual incline in the period between 20-30 days as seen in other curves. But what Kerala managed to do was it did not let the curve incline further and manage to flatten the curve. As a result, the state has been able to contain the situation.
# 
# The situation in Maharashtra looks very grave indeed. The curve has had an immense steep incline and shows no signs of slowing down. Gujarat's curve steeped at a later time interval compared to the rest. It remained in control till the 30-day mark and the steep worsened after 40 days.
# 
# The only way we can as a whole prevent this impending crisis is by flattening the curve. All state governments needs to follow the Kerala model. It is the only state which managed to flatten the curve and hence, must have done most things right. It's time we followed the Kerala model.

# In[ ]:




