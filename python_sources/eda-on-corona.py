#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


seed = 42
random.seed(seed)


# # Corona Virus

# ![](http://../input/images/img1.jpg)

# Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans.  
# 
# Coronaviruses are zoonotic, meaning they are transmitted between animals and people.  Detailed investigations found that SARS-CoV was transmitted from civet cats to humans and MERS-CoV from dromedary camels to humans. Several known coronaviruses are circulating in animals that have not yet infected humans. 
# 
# Common signs of infection include respiratory symptoms, fever, cough, shortness of breath and breathing difficulties. In more severe cases, infection can cause pneumonia, severe acute respiratory syndrome, kidney failure and even death. 
# 
# Standard recommendations to prevent infection spread include regular hand washing, covering mouth and nose when coughing and sneezing, thoroughly cooking meat and eggs. Avoid close contact with anyone showing symptoms of respiratory illness such as coughing and sneezing.

# In this analysis and prediction we will analiys data provided by Johns Hopkins university. We will also build a predictive model to predict spread of this deadly virus

# # Importing Dataset

# In[ ]:


data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# # EDA

# <h3> Data Familiarization </h3>

# In[ ]:


data.head()


# In[ ]:


data.tail()


# Dataset is a time series data from 22 January 2020 to 22 February, It have 7 columns with a mix of categorical and numerical data

# ObservationDate - Date of the observation in MM/DD/YYYY<br>
# Province/State - Province or state of the observation (Could be empty when missing)<br>
# Country/Region - Country of observation<br>
# Last Update - Time in UTC at which the row is updated for the given province or country. (Not standardised and so please clean before using it)<br>
# Confirmed - Cumulative number of confirmed cases till that date<br>
# Deaths - Cumulative number of of deaths till that date<br>
# Recovered - Cumulative number of recovered cases till that date<br>

# In[ ]:


data.info()


# In[ ]:


data['Province/State'] = data['Province/State'].fillna('Unknown')


# <h3>Converting to datetime  <h3>

# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')


# In[ ]:


data['Last Update'] = pd.to_datetime(data['Last Update'])
data['Last Update'] = data['Last Update'].dt.strftime('%d/%m/%Y')


# In[ ]:


data.info()


# In[ ]:


# missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(10))


# Province/state column have lots of missing values, We can totally remove this column to analyis only country wise data or we can impute this value to preserve state feature for deeper analysis.

# In[ ]:


print([pd.value_counts(data[cols]) for cols in data.columns])


# Data was not recorded evenly on each date<br>
# Most cases of corona virus is in China followed by USA and Australia. Least cases are in Brazil, Ivory Coast and Mexico<br>

# In[ ]:


data.describe()


# Max confirmed cases are : 59989<br>
# Max Deaths are : 1789<br>
# Max recovered : 7862<br>
# <br>
# 

# In[ ]:


data[data.Confirmed > 500]


# <br>
# Data shows that China's Hubei region have huge amount of cases confirmed.
# <br>
# On some research over internet it was confirmed that Hubei region cases are not outliers rather true value. Here Hubei region data is not good for the model we might remove it later in model building phase because it will definetly interfere with our predictions.

# <h3>Data Visualization</h3>

# Data in number form is not very intuitive. It provides numerical insights like central tendency of data its variance and standard deviation but to fully understand data we need to visualize it on graph.<br>
# Visualizing data shows hidden patterns in data. These hidden patterns might provide us useful insight which can be helpfull to tackle this deadly virus.
# <br>
# <br>

# In[ ]:


data['Country'].value_counts().head(30).plot(kind='barh', figsize=(20, 6))


# <br>
# <br>Graph shows that most number of cases are in Mainland China<br>
# Data also shows that Top 5 countries have majority of cases of Corona virus
# <br>
# <br>

# In[ ]:


confirmed_plt = sns.relplot(x='Date', y='Confirmed',  data=data, kind='line',aspect=2, height=10, sort=False)
confirmed_plt.set_xticklabels(rotation=-45)


# <br>
# <br>
# Cases of Corona virus increased as we observed more data, it is clearly visble by slope of the line. Also notice as observation increases deveation in number of cases also increases, this huge deviation is due to Hubei region of china.
# There is sudden rise in number of cases 11 Feb and 13 Feb 2020. (I wonder what it would be?)
# <br>
# <br>

# <h3> Binning Data <h3>

# Binning data provides us an insight on overall patterns in data. Here we will bin confirmed, deaths, Recovery, Date and Last Update

# <h5> Binning Confirmed</h5>
# <br>
# Confirmed has an interesting take most of the cases in confirmed are below 12 in a day therefore we can bin cases of below 2 as low_cases, cases between 2 and 6 as medium_cases and cases above 12 as high_cases. This binning can be done according to any other criteria. As this virus is well contaminated anything above 2 cases per day should be more than controlled.

# In[ ]:


data['Confirmed_cases'] = data['Confirmed'].apply(lambda x: 'Low' if x < 3 else('Medium' if 3 <= x >=6 else 'High'))


# In[ ]:


data['Confirmed_cases'].value_counts()


# Here most of the cases detected in a day are between 3 and 6 which is good for this deadly virus.<br>
# Low cases are also quite high, but we need to increase ratio low cases to sucessfully contain the spread of virus.<br>
# High cases are not that prevalant which is a good news but it can be decreased.
# Note: Here increase in cases are considered as bad because as more confirmed cases are observed higher the risk of virus to the masses.<br>

# <h5> Binning deaths</h5>
# According to numerical analysis of deaths we can see 75% data have zero deaths with this much high deaths we will bin deaths in binary category of yes or no.

# In[ ]:


data['Deaths_Status'] = data['Deaths'].apply(lambda x: 'No' if x < 1 else 'Yes')


# In[ ]:


data['Deaths_Status'].value_counts()


# Here 1351 cases shows no deaths but in 368 days there are deaths this maybe due to accumlation of data and releasing it in one go.

# <h5> Binning Recovered</h5>
# According to numerical analysis of recovered we can see 50% data have zero recovery but 25% data have recovery of upto 7 persons in a day and 75% to 100% have recovery greater than 7, Hence we will bin data in 3 category. No_recovery for zero recovery datapoint, medium_recovery for 1 to 7 and high recovery for more than 7.<br>
# Again this categorization is used by me based on heuristics, better binning may increase information gain.

# In[ ]:


data['Recovered_Status'] = data['Recovered'].apply(lambda x: 'No' if x < 1 else('Medium' if 1 <= x >=7 else 'High'))


# In[ ]:


data['Recovered_Status'].value_counts()


# No recovery is pre-dominant in dataset which needs critical attention of medical professionals as well as researchers to find a cure and speedy recovery guides.<br>
# Although Medium and High recovery are there but in dataset maximum recovery is 7862 which is way lower than maximum cases confimed 59989

# In[ ]:


confirmed_death_reco_plt = sns.relplot(x='Date', y='Confirmed', hue='Deaths_Status', size='Recovered_Status', sizes=(50, 200),
                                       data=data, legend='brief', aspect=2)
confirmed_death_reco_plt.set_xticklabels(rotation=-45)


# As time passes confirmed cases, deaths and recovery of patients increasing. This shows as linear relationship between time and cases.
# <br>
# <br>

# Due to high voloume of people infected by virus in Hubei, China we are not able to analyis majority of trend. <br>
# We might consider removing data of Hubei or limiting number of confirmed cases.

# In[ ]:


data_no_hubei = data.drop(data[data['Province/State'] == 'Hubei'].index)


# In[ ]:


confirmed_death_no_hubei_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status', sizes=(50, 200),
                                       data=data_no_hubei, legend='brief', aspect=2)
confirmed_death_no_hubei_plt.set_xticklabels(rotation=-45)


# Above plot shows there are diffrent trends in data. We will see most of them closely<br>

# In[ ]:


data_conf_1000 = data.iloc[data[data.Confirmed.between(1000, 2000)].index, :]


# In[ ]:


confirmed_death_no_hubei_1000_plt = sns.relplot(x='Date', y='Confirmed', hue='Deaths_Status', size='Recovered_Status', row='Country',
                                             data=data_conf_1000, legend='brief', aspect=2, kind='line', sort=False)
confirmed_death_no_hubei_1000_plt.set_xticklabels(rotation=-45)


# There is sharp increase and decrease in deaths due to virus between 25 Jan 2020 to 7 Feb 2020, this maybe due to release of holded data about number of deaths in China. (Thanks to China's Censorship). There is high deviation in number of deaths recorded after 8 Feb 2020 and significant deviation after 15 Feb 2020.<br>
# No death is increasing gradually with a gentle slope after 15 Feb 2020, This needs to increased exponentially.<br>
# Recovered status is almost negligible which indicates dire need of a CURE!!!!

# In[ ]:


data_conf_800 = data.iloc[data[data.Confirmed.between(100, 500)].index, :]


# In[ ]:


confirmed_800_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Country', sizes= (20, 200), row='Deaths_Status',
                                             data=data_conf_800, legend='brief', aspect=2)
confirmed_800_plt.set_xticklabels(rotation=-45)


# Data shows that Majority of cases are from Mainland China, which is not a surprise as its origin is China.<br>
# Other countries have exponential amount of increase in confirmed cases after 13 Feb 2020.<br>
# There is moderate of recovery which is between 1 and 7. This is nothing compared to number of cases confirmed.

# In[ ]:


sns.countplot(x='Confirmed_cases', hue='Recovered_Status', data=data)


# In[ ]:


sns.countplot(x='Confirmed_cases', hue='Deaths_Status', data=data)


# In[ ]:


sns.countplot(x='Deaths_Status', hue='Recovered_Status', data=data)


# In[ ]:





# <h2> Country Wise Analysis </h2>

# insights provided overall analysis is not very clear as data from diffrent countries and states are making dataset very confusing. Here in this section we will analyis data on the basis of per country.

# <h4>Mainland China</h4>

# China has most number of cases of corona virus hence it is only natural to analyis china's situation in depth.

# In[ ]:


data_china = data.iloc[data[data.Country == 'Mainland China'].index, :]


# In[ ]:


data_china


# In[ ]:


print('There are ', len(data_china['Province/State'].value_counts()), 'Districts in China where virus is observed')


# In[ ]:


data_china['Province/State'].value_counts()


# Data observed in each district is 26 except for Tibet where it is 21.

# <h6>Visualizing China</h6>

# ![title](img/img2.png)

# <h6> Hubei District </h6>

# In[ ]:


china_date_plt = sns.relplot(x='Date', y='Confirmed', data=data_china, aspect=2.5, kind='line', sort=False)
china_date_plt.set_xticklabels(rotation=-45)


# Data of China is most similar to the overall data this shows that China's confirmed cases have heavy influence on dataset. It will not be intutive to analyse this data as same as overall because it will not reveal anything interesting but visualizing data as per district might reveal something interesting.<br>
# We will divide data in 2 parts on basis of district of Hubei as Hubei's record distorts the whole data due to its large number of cases.

# In[ ]:


data_china_hubei = data.iloc[data[data['Province/State'] == 'Hubei'].index, :]


# In[ ]:


china_hubei_plt = sns.relplot(x='Date', y='Confirmed', data=data_china_hubei, aspect=2.5,
                                 kind='line', sort=False)
china_hubei_plt.set_xticklabels(rotation=-45)


# This line shows gradual increase in number off reported cases in Hubei district with exponential growth during 11 abd 13 Feb 2020.<br>
# Hubei is the most hit region in the world by corona virus and this data tells us no diffrent. We need to analyis further.

# In[ ]:


china_hubei_plt1 = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status', sizes=(100, 20),
                               data=data_china_hubei, aspect=2.5)
china_hubei_plt1.set_xticklabels(rotation=-45)


# This zoomed in plot shows recovery is increasing as observation is increasing but number of death is also way to high daya by day. This is a matter of concern

# <h6> Other Districts </h6>

# In[ ]:


data_china_no_hubei = data_china.drop(data_china[data_china['Province/State'] == 'Hubei'].index)


# In[ ]:


china_no_hubei_plt = sns.relplot(x='Date', y='Confirmed', hue='Recovered_Status', size='Deaths_Status',
                               row='Province/State', data=data_china_no_hubei, aspect=2)
china_no_hubei_plt.set_xticklabels(rotation=-45)


# Most of the districts have cases less than 400 but a few districts have upward of 1200 cases.<br>
# There was no recovery in starting of the observation but once as more cases are observed recovery increased.<br>
# Death rates are still increasing in some districts and there are no death observed but overall death ratio is quite high

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




