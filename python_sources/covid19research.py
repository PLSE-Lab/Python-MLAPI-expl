#!/usr/bin/env python
# coding: utf-8

# # COVID-19: A STATISTICAL AND VISUAL ANALYSIS

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ##### To begin the project, let's first observe the spread of the pandemic in some countries like China, Italy, US, Spain, France and Germany that are believed to be the hotspots of the pandemic. This might give us an actual idea of how serious the outbreak is.  
# 
# For this, we will observe a plot that shows how, and at what rate the pandemic spread in these countries. 
# 
# But before we do that, let us see what different types of graphs are observed in the spread of a pandemic and also what the ideal graph for a country should look like so that its healthcare services should be able to efficiently handle the pandemic.
# 
# ![Pandemic Spread Graph (Source: CDC)](https://media.winnipegfreepress.com/images/cp-coronavirus-flattening-the-curve.png)

# In[ ]:


# Importing the dataset
perMilTestDF = pd.read_csv('/kaggle/input/uncover/UNCOVER/our_world_in_data/per-million-people-tests-conducted-vs-total-confirmed-cases-of-covid-19.csv')

#Grouping the data by country
countryDF = perMilTestDF.groupby(perMilTestDF['entity'])

height = 10
width = 10
plt.figure(figsize = (height, width))
ax = plt.subplot()

#PLotting data for India
x_india = list(range(1,87))
y_india = countryDF.get_group('India')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_india, y_india, 'r', label = 'India')

#Plotting data for USA
x_us = list(range(1,88))
y_us = countryDF.get_group('United States')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_us, y_us, 'g', label = 'USA')

#Plotting data for Italy
x_italy = list(range(1,88))
y_italy = countryDF.get_group('Italy')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_italy, y_italy, 'y', label = 'Italy')

#Plotting data for China
x_china = list(range(1,88))
y_china = countryDF.get_group('China')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_china, y_china, 'b', label = 'China')

#Plotting data for Spain
x_spain = list(range(1,88))
y_spain = countryDF.get_group('Spain')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_spain, y_spain, 'm', label = 'Spain')

#Plotting data for Germany
x_ger = list(range(1,88))
y_ger = countryDF.get_group('Germany')['total_confirmed_cases_of_covid_19_per_million_people_cases_per_million']
ax.plot(x_ger, y_ger, 'k', label = 'Germany')

plt.title('SPREAD OF PANDEMIC IN DIFFERENT COUNTRIES')
plt.xlabel('NUMBER OF DAYS SINCE OUTBREAK')
plt.ylabel('NUMBER OF CASES')
#plt.legend([line1, line2, line3, line4], ['India', 'USA', 'Italy', 'China'])
ax.legend()
plt.show()


# 
# ### In the above plot, it is clear that the European countries and the US failed to intervene on time. They didn't initiate lockdowns and social distancing measures on time, which resulted in a widespread community spread. 
# 
# #### *As a result, the pandemic spread in Europe and US at a rate much higher than what their medical services were prepared for.*
# 
# #### *On the other hand, countries like China and India were quick to realize the importance of social distancing. Measures were taken by the governments on time, resulting a relatively low rate of community spread.*

# # POPULATIONS AT RISK

# #### *First, we will observe how age of a person affects his chances/risk of contacting and getting infected from the COVID-19 virus.*

# In[ ]:


# Loading the dataset 
canada_age_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/howsmyflattening/canada-testing-data.csv')

# Clearing up the dataset
canada_age_df.replace('Not Reported', np.nan, inplace = True)
canada_age_df.dropna(subset = ['age'], axis=0, inplace = True)
canada_age_df['age'].replace({'<20':'10-19', '<18':'10-19', '<10':'0-9', '61':'60-69', '<1':'0-9', '2':'0-9', '50':'50-59'}, inplace = True)

# PLotting the data
ageVScovid = canada_age_df['age'].value_counts()
ageVScovid = ageVScovid[:]
plt.figure(figsize = (10,10))
sns.barplot(ageVScovid.index, ageVScovid.values)
plt.xlabel('AGE GROUPS')
plt.ylabel('NUMBER OF CASES')
plt.title('CASES OBSERVED IN DIFFERENT AGE GROUPS')
plt.show()
plt.close()


# ### After analysing the above graph, here's what one can deduce from the visualization-
# 
# * Age groups 50-59 are at most risk, followed by those in age groups 60-69.
# 
# * It is observed that the number of cases in children and teenagers is relatively low. A lot of factors affect this data. Generally, younger people tend to have a stronger immunity as compared to their elders. Another major reason can be timely lockdowns and declaration of holidays for schools and universities, which significantly reduced the chance of exposure of those in age group 0-19 to the virus. Also, in the northern hemisphere, summer vacations, and in the southern hemisphere, winter vacations were a reason that confined the younger population to their homes, thus reducing the danger for them.
# 
# * If we look at the data, the workforce of the population, i.e., people in age groups 20-60 were the worst affected. A major reason for this is that while lockdowns or preventive measures were being initiated, industries, supermarkets and shops were sill in operation. Governments of many countries, in the name of 'saving their economies' didn't pay much attention to the risk the adults were being exposed to. This is the main reason why this outbreak turned into a pandemic.
# 
# * Now, if we observe the data for senior citizens, i.e., people in age group 60+, we see that people in age group 60-79 were exposed to the virus on a larger scale as compared to those 80+. The reason here is that people aged 60-80 are more social as compared to those 80+. Most of them retiries, spread their time with friends/acquaintances. As a result, community spread was much more severe in case of this age group. On the contrary, people aged 80+ are usually confined to their homes/old-age(retirement) homes. They are comparatively less social. This is the reson the number of cases in the 80+ age group is relativelty lower.  

# ### Now, let us observe how the infection affected different sex, i.e., the male and the female populations.

# In[ ]:


# Loading the dataset
canada_sex_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/howsmyflattening/canada-testing-data.csv')

# Cleaning up the dataset
canada_sex_df.replace('Not Reported', np.nan, inplace = True)
canada_sex_df.dropna(subset = ['sex'], axis=0, inplace = True)

# PLotting the data
sexVScovid = canada_sex_df['sex'].value_counts()
sexVScovid = sexVScovid[:]
plt.figure(figsize = (5,5))
sns.barplot(sexVScovid.index, sexVScovid.values)
plt.xlabel('SEX')
plt.ylabel('NUMBER OF CASES')
plt.title('CASES OBSERVED (MALES vs FEMALES)')
plt.show()
plt.close()


# #### Upon analyzing the graph above, we can come to the following conclusion-
# > The number of cases in males and females are almost equal.
# 
# > However, if we observe the data more carefully, we will notice that the number of male patients is slightly higher than female patients. The reason for this can be a variety of factors.
# 
# > One such reason is that in some countries, males have a higher representation in the sex ratio. 
# 
# > Other reason can be that in a lot of third world countries, men are considered to be the working section of the society whereas women are expected to be stay-at-home housewives. As a result, men in those countries tend to have a higher exposure to the virus.

# ### Now, let us observe the following graph that analyzes the spread of the disease in the male and female population of different age groups.

# In[ ]:


# Importing the dataset
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/nextstrain/covid-19-genetic-phylogeny.csv')
df = df[['age','sex']]
df.replace({'?': np.nan, 'Unknown': np.nan, 'U': np.nan, 'unknwon': np.nan, 'FEmale': 'Female' }, inplace = True)
df.dropna(inplace = True)
df = df.astype({'age': 'float64', 'sex': 'category'})

# Binning the age data
bins = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
df['age_grp'] = pd.cut(df['age'], bins = bins, labels = ['0-20','20-40','40-60','60-80','80+'])

# Getting the number of male and female patients
grp = df.groupby(['age_grp'])
df = []
for key, item in grp:
    cases = item['sex'].value_counts().to_list()
    df.append([key, cases[0], cases[1]])
df = pd.DataFrame(df)
df.columns = ['age_grp', 'males', 'females']

# Plotting on a population graph
import plotly.graph_objs as go
import plotly.io as pio
           
women_bins = np.array(df['females'])*-1
men_bins = np.array(df['males'])

y = [10, 30, 50, 70, 90]

fig = dict(layout = go.Layout(yaxis=go.layout.YAxis(title='AGE GROUPS'),
                   xaxis=go.layout.XAxis(
                       range=[-800, 800],
                       tickvals=[-1000, -700, -300, 0, 300, 700, 1000],
                       ticktext=[1000, 700, 300, 0, 300, 700, 1000],
                       title='NUMBER OF PATIENTS'),
                   barmode='overlay',
                   bargap=0.1),

        data = [go.Bar(y=y,
               x=men_bins,
               orientation='h',
               name='Males',
               hoverinfo='x',
               marker=dict(color='#3283FE')
               ),
        go.Bar(y=y,
               x=women_bins,
               orientation='h',
               name='Females',
               text=-1 * women_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='#FE00FA')
               )]
            )

pio.show(fig)


# Here too, we notice that in each age group, there is almost an equal number of male and female patients, the male population of patients being slightly greater in each age group. 
# 
# #### Since the number of male and female patients are almost equal, this proves to an exten that the sex of a person doesn't play any significant role in determining the risk factor of them getting the infection.
# 
# \*\*One thing to be noted is that since there isn't enough data, making a stong prediction isn't possible.

# ### Now, let us analyze the relation between the age and sex of the patients and the mortality rate.

# In[ ]:


# Loading the dataset
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/individual-level-mortality.csv')

# Cleaning up the data
df.replace('Not Reported', np.nan, inplace = True)
df.dropna(subset = ['age'], axis=0, inplace = True)
df['age'].replace({'>70':'70-79','78':'70-79', '>50':'50-59', '>65':'60-69','61':'60-69', '82':'80-89','83':'80-89','>80':'80-89', '92':'90-99'}, inplace = True)

# Plotting the data
ageVSdeath = df['age'].value_counts()
ageVSdeath = ageVSdeath[:]

#Plotting bar graph 
plt.figure(figsize = (10,10))
sns.barplot(ageVSdeath.index, ageVSdeath.values)
plt.xlabel('AGE GROUPS')
plt.ylabel('NUMBER OF DEATHS')
plt.title('DEATHS OBSERVED IN DIFFERENT AGE GROUPS')
plt.show()
plt.close()

#PLotting pie chart
df['age'].replace({'70-79':'60+', '80-89':'60+', '90-99':'60+', '60-69':'60+', '50-59':'20-60', '40-49':'20-60', '100-109':'60+', '30-39':'20-60', '20-29':'20-60'}, inplace = True)
ageVSdeath = df['age'].value_counts()
ageVSdeath = ageVSdeath[:]
plt.figure(figsize = (10,10))
plt.pie(ageVSdeath.values, labels = ageVSdeath.index, startangle = 90)
plt.title('AGE vs DEATHS')
plt.show()
plt.close()


# ### Upon analyzing the graphs, we can come to the following deductions-
# 
# - Majority of the patient population that lost their lives belonged to the 60+ age group.
# 
# - Out of these, those in the age group 70-90 showed the highest mortality rate towards the virus. The reason here is the weakened immunity due to the old age of the patients. In the analysis of how age affected the spread of the disease, there too we noticed that those in age groups 60-80 caught the infection relatively easily.
# 
# - On the other hand, those in the the age group 20-40 showed a comparatively much lower mortality rate. Here too, most of the patients were either immuno-compromized, or had a medical history of some kind.

# # MOST DANGEROUS COUNTRIES TO LIVE IN DURING A PANDEMIC

# Based on the current performance by the governments of the respective countries in dealing with the COVID-19 pandemic, as well as factors like the population densities of the countries, availability of physicians and healthcare services etc., the following graph shows which countries are the most dangerous to live in the event of a global pandemic. 

# In[ ]:


# Loading the dataset
RiskData = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/inform-covid-indicators.csv')

# Plotting the graph on a geomap

fig = go.Figure(data=go.Choropleth(
    locations = RiskData['iso3'],
    z = RiskData['inform_risk'],
    text = RiskData['country'],
    colorscale='bluered_r',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'RISK</br></br>FACTOR'
))

fig.update_layout(
    title_text='MOST DANGEROUS COUNTRIES TO LIVE IN DURING A PANDEMIC- HUMAN DATA EXCHANGE',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='',
        showarrow = False
    )]
)

fig.show()


# #### The following map shows which countries took the most number of precautionary and preventive steps for their citizens. 
# 
# ** One thing to be notes is that taking a lot of steps doesn't mean they were actually succesful in dealing with the pandemic. It is just an experimental thing I wished to do. 

# In[ ]:


df = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv')
measures = df['country'].value_counts().to_frame()
measures = measures.reset_index()
measures.columns = ['country', 'total_govt_measures']

country = list(measures['country'])
countryDF = list(df['country'])
isoDF = list(df['iso'])

measures['iso'] = [isoDF[countryDF.index(i)] for i in country]

# Plotting the graph on a geomap

fig = go.Figure(data=go.Choropleth(
    locations = measures['iso'],
    z = measures['total_govt_measures'],
    text = measures['country'],
    colorscale='reds_r',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'TOTAL</br></br>MEASURES</br>TAKEN'
))

fig.update_layout(
    title_text='TOTAL MEASURES TAKEN BY DIFFERENT COUNTRIES OF THE WORLD',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='',
        showarrow = False
    )]
)

fig.show()

