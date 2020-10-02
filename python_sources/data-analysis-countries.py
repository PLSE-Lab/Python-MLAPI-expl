#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import matplotlib as ml
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
ml.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


cont = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
print(cont.shape)
cont.head()


# In[ ]:


cont.info()


# In[ ]:


cont.describe()


# #### CHECKS TO PERFORM :
# ##### Quality :
# *  Missing values from feature 'Net migration' onwards.
# *  Climate column shows deviation from data. (float values instead of int in an ordinal column)
# *  All decimal points have been replaced wrongly by commas.
# *  Unit of measurement of Population density is unclear.
# *  There's a column - Other(%). Information imparted is unclear.
# *  Correct the data types of the erronous numerical columns.
# 
# ##### Tidiness :
# *  Rename columns for better usability.

# In[ ]:


type(cont['Infant mortality (per 1000 births)'].values[2])


# In[ ]:


# Creating copy
cnt = cont.copy()
cnt.head()


# #### COMPLETENESS ISSUES
# *  Missing values from feature 'Net migration' onwards.
# *  There's a column - Other(%). Information imparted is unclear.

# ##### DEFINE
# Replacing missing values in the columns with missing values with zero. But will have to handle the 'Climate' column later.
# ##### CODE

# In[ ]:


# Renaming columns for better usability
new_column_name = {'Area (sq. mi.)':'Area' , 'Pop. Density (per sq. mi.)':'Pop_density' , 
                  'Coastline (coast/area ratio)':'Coastline' , 
                  'Infant mortality (per 1000 births)':'Infant_mortality' , 'GDP ($ per capita)':'GDP_per_capita' ,
                  'Literacy (%)':'Literacy_percent' , 'Phones (per 1000)':'Phones_per_k' , 'Arable (%)':'Arable' ,
                   'Crops (%)':'Crops' ,'Other (%)':'Other'}
cnt = cnt.rename(columns = new_column_name )
cnt


# In[ ]:


cnt = cnt.fillna(0)


# ##### TEST

# In[ ]:


cnt.isnull().sum()


# In[ ]:


cnt.info()


# ##### TEST

# ##### DEFINE
# Removing 'Other' if it has no significance
# ##### CODE

# In[ ]:


'''plt.figure(figsize=(20,10))
sns.heatmap(cnt.corr(),annot=True)
plt.show()'''

# Can't test before tidiness issues are handled.


# ##### TEST

# In[ ]:


# Nothing to test, no changes made


# ##### TIDINESS ISSUES
# *  Climate column shows deviation from data. (float values instead of int in an ordinal column)
# *  All decimal points have been replaced wrongly by commas.
# *  Correct the data types of the erronous numerical columns.

# ##### DEFINE
# Replace all decimal values with decimal point instead of commas and change data type from str to float

# ##### CODE

# In[ ]:


def rectify(cols):
    for c in cols:
        cnt[c] = cnt[c].astype(str)
        new_data = []
        for val in cnt[c]:
            val = val.replace(',','.')
            val = float(val)
            new_data.append(val)

        cnt[c] = new_data

# Running on dataset
cols = cnt[['Pop_density' , 'Coastline' , 'Net migration' , 'Infant_mortality' , 
                   'Literacy_percent' , 'Phones_per_k' , 'Arable' , 'Crops' , 'Other' , 'Climate' , 'Birthrate' , 'Deathrate' , 'Agriculture' ,
                   'Industry' , 'Service']]
rectify(cols)


# ##### TEST

# In[ ]:


cnt.head()


# ##### DEFINE
# Convert the decimal values in 'Climate' column to integer values to match the feature type(Ordinal). Must be in one of the classes : {1,2,3,4,}

# ##### CODE

# In[ ]:


cnt.Climate.unique()


# In[ ]:


cnt['Climate'] = cnt['Climate'].astype('int')


# ##### TEST 1

# In[ ]:


cnt.Climate.unique()


# ### We have to handle the zero, because it is not a member of the label set. Replace with most frequent climate label. Though this is not a feasible solution and we cannot conclude without proper data.

# In[ ]:


cnt.Climate.value_counts().sort_values(ascending=False)


# In[ ]:


cnt.Climate.replace(0,2,inplace=True)


# ##### TEST 2

# In[ ]:


cnt.Climate.unique()


# ##### HANDLING THE ONE COMPLETEMESS ISSUE LEFT DUE TO TIDINESS ISSUES
# *  There's a column - Other(%). Information imparted is unclear.

# ##### DEFINE
# Check the realtion with other features. If relation/dependency observed, keep. Else, remove.

# ##### CODE

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(cnt.corr(),annot=True)
plt.show()


# #### We see that none of the features seem to show linear dependency/relation with the 'Other' feature, except 'Arable' and 'Crops'. So, maybe 'Other' is related to them and is a feature similar to them. 'Arable' is the most related to 'Other'. 'Other' is kept.

# ##### TEST
# Nothing to test. No changes made

# ### All noted anomalies handled.

# ##### FINAL CHECK

# In[ ]:


cont = cnt
cont.head()


# In[ ]:


cont.info()


# In[ ]:


cont.describe()


# In[ ]:


cont.isnull().sum()


# ## EXPLORATORY DATA ANALYSIS

# In[ ]:


cont.head()


# #### POPULATION V/S COUNTRY AND V/S REGION

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data = cnt.nlargest(20, 'Population'), x = 'Country', y = 'Population')
plt.title("TOP 20 MOST POPULATED COUNTRIES")
plt.show()

# Region
plt.figure(figsize=(20,10))
sns.barplot(data = cnt.nlargest(20, 'Population'), x = 'Region', y = 'Population')
plt.title("MOST POPULATED REGIONS")
plt.show()


# *  We observe that India has the 2nd highest population.
# *  USA accounts to the most of Northern American population.

# ##### INFANT MORTALITY, BIRTH RATE, DEATH RATE

# In[ ]:


# Group data together
hist_data = [cont['Infant_mortality'], cnt['Birthrate'], cnt['Deathrate']]

group_labels = ['Infant_mortality', 'Birth Rate', 'Death Rate']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    margin=dict(l=10, b=10))
fig.show()


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='INFANT MORTALITY', x=cont.nlargest(10, 'Population')['Country'], y=cont['Infant_mortality']),
    go.Bar(name='BIRTH RATE', x=cont.nlargest(10, 'Population')['Country'], y=cont['Birthrate']),
    go.Bar(name='DEATH RATE', x=cont.nlargest(10, 'Population')['Country'], y=cont['Deathrate'])
])

fig.update_layout(barmode='group')
fig.show()


# *  All measurements are in % (per cent)
# *  This is a comparison between the infant mortality rate, birth rate, death rate of the highest populated countries. 
# *  Pakistan has the highest infant mortality rate, followed by China. 
# *  It is shocking to see India still has such a low infant mortality rate.
# *  However, India has a lower death rate compared to China and Pakistan. 
# *  Birth rate seems fairly normal.
# *  China and Pakistan have an alarmigly high birth ad death rates.

# In[ ]:


trace1 = go.Scatter(
    x = cont.index,
    y = cont.Deathrate,
    mode = 'lines+markers',
    name = 'Death Rate',
    marker = dict(color = 'rgba(255, 81, 51, 0.5)'),
    text = cont.Country)

trace2 = go.Scatter(
    x = cont.index,
    y = cont.Birthrate,
    mode = 'lines+markers',
    name = 'Birth Rate',
    marker = dict(color = 'rgba(105, 100, 255, 0.5)'),
    text = cont.Country)

layout = dict(title = 'Birth Rate v/s Death Rate of Countries',
             xaxis= dict(zeroline= False)
             )

data = [trace1, trace2]

fig = dict(data = data, layout = layout)

iplot(fig)


# *  The Birth Rate is, in general, higher for most countries.
# *  A few like like, Bostwana, Bulgaria, Estonia etc show deviation.
# *  Countries with higher Death rate -

# In[ ]:


cont[cont.Deathrate > cont.Birthrate]


# In[ ]:


perc = cont[cont.Deathrate > cont.Birthrate].shape[0]
perc


# In[ ]:


# PERCENTAGE OF COUNTRIES WITH HIGHER DEATH RATE
fig = go.Figure(data=[go.Pie(labels=['Death Rate > Birth Rate','Birth Rate > Death Rate'], values=[perc,(cont.shape[0]-perc)])])
fig.show()


# *  Only 11%

# ##### AGRICULTURE, INDUSTRY, SERVICES : HOW WELL ARE THEY IN THE SO CALLED DEVELOPED COUNTRIES ?
# Note : All measurements are in % (per cent)

# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Agriculture')
plt.title("ANALYSIS OF AGRICULTURE IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Industry')
plt.title("ANALYSIS OF INDUSTRY IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(data = cont.nlargest(10, 'GDP_per_capita'), x = 'Country', y = 'Service')
plt.title("ANALYSIS OF SERVICE IN TOP 10 COUNTRIES WITH THE HIGHEST GDP_per_capita")
plt.show()


# *  San Marino looks confusing, and strikes as odd having no revenue at all from any of the sectors, yet being a high GDP country.
# *  Almost all countries have a fairly high revenue from Service, information goes hand in hand with their GDP.
# *  Iceland achieves the highest revenue from Agriculture(very obvious and expected).
# *  Investigating San Marino - 

# In[ ]:


cnt[cnt['Country'] == "San Marino "]


# #### Since data for San Marino under all 3 sectors is 0, there are two possibilities :
# 1.  Data is really 0 - Highly impossible. Because we know it is a high GDP country, information is contradictory.
# 2.  Data was missing - Most possible scenario.
# 
# ##### SECTOR WISE CONTRIBUTION OF TOP 30 HIGH GDP COUNTRIES

# In[ ]:


cont_gdp_sorted = pd.DataFrame(cont.sort_values(ascending=False,by=['GDP_per_capita']))
cont_gdp = cont_gdp_sorted.nlargest(30,'GDP_per_capita')

trace0 = go.Bar(
    x = cont_gdp.Country,
    y = cont_gdp['Agriculture'],
    name = "Agriculture",
    marker = dict(color = 'rgba(255, 26, 26, 0.5)',
                    line=dict(color='rgb(100,100,100)',width=3)))

trace1 = go.Bar(
    x = cont_gdp.Country,
    y = cont_gdp['Industry'],
    name = "Industry",
    marker = dict(color = 'rgba(255, 255, 51, 0.5)',
                line=dict(color='rgb(100,100,100)',width=3)))

trace2 = go.Bar(
    x = cont_gdp.Country,
    y = cont_gdp['Service'],
    name = "Service",
    marker = dict(color = 'rgba(77, 77, 255, 0.5)',
                    line=dict(color='rgb(100,100,100)',width=3)))

data = [trace0, trace1, trace2]
layout = go.Layout(barmode = "stack")
fig = go.Figure(data = data,layout = layout)
iplot(fig)


# ##### CONCLUSION :
# *  WE SEE THAT FOR MOST COUNTRIES, THE MAJOR THRIVING SECTOR IS SERVICE FOLLOWED BY INDUSTRY. 
#     -  Except for UAE, which has Industry as its major sector.
# *  SAN MARINO AS WE KNOW, HAS NO DATA FOR THESE SECTORS, BUT ITS PRESENCE SHOWS THAT IT IS A HIGH GDP COUNTRY
# *  MONACO HAS NO RECORDED DATA FOR ANY SECTOR OTHER THAN AGRICULTURE. THERE CAN BE CONCLUSIONS :
#     -  Monaco has only agriculture as its major sector
#     -  Data has not been recorded for Monaco
#     
# ##### CORRELATION OF GDP_PER_CAPITA WITH AGRICULTURE, INDUSTRY AND SERVICE SECTORS : HOW MUCH RELATED ARE THEY WITH GDP ?

# In[ ]:


sns.jointplot(x="GDP_per_capita", y="Agriculture", data=cont, height=10, ratio=3, color="g")
plt.show()


# ##### AGRICULTURE HAS MODERATE NEGATIVE CORRELATION WITH GDP

# In[ ]:


sns.jointplot(x="GDP_per_capita", y="Industry", data=cont, height=10, ratio=3, color="y")
plt.show()


# In[ ]:


sns.jointplot(x="GDP_per_capita", y="Service", data=cont, height=10, ratio=3, color="maroon")
plt.show()


# ##### INDUSTRY AND SERVICE, BEING THE IMPORTANT SECTORS, YET THEY SHOW POOR CORRELATION WITH GDP.
# ##### ANALYSING THE EFFECT OF LITERACY RATE
# ##### LITERACY RATE TREND : 20 HIGHEST AND 20 LOWEST LITERACY RATES 

# In[ ]:


cont_lit_sorted = pd.DataFrame(cont.sort_values(ascending=False,by=['Literacy_percent'])).head(20)

fig = go.Figure([go.Bar(x=cont_lit_sorted.Country, y=cont_lit_sorted.Literacy_percent)])
fig.update_traces(marker_color='rgb(225,140,160)', marker_line_color='rgb(110,48,10)',
                  marker_line_width=1.5)
fig.show()


# In[ ]:


cont_lit_sorted


# ##### THIS SEEMS CONFUSING. UPON EXPLORING THE DATA, IT WAS NOTED THAT MOST OF THESE COUNTRIES HAVE LITERACY RATE > 99

# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(cont_lit_sorted['Literacy_percent'])
plt.show()


# ##### RELATION BETWEEN POPULATION AND LITERACY
# ##### LITERACY RATES OF THE TOP 20 MOST POPULATED COUNTRIES

# In[ ]:


fig = go.Figure(data=[go.Bar(x=cont.nlargest(20,'Population')['Country'], y=cont.nlargest(20,'Population')['Literacy_percent'])])
fig.update_layout(title_text='LITERACY RATES OF 20 MOST POPULATED COUNTRIES')
fig.show()


# ###### Relation between literacy rate and population.

# In[ ]:


sns.jointplot(x="Literacy_percent", y="Population", data=cont, height=10, ratio=3, color="g")
plt.show()


# ##### SHOCKINGLY ENOUGH, A HIGHER POPULATION DOESN'T GUARANTEE A HIGH LITERACY RATE

# ##### RELATION BETWEEN GDP AND LITERACY
# ##### LITERACY RATES OF THE TOP 20 HIGH GDP COUNTRIES

# In[ ]:


fig = go.Figure(data=[go.Bar(x=cont.nlargest(20,'GDP_per_capita')['Country'], y=cont.nlargest(20,'GDP_per_capita')['Literacy_percent'])])
fig.update_layout(title_text='LITERACY RATES OF 20 HIGH GDP COUNTIRES')
fig.show()


# ##### Relation between GDP and literacy

# In[ ]:


sns.jointplot(x="Literacy_percent", y="GDP_per_capita", data=cont, height=10, ratio=3, color="r")
plt.show()


# ##### LOW POSITIVE CORRELATION

# ### FINAL CONCLUSIONS :
# The data was analysed to find if the population, GDP etc of a country hold any specific dependence on factors like immortality rate, literacy, sectors of resource and revenue like agriculture, industry and service.
# 
# *  China is, as expected, the most populated country in the world, followed closely by India. This supports the next observation that, Asia is the most populated region.
# *  However, it shocking to find from analysis that some of the most populated countries(which happen to be developed/developing countries too) have a low Infant Immortality rate.
# *  However, analysis shows a positive result that asserts that only 11% of the countries around the world have Death Rates greater than Birth Rates.
# *  During analysis of the 3 major sectors of revenue, we saw an anomaly in the data for San Marino. It is impossible for a high GDP country(found from analysis) like San Marino, to have no contribution from the major sectors of revenue. So we concluded that :
#     -  The data is missing.
#     -  These types of data cannot be estimated as they are precise observations or collections.
#     -  We cannot remove the data for San Marino as it provides other vital information on Literacy rate, GDP_per_capita, Mortality rates etc.
#     -  The data is kept as it is.
# *  WE SEE THAT FOR MOST COUNTRIES, THE MAJOR THRIVING SECTOR IS SERVICE FOLLOWED BY INDUSTRY. 
#     -  Except for UAE, which has Industry as its major sector.
# *  SAN MARINO AS WE KNOW, HAS NO DATA FOR THESE SECTORS, BUT ITS PRESENCE SHOWS THAT IT IS A HIGH GDP COUNTRY
# *  MONACO HAS NO RECORDED DATA FOR ANY SECTOR OTHER THAN AGRICULTURE. THERE CAN BE CONCLUSIONS :
#     -  Monaco has only agriculture as its major sector
#     -  Data has not been recorded for Monaco
# *  The highest literacy rate obtained as per the data is 100%. Countries corresponding to this observation are : 
#     -  Australia
#     -  Liechtenstein
#     -  Andorra
#     -  Norway
#     -  Luxembourg
#     -  Denmark
#     -  Finland
# *  High GDP and high population doesn't mean a high literacy rate. According to analysis, they are lowly correlated.
# 
# THANK YOU :)

# In[ ]:




