#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates On Charts
# 

# ## Introduction
# The dataset was obtained on kaggle.
# *This dataset has stats of number of suicides commited in 101 countries from year 1986 to 2016.
# the dataset has information age groups, generations commiting suicide which is further categorised into gender i.e. "male" and "female".*

# ## Dataset
# 

# ### 1. **Features of dataset**

# 
# * **Country** : Country names, a total of 101 countries are considered in this dataset. 
# * **Year**: time line from year 1986 to 2016
# * **Gender**: sex(male/female)
# * **Age**: total of 6 categories based on Age.
# * **SuicidesNo**: Total number of suicides 
# * **Population**: Population of the resepective country
# * **Suicides100kPop**: number of suicides per 100k population.
# * **GdpForYear**: Gross Domestic Product.
# * **GdpPerCapital**:Obtained by dividing the GDP by the total population of the country for that year
# * **Generation**: total of 6 generation are included, based on age and country and gender.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0,10.0)
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### 2. Reading The Dataset csv file

# In[ ]:


suicide_data = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
suicide_data.head()


# In[ ]:


# Describing the dataset:
suicide_data.describe()


# ### 3. Renaming the columns. 
# 

# In[ ]:


suicide_data.columns


# In[ ]:


suicide_data = suicide_data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age','suicides_no':'SuicidesNo',
                          'population':'Population','suicides/100k pop':'Suicides100kPop','country-year':'CountryYear',
                          'HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYear',
                          'gdp_per_capita ($)':'GdpPerCapital','generation':'Generation'})


# In[ ]:


suicide_data.head()


# In[ ]:


suicide_data.columns


#  ### 4. Handling the missing values

# In[ ]:


suicide_data.isnull().sum()


# Dropping columns countryyear and HDI because HDI column is almost null and we dont need country-year column.
# 

# In[ ]:


suicide_data = suicide_data.drop(['HDIForYear', 'CountryYear'],axis=1)


# In[ ]:


suicide_data.head()


# ### 5. Data Visualization
# 

# **Plotting graph of countries with highest number of suicides**

# In[ ]:


data = suicide_data.groupby('Country').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)
data = data.head(15)
data


# In[ ]:


sb.barplot(data['SuicidesNo'],data.index, palette='Reds_r')


# A barplot Country Vs SuicideNo
# 
# Graph Insights:
# 1. The countries with highest number of suicides is Russia then comes USA, Japan, France.
# 2. Number of suicides in Russia and USA are more than 1 million each from 1986 to 2016.
# 3. Total number of suicide in Russia is equals to total number of suicides in France, Germany and Ukrine.

# **Graph of number of suicides commited with respect to gender**

# In[ ]:


data1 = suicide_data.groupby('Gender').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)
data1 = data1.head(15)


# In[ ]:


sb.barplot(data1.index, data1['SuicidesNo'], palette='Blues_r')


# A Barplot Gender versus SuicideNo
# 
# Graph Insights:
# 1. the highest number of suicides is dominated by males, males tend to commit suicide than females.
# 2. Number of suicides by Males and females is around 5 million and 1.5 million respectively.

# **Graph showing total suicides with respect to Age groups.**

# In[ ]:


#unique values
pd.unique(suicide_data['Age'])


# In[ ]:


data2 = suicide_data.groupby('Age').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)
data2


# In[ ]:


# Pie chart
labels = data2.index
sizes = data2['SuicidesNo']
#colors
colors = ['#ff6666','#ff9999','#66b3ff','#99ff99','#ffcc99','#ffcc60']
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# Pie chart to display suicide percentage rate in various Age groups
# 
# Graph Insights:
# 1. Highest number of suicides are commited by Age group 35-54 years and 55-74 years.
# 2. Lowest number of suicides : age group "75+" and "5-14" years.

# ** Total Number Suicides Yearwise**

# In[ ]:


suicide_data['Year'].unique()


# In[ ]:


data3 = suicide_data.groupby('Year').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)


# In[ ]:


plt.figure(figsize=(15,7))
sb.barplot(data3.index, data3['SuicidesNo'])
plt.title('Total number of Suicides from year 1985-2016 ')
plt.show()


# This plot represents total number of suicides over the years.People commiting suicides from 1986 to 2016.
# 
# Graph Insights:
# 1. years 1985,1986,1987,1988 had low suicide rates.
# 2. highest suicide rate was during 1998 to 2003.
# 3. 2016 has the least number of suicides.

# ### Suicide numbers generation wise.

# In[ ]:


data4 = suicide_data.groupby('Generation').agg({'SuicidesNo':'sum'}).sort_values(by='SuicidesNo', ascending = False)
data4


# In[ ]:


data5 = suicide_data.groupby('Generation').agg({'Suicides100kPop':'sum'}).sort_values(by='Suicides100kPop', ascending = True)
data5.head()


# In[ ]:


plt.figure(figsize=(10,9))
sb.barplot(data4.index,data4['SuicidesNo'], palette = 'Blues_d')# color = '#ff9449')
#sb.barplot(data5.index,data5['Suicides100kPop'])
plt.plot()
plt.show()


# Above plot shows number of suicide death with repect to generation.(Highest to lowest).
# 
# Insights from graph:
# 1. highest number of sucides are by boomers generation and then comes 'silent'.
# 2. Generation Z barely commit suicides.
# 

# In[ ]:


data6 = suicide_data.groupby('Generation').agg({'Population':'sum'}).sort_values(by='Population', ascending = True)
data6 
data6['suicides'] = data5.values
data6['suicide_percentage'] = (data6['suicides']/ data6['Population'])*100
data6.sort_values(by='suicide_percentage', ascending = False)


# * The above table displays total population for each generation followed by total number of suicides.
# * we can calculate suicide percentage of each generation,
# * (suicides / population) * 100
# * The sucide percentage shows that generation Z has the highest percentage of suicide deaths.
# * G.I generation with the least suicide percentage.

# **Population Of Each Generation**

# In[ ]:


# Pie chart
labels = data6.index
sizes = data6['Population']
#colors
colors = ['#ff6666','#ff9999','#66b3ff','#99ff99','#ffcc99','#ffcc60']
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()


# In[ ]:


sb.countplot(x='Generation',hue ="Gender",
                 data=suicide_data)
plt.xticks(rotation=45)
plt.title('Generations vs Gender count')
plt.show()


# **Pairplots** 

# In[ ]:


sb.pairplot(suicide_data, hue= "Generation", diag_kind = "kde", kind = "scatter", palette = "husl")
plt.show()


# In[ ]:


sb.pairplot(suicide_data, hue= "Gender", diag_kind = "kde", kind = "scatter", palette = "autumn")
plt.show()


# In[ ]:


print('Total population of 101 countries:', suicide_data['Population'].sum())
print('Total number of suicide deaths:', suicide_data['SuicidesNo'].sum())


# ## Conclusion:
# 

# * Total suicide deths in all countries from 1986-2016 is: **6748420**
# * **Russia** has the highest number of suicide deaths **(1.2 million)** followed by **USA (1 million)**.
# * **Males** has highest total number of suicide deaths than **females**. men suicide numbers are **5 million** and women with **1.5 million.**
# * **36.3%** of **Age group "34-54"** died due to suicide, most among any age group.
# * **2 million boomers** died of suicide, most among the rest of the generations.
