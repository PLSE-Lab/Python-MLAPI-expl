#!/usr/bin/env python
# coding: utf-8

# <a id="top"></a> 
# # Suicides - Visualization, Correlation and Maps
# 
# **Suicide Rates Overview 1985 to 2016** dataset compares socio-economic information with suicide rates by year and country and contains suicide data of 101 countries spanning 32 years.  This report contains overview of the dataset, maps, data visualizations and correlations that impacts the suicide rates.
# 
# The following two location datasets were also utilized:
# 
# *  **World capitals GPS** contains countries, continents and latitude/longitude information and is combined with the above dataset to look at suicides per continents and to create a *Folium World map*.
# *  **World Countries** is a JSON file with country geographical shape information and is used to create the *Choropleth map*.
# 
# 
# 
# 
# ### Table Of Content
# 1.  [Data Collection](#coll)<br>
# 
# 2.  [Understanding the Data](#data)<br>
# 2.1  [Data in the Dataset](#data_data)<br>
# 2.2  [Types of Data](#data_type)<br>
# 2.3  [Data Types](#data_info)<br>
# 2.4  [Statistical Summary](#data_summ)<br>
# 
# 3.  [Data Cleaning ](#prep)<br>
# 
# 4. [Data Visualization ](#eda)<br>
# 4.1  [Suicide Rates per Country](#eda_1)<br>
# 4.2  [Population and GDP over Time](#eda_2)<br>
# 4.3  [Suicides and Suicide Rates over Time](#eda_3)<br>
# 4.4  [Suicide Rates per Age Groups and GDP](#eda_4)<br>
# 4.5  [Male & Female Suicide Rates per Continents](#eda_5)<br>
# 4.6  [Male & Female Suicide Rates per Age Groups](#eda_6)<br>
# 4.7  [Suicide Trends over Time - Age Groups](#eda_7)<br>
# 4.8  [Suicide Trends over Time - Male/Female](#eda_8)<br>
# 
# 5.  [Correlations](#corr)<br>
# 5.1  [Encoding and Normalization](#corr_encode)<br>
# 5.2  [Overall Correlation](#corr_over)<br>
# 5.2  [Correlation - Male & Female](#corr_male)<br>    
# 
# 6.  [Choropleth and Folium Maps](#maps)<br>
# 
# <br><br>
# *Limitations:*  Dataset contains around half countries in the world, so by far is not complete, but is an excellent resource for creating maps and data visualizations.

# [go to top of document](#top)     
# 
# ---
# #  1.  Data Collection<a id="coll"></a>
# 
# This section involves importing Python libraries, importing data, cleaning and finally merging the datasets. 
# 
# ###  Import Libraries

# In[ ]:


#  Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  maps
import folium
from folium.plugins import MarkerCluster


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 80)

#  Kaggle directories
import os
print(os.listdir("../input"))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ###  Load the Datasets

# In[ ]:


df  = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")  # suicides
gps = pd.read_csv("../input/world-capitals-gps/concap.csv")   # world GPS


# ###  Merge the Datasets
# 1.  **Check 'df' for country names that do not match country names in 'gps'**
# 2.  **Update the country names**

# In[ ]:


# check df against gps
count = 0
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))
        count = count + 1
print('check complete:  {} missing'.format(count)) 

#  update names in df to match the gps file
df.replace({'Cabo Verde':'Cape Verde','Republic of Korea':'South Korea','Russian Federation':'Russia','Saint Vincent and Grenadines':'Saint Vincent and the Grenadines'},inplace=True)


# 3.  **Re-check for NULLs**

# In[ ]:


# check df against gps
count = 0
for i in sorted(df.country.unique()):
    if len(gps.CountryName[gps.CountryName == i].values) == 0:
        print('MISSING in gps:  df: {}\t\tgps:{}'.format(i,gps.CountryName[gps.CountryName == i].values))
        count = count + 1
print('check complete:  {} missing'.format(count))        


# 4.  **Combine the datasets keyed on country cames**
# 5.  **Drop columns that are not needed**

# In[ ]:


df = df.join(gps.set_index('CountryName'), on='country')
df = df.drop(['HDI for year','country-year','CountryCode','CapitalName'], axis=1)
df.info()


# [go to top of document](#top)     
# 
# ---
# #  2.  Understanding the Data <a id="data"></a>

# ##  2.1  Data in the Dataset   <a id="data_data"></a>
# 
# ### Countries in the dataset:
# The dataset contains suicide information on 101 out of 195 countries.  More significantly, 6 of the top 10 most populous countries are not in the dataset.
# 
# source:  https://www.worldometers.info/geography/how-many-countries-are-there-in-the-world/

# In[ ]:


#  Top 10 most populous countries in the world
top10 = ['China','India','United States','Indonesia','Brazil','Pakistan','Nigeria','Bangladesh','Russia','Mexico']
in_set = df.country[df.country.str.contains('|'.join(top10))].unique().tolist()

print('Out of the top 10 most populous countries: \n{}\n\nonly the following {} are present:\n{}'.format(top10,len(in_set),in_set))

#  dataset
print('\n\nDataset has', len(df['country'].unique()),'countries (out of 195) on' ,len(df['ContinentName'].unique()),'continents spanning' ,len(df['year'].unique()),'years.')


# ##  2.2  Types of Data <a id="data_type"></a>
# -  **Categorical data** represents characteristics, and are values or observations that can be sorted into groups or categories.  
# -  **Numeric data** are values or observations that can be measured, and placed in ascending or descending order. 
# 
# From the dataset, we can identify the **Categorical** and **Numeric** data as:
# 
# 
# |  **CATEGORICAL** |  	  **NUMERIC**   |  
# | :-- | :-- |
# | country| 	 year| 
# |  age| 	  suicides_no| 
# |  sex| 	  population| 
# |  gdp_for_year| 	  suicides/100k pop| 
# |  generation| 	  gdp_per_capita| 
# |  ContinentName|	  CapitalLatitude| 
# |  -         |  CapitalLongitude|
# 
# <br>   
# **note:**  2016 data is incomplete and will not be used.

# ##  2.3  Data Types   <a id="data_info"></a>
# Categorical attribute data type needs to be of type "object" for analysis.

# In[ ]:


print(df.info())         #  dataset size and types
print('\nDATA SHAPES:  {}'.format(df.shape))


# ##  2.4  Statistical Summary <a id="data_summ"></a>
# Summarize descriptive statistics of the dataset for *categorical* and *numeric* attributes. 

# > ###  2.4.1 Statistical Summary - CATEGORICAL DATA
# Summarize the count, uniqueness and frequency of categorical features, excluding numeric values.

# In[ ]:


df.describe(include=['O'])   #  CATEGORICAL DATA


# ###  2.4.2 Statistical Summary - NUMERIC DATA
# Summarize the central tendency, dispersion and shape of numeric features, excluding categorical and NaN values.

# In[ ]:


df.describe()   #  NUMERIC DATA


# [go to top of document](#top)     
# 
# ---
# #  3.  Data Cleaning <a id="prep"></a>
# Data clean-up was completed in [Data Collection](#coll) section with  renaming of the countries and dropping columns.  
# 
# This is a final re-check for NULL and duplicate values in the dataset.

# In[ ]:


nulls = df.isnull().sum().sort_values(ascending = False)
prcet = round(nulls/len(df)*100,2)
df.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])

print('List of NULL rows\n{}'.format(df.null))
print('\nDUPLICATED rows:\t{}'.format(df.duplicated().sum()))

plt.title('NULLs heatmap')
sns.heatmap(df.isnull())


# **There are no nulls or duplicates in the 'df' dataset.**

# [go to top of document](#top)     
# 
# ---
# #  4. Data Visualization <a id="eda"></a>
# note:  data from 2016 is incomplete and removed from analysis.

# ## 4.1  Suicide Rates per Country<a id="eda_1"></a>
# Plot the average suicide rates per country and compare with the global mean.

# In[ ]:


suicideRate = df['suicides/100k pop'].groupby(df['country']).mean().sort_values(ascending=False).reset_index()
suicideMean = suicideRate['suicides/100k pop'].mean()

plt.figure(figsize=(8,20))
plt.title('Suicide Rates per Country (mean={:.2f})'.format(suicideMean), fontsize=14)
plt.axvline(x=suicideMean,color='gray',ls='--')
sns.barplot(data=suicideRate, y='country',x='suicides/100k pop')

suicideRate.head(10)


# **OBSERVATIONS**   
# Lithuania, Sri Lanka and Russia top the list with suicide rates much higher than the global mean of 12.

# ## 4.2  Population and GDP over Time<a id="eda_2"></a>

# In[ ]:


YRS = sorted(df.year.unique()-1)  # not including 2016 data
POP = []    # population
GDC = []    # gdp_per_capita ($)
SUI = []    # suicides_no
SUR = []    # suicides/100k pop

for year in sorted(YRS):
    POP.append(df[df['year']==year]['population'].sum())
    GDC.append(df[df['year']==year]['gdp_per_capita ($)'].sum())
    SUI.append(df[df['year']==year]['suicides_no'].sum())
    SUR.append(df[df['year']==year]['suicides/100k pop'].sum())

#  plot population and gdp_per_capita ($), 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Population vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Population', fontsize=12)
plt.axis('auto')
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,POP)
fig.add_subplot(122)
plt.title('GDP per Capita vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('GDP per Capita (in $)', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,GDC)
plt.show()


# **OBSERVATIONS**   
# Population and GDP per Capita were steadily increasing from 1985 to 2008, then leveling off and markedly declining after 2014.

# ## 4.3  Suicides and Suicide Rates over Time<a id="eda_3"></a>

# In[ ]:


#  plot suicides_no and suicides/100k pop, 1985-2015
fig = plt.figure(figsize=(12,4))
fig.add_subplot(121)
plt.title('Total Suicides vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUI)
fig.add_subplot(122)
plt.title('Suicides per 100k vs Years', fontsize=14)
plt.xlabel('Years', fontsize=12)
plt.ylabel('Suicides/100k Population', fontsize=12)
plt.xlim(1985,2015)
plt.grid();plt.plot(YRS,SUR)
plt.show()


# **OBSERVATIONS**   
# Total number of suicides have been leveling off since the mid-90s, but more importantly, the rate of suicides has been declining since the mid-90s.  It is still difficult to correlate any information between populations, GDP and suicide rates.

# ## 4.4  Suicide Rates per Age Groups and GDP<a id="eda_4"></a>

# In[ ]:


ageList = sorted(df.age.unique())
ageList.remove('5-14 years')
fig = plt.figure(figsize=(12,5))

for i in ageList:
    fig.add_subplot(121)
    plt.title('Suicide Rates per Age Group', fontsize=14)
    plt.xlabel('suicides/100k pop', fontsize=12)
    plt.xlim(0,50)
    plt.legend(ageList)
    df['suicides/100k pop'][df['age'] == i].plot(kind='kde')

    fig.add_subplot(122)
    plt.title('Suicide Rates vs GDP', fontsize=14)
    plt.xlabel('gdp_per_capita ($)', fontsize=12)
    plt.yticks([], [])
    plt.xlim(0,100000)
    #df['gdp_per_capita ($)'][df['age'] == i].plot(kind='kde')
    df['gdp_per_capita ($)'].plot(kind='kde')


# **OBSERVATIONS**   
# GDP per capita has an inverse effect on the rate of suicides; lower the GDP per Capita, higher the rate of suicides.  Rate of people 75+ yrs. are far more likely to commit suicide, and is significantly higher in countries with a lower GDP.

# ## 4.5  Male & Female Suicide Rates per Continents<a id="eda_5"></a>

# In[ ]:


fig = plt.figure(figsize=(10,6))
plt.title('Male/Female Suicides/100k per Continents', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data =df, x='sex',y='suicides/100k pop', hue='ContinentName',palette='Blues_r')


# **OBSERVATIONS**   
# Europeans have a higher rate of suicides than any other continents, and males are four times more likely to commit suicides then females.
# 
# We cannot assume that this is actually true since this dataset contains only 101 out of 245 countries.

# ## 4.6  Male & Female Suicide Rates per Age Groups<a id="eda_6"></a>
# **Age** and **Generation** attributes ranges are very similar. 

# In[ ]:


fig = plt.figure(figsize=(10,5))
fig.add_subplot(121)
plt.title('Male/Female Suicides/100k vs Age', fontsize=14)
plt.xlabel('sex', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='age')
fig.add_subplot(122)
plt.title('Male/Female Suicides/100k vs Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
sns.barplot(data=df, x='sex',y='suicides/100k pop', hue='generation')
plt.show()


# **OBSERVATIONS**   
# Males are almost four times more likely to commit suicide then females.  Both males and females over 55 years are more susceptible then other age groups.

# ## 4.7  Suicide Trends over Time - Age Groups<a id="eda_7"></a>

# In[ ]:


df_sort =  df.sort_values(by='age')  # sort by age
plt.figure(figsize=(10,8))
plt.title('Suicide Trend', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort,x='year',y='suicides/100k pop',hue='age',ci=None)


# **OBSERVATIONS**  
# Suicide rates have been steadily declining since 1995 for all age groups, however, the past few years are seeing an alarming uptick in suicide rates for 55+ age groups.

# ## 4.8  Suicide Trends over Time - Male/Female<a id="eda_8"></a>

# In[ ]:


fig = plt.figure(figsize=(14,6))
fig.add_subplot(121)
plt.title('Suicide Trend - MALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'male'], x='year',y='suicides/100k pop',hue='age',ci=None)
fig.add_subplot(122)
plt.title('Suicide Trend - FEMALE', fontsize=14)
plt.xlim(1985,2015)     # disregarding 2016
sns.lineplot(data=df_sort[df_sort['sex'] == 'female'], x='year',y='suicides/100k pop',hue='age',ci=None)
plt.show()


# **OBSERVATIONS**   
# Rate of suicides have been declining for both males and females, however, there has been a significant uptick in rates of females in age groups 55+ in recent years.

# [go to top of document](#top)     
# 
# ---
# # 5  Correlations<a id="corr"></a>

# Correlation is a statistical metric for measuring to what extent different variables are interdependent.  In the analysis, we will look at the overall correlation, as well as the correlations based on male/female.
# 
# In order to perform correlation, we need to first take care of two very important processes:
# 
#   *  Encoding categorical attributes with numerical values
#   *  Normalization of the data

# ## 5.1  Encoding and Normalization<a id="corr_encode"></a>
# ### Encoding
# Machine learning algorithms cannot process categorical or text data unless they have been converted to numbers. Encoding maps categorical values to integer values, which are represented as a binary vector that are all zero values, except the index of the integer, which is set to 1.
# 
# Categorical attributes will be manually encoded with numeric values.  The steps involved are:
# 
# 1.  drop columns not needed for correlation
# 2.  rearrange column names
# 3.  encode

# In[ ]:


df.columns


# In[ ]:


#  1.  drop columns not needed for correlation
df_corr = df.drop(['country','year','CapitalLatitude','CapitalLongitude'], axis=1)

#  2.  rearrange column names
df_corr = df_corr[['suicides/100k pop', 'sex', 'age', 'population',' gdp_for_year ($) ','gdp_per_capita ($)', 'generation','suicides_no','ContinentName']]

#  3.  encode
df_corr['sex'] = df_corr['sex'].map({'female':0,'male':1})
df_corr['age'] = df_corr['age'].map({
        '5-14 years':0,'15-24 years':1,'25-34 years':2,
        '35-54 years':3,'55-74 years':4,'75+ years':5})
df_corr['generation'] = df_corr['generation'].map({
        'Generation Z':0,'Millenials':1,'Generation X':2,
        'Boomers':3,'Silent':4,'G.I. Generation':5})
df_corr['ContinentName'] = df_corr['ContinentName'].map({
        'Africa':0,'Asia':1,'Australia':2,'Central America':3,
        'Europe':4,'North America':5,'South America':6})

#  remove commas and save as float64
df_corr[' gdp_for_year ($) '] = df_corr[' gdp_for_year ($) '].str.replace(',','').astype('float64')

#df_corr.describe(include=['O'])   #  CATEGORICAL DATA
df_corr.info()


# ###  Normalization
# Normalization is a rescaling of the data from the original range so that all values are within a certain range, typically between 0 and 1. Normalized data is essential in machine learning. Correlation and models will not produce good results if the scales are not standardized.
# 
# Data in **df_corr** will be normalized and the **df** data frame will be updated with the encoded and normalized data.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
df_norm = MinMaxScaler().fit_transform(df_corr)
df_C = pd.DataFrame(df_norm, index=df_corr.index, columns=df_corr.columns)


# ## 5.2  Overall Correlation<a id="corr_over"></a>

# In[ ]:


dataCorr = df_C.corr()
plt.figure(figsize=(8,8))
plt.title('Suicide Correlation', fontsize=14)
sns.heatmap(dataCorr, annot=True, fmt='.2f', square=True, cmap = 'Greens_r')

dataCorr['suicides/100k pop'].sort_values(ascending=False)


# **OBSERVATIONS** 
# As noted in the plots, age/generation and sex are significant factors in determining  suicide rates.

# ## 5.3  Correlation - Male & Female<a id="corr_male"></a>

# In[ ]:


#  Correlation MALE - filter dataframe for male/female
dataMale   = df_C[(df_C['sex'] == 1)]                       # male
dataMaleCorr = dataMale.drop(["sex"], axis=1).corr()        # male corr
corrM = dataMaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation FEMALE - filter dataframe for male/female
dataFemale = df_C[(df_C['sex'] == 0)]                       # female
dataFemaleCorr = dataFemale.drop(["sex"], axis=1).corr()    # female corr
corrF = dataFemaleCorr['suicides/100k pop'].sort_values(ascending=False)

#  Correlation heatmaps for FEMALE/MALE
fig = plt.figure(figsize=(16,8))
fig.add_subplot(121)
plt.title('Suicide Correlation - MALE', fontsize=14)
sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues_r')
fig.add_subplot(122)
plt.title('Suicide Correlation - FEMALE ', fontsize=14)
sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds_r')
plt.show()


# **OBSERVATIONS** 
# Age/generation are significant factors for both males and females in determining  suicide rates.

# [go to top of document](#top)     
# 
# ---
# # 6  Choropleth and Folium Maps<a id="maps"></a>
# 
# ## 6.1  Choropleth Map   
# Choropleth map showing the suicide rates of the countries in the dataset.

# In[ ]:


#  create dataframe with Country and mean of Suicide rates per 100k Population
df_choro = df[['suicides/100k pop','country']].groupby(['country']).mean().sort_values(by='suicides/100k pop').reset_index()

#  Update US name to match JSON file
df_choro.replace({'United States':'United States of America'},inplace=True)

#  https://www.kaggle.com/ktochylin/world-countries
world_geo = r'../input/world-countries/world-countries.json'
world_choropelth = folium.Map(location=[0, 0], tiles='cartodbpositron',zoom_start=1)

world_choropelth.choropleth(
    geo_data=world_geo,
    data=df_choro,
    columns=['country','suicides/100k pop'],
    key_on='feature.properties.name',
    fill_color='PuBu',  # YlGn
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Suicide Rates per 100k Population')

 
# display map
world_choropelth


# ## 6.2 Folium World Map
# 
# Folium map showing the number of suicides per countries in the dataset.

# In[ ]:


#  create dataframe for mapping
mapdf = pd.DataFrame(columns =  ['country','suicides_no','lat','lon'])

mapdf.lat = mapdf.lat.astype(float).fillna(0.0)
mapdf.lon = mapdf.lat.astype(float).fillna(0.0)

mapdf['country']     = df['suicides_no'].groupby(df['country']).sum().index
mapdf['suicides_no'] = df['suicides_no'].groupby(df['country']).sum().values
for i in range(len(mapdf.country)):
    mapdf.lat[i] =  df.CapitalLatitude[(df['country'] == mapdf.country[i])].unique()
    mapdf.lon[i] = df.CapitalLongitude[(df['country'] == mapdf.country[i])].unique()


#  make map - popup displays country and suicide count
#  lat/lon must be "float"
world_map = folium.Map(location=[mapdf.lat.mean(),mapdf.lon.mean()],zoom_start=2)
marker_cluster = MarkerCluster().add_to(world_map)

for i in range(len(mapdf)-1):
    label = '{}:  {} suicides'.format(mapdf.country[i].upper(),mapdf.suicides_no[i])
    label = folium.Popup(label, parse_html=True)
    folium.Marker(location=[mapdf.lat[i],mapdf.lon[i]],
            popup = label,
            icon = folium.Icon(color='green')
    ).add_to(marker_cluster)


world_map.add_child(marker_cluster)
world_map         #  display map


# [go to top of document](#top)
# 
# ---
# ###  END
# Please upvote if you found this helpful :-)
