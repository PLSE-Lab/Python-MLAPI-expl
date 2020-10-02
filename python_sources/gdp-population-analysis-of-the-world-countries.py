#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing data
data = pd.read_csv('../input/countries of the world.csv')
data.head()


# In[ ]:


#Renaming Columns
data.rename(columns={"Area (sq. mi.)": "Area", "Pop. Density (per sq. mi.)":"Pop_Density",
                        "Coastline (coast/area ratio)":"Coastline","Net migration":"Net_migration",
                        "Infant mortality (per 1000 births)":"Infant_mortality","GDP ($ per capita)":"GDP",
                        "Literacy (%)":"Literacy","Phones (per 1000)":"Phone_using","Arable (%)":"Arable",
                        "Crops (%)":"Crops","Other (%)":"Other"},inplace = True)
#We are using inplace=True to change column names in place.
data.columns


# In[ ]:


#Changing Data Type to float to be used in the analysis
data.Literacy = data.Literacy.str.replace(",",".").astype(float)
data.Pop_Density = data.Pop_Density.str.replace(",",".").astype(float)
data.Coastline = data.Coastline.str.replace(",",".").astype(float)
data.Net_migration = data.Net_migration.str.replace(",",".").astype(float)
data.Infant_mortality = data.Infant_mortality.str.replace(",",".").astype(float)
data.Phone_using = data.Phone_using.str.replace(",",".").astype(float)
data.Arable = data.Arable.str.replace(",",".").astype(float)
data.Crops = data.Crops.str.replace(",",".").astype(float)
data.Birthrate = data.Birthrate.str.replace(",",".").astype(float)
data.Deathrate = data.Deathrate.str.replace(",",".").astype(float)
data.Agriculture = data.Agriculture.str.replace(",",".").astype(float)
data.Industry = data.Industry.str.replace(",",".").astype(float)
data.Service = data.Service.str.replace(",",".").astype(float)
data.Other = data.Other.str.replace(",",".").astype(float)
data.Climate = data.Climate.str.replace(",",".").astype(float)


# In[ ]:


data.info()


# In[ ]:


data.describe


# In[ ]:


#listing the columns of the Dataframe
list(data)


# In[ ]:


#Getting Top 10 Populated Countries
Top10Coutn = data.nlargest(10, 'Population')
print(Top10Coutn)


# In[ ]:


#Calculating Population Density of the countries
data['Persons/sq. mi.'] = data['Population']/data['Area']
data.head()


# In[ ]:


#Graphing Top Countries
f, ax = plt.subplots(1,3, figsize = (21, 6))
k1 = sns.barplot(data = data.nlargest(5, 'Population'), x = 'Country', y = 'Population', ax = ax[0])
k2 = sns.barplot(data = data.nlargest(5, 'Area'), x = 'Country', y = 'Area', ax = ax[1])
k3 = sns.barplot(data = data.nlargest(5, 'Persons/sq. mi.'), x = 'Country', y = 'Persons/sq. mi.', ax = ax[2])
plt.show()
#ka = sns.barplot(data = data.nlargest(5, 'Population'), x = 'Country', y = 'Area (sq. mi.)')


# **An Interesting observation here is none of the Top 5 Countries either by population or by Land Area seem to be the densely population in terms of person/ Sq. mi.**

# In[ ]:


#Bottom Countries
f, ax = plt.subplots(1,3, figsize = (21, 6))
k1 = sns.barplot(data = data.nsmallest(5, 'Population'), x = 'Country', y = 'Population', ax = ax[0])
k2 = sns.barplot(data = data.nsmallest(5, 'Area'), x = 'Country', y = 'Area', ax = ax[1])
k3 = sns.barplot(data = data.nsmallest(5, 'Persons/sq. mi.'), x = 'Country', y = 'Persons/sq. mi.', ax = ax[2])
plt.show()


# **In the least populated Contries as well as the smallest countries by area, the population density does not seem to have any dependency on the Population as well as Area**

# In[ ]:


N = 5

fig, ax = plt.subplots(figsize = (16,4))
ind = np.arange(N)
width = 0.2
p1 = ax.bar(ind, data.nsmallest(5, 'Population')['Population']/100, width, color= 'r')
p2 = ax.bar(ind + width, data.nsmallest(5, 'Population')['Area'], width, color= 'y')
p3 = ax.bar(ind + width + width, data.nsmallest(5, 'Population')['Population']/data.nsmallest(5, 'Population')['Area'], width, color= 'g')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(data.nsmallest(5, 'Population')['Country'])
ax.legend((p1[0], p2[0], p3[0]), ('Population (in 100s)', 'Area', 'Person/ sq.mi.'))
plt.show()


# In[ ]:


N = 5
fig, ax = plt.subplots(figsize = (16,4))
ind = np.arange(N)
width = 0.2
p1 = ax.bar(ind, data.nlargest(5, 'Persons/sq. mi.')['Population']/100, width, color= 'r')
p2 = ax.bar(ind + width, data.nlargest(5, 'Persons/sq. mi.')['Area'], width, color= 'y')
p3 = ax.bar(ind + width + width, data.nlargest(5, 'Persons/sq. mi.')['Population']/data.nlargest(5, 'Persons/sq. mi.')['Area'], width, color= 'g')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(data.nlargest(5, 'Persons/sq. mi.')['Country'])
ax.legend((p1[0], p2[0], p3[0]), ('Population (in 100s)', 'Area', 'Person/ sq.mi.'))
plt.show()


# **The Above two plots show the Bootom 5 and Top 5 Countries by population comparing with them the other attributes like Area and Persons/Sq. mi.** 
# 
# **The Most Dense place in the world is Hong Kong where the Persons/ Sq. mi. is almost 66,000 while the least dense place in the world is Saint Helena where the Density is as low as 20**

# In[ ]:


plt.subplots(figsize = (20,6))
s = sns.boxplot(data = data, y = 'Population',orient = 'h')


# **The Above Boxplot shows that most countries have population less than 1 Million**

# In[ ]:


plt.subplots(figsize = (10,10))
labels = data['Country']
label = data['Population']
plt.pie(data['Population'])
plt.show()


# **This shows that the population distribution is largely in few countries of the world while most other countries have less population**

# In[ ]:


#f, ax = plt.subplots(figsize = (20,5))
s1 = sns.jointplot(data = data[data['Population']<25000000], x = 'Population', y = 'Area', kind = 'kde',xlim = (0, 25000000), ylim = (0, 200000), height = 8, ratio = 6, color = 'Green')
s2 = sns.jointplot(data = data[data['Population']<25000000], x = 'Population', y = 'Area', kind = 'reg',xlim = (0, 25000000), ylim = (0, 200000), height = 8, ratio = 6, color = 'Green', fit_reg=False)
plt.xlabel = "Population in Millions"
#ax[0].scatter(data['Persons/sq. mi.'], data['Country'], edgecolor = 'r', cmap = 'blue',label=False)
plt.show()


# **The Above Analysis also proves that most countries have population less that 1 Million while the area is peaked at around 30000 sq. mi. **

# In[ ]:


#Countries with Highest and Lowest GDP
f, ax = plt.subplots(2, 2, figsize = (16,6))
k1 = sns.barplot(data = data.nlargest(5, 'GDP'), x = 'Country', y = 'GDP', ax = ax[0,0])
k2 = sns.barplot(data = data.nsmallest(5, 'GDP'), x = 'Country', y = 'GDP', ax = ax[0,1])
k3 = sns.barplot(data = data.nlargest(5, 'Literacy'), x = 'Country', y = 'Literacy', ax = ax[1,0])
k4 = sns.barplot(data = data.nsmallest(5, 'Literacy'), x = 'Country', y = 'Literacy', ax = ax[1,1])
plt.show()


# **Here is another interesting observation**
# 
# **1. None of the Top 5 Countries by GDP feature in the Top 5 Countries by Literacy. Similary None of the Bottom 5 Countries by GDP feature in the Bottom 5 Countires by Literacy**
# 
# **2. This proves that Literacy is not the only factor that is going to determine the economy of the Countries where GDP is a representative measure of the same**

# In[ ]:


#Identifying the relationship between GDP and Literacy
data1 = data[["Country","Region","Literacy","GDP"]]
#data1.head()
data1.corr()


# **This shows that the GDP and Literacy are only partially related to each other i.e. 51.31%**

# In[ ]:


#Details analysis of the relationship between GDP and other Attributes
f, ax = plt.subplots(3,3, figsize = (18,10))
sns.scatterplot(data = data, x = "GDP", y = "Literacy", ax = ax[0,0])
sns.scatterplot(data = data, x = "GDP", y = "Birthrate", ax = ax[0,1])
sns.scatterplot(data = data, x = "GDP", y = "Deathrate", ax = ax[0,2])
sns.scatterplot(data = data, x = "GDP", y = "Infant_mortality", ax = ax[1,0])
sns.scatterplot(data = data, x = "GDP", y = "Agriculture", ax = ax[1,1])
sns.scatterplot(data = data, x = "GDP", y = "Industry", ax = ax[1,2])
sns.scatterplot(data = data, x = "GDP", y = "Climate", ax = ax[2,0])
sns.scatterplot(data = data, x = "GDP", y = "Service", ax = ax[2,1])
sns.scatterplot(data = data, x = "GDP", y = "Phone_using", ax = ax[2,2])
plt.show()


# **The Above detail analysis gives the following pointers**
# 1. Countries with High GDP also have Good Literacy rates
# 2. The Birth Rate is much Controlled/ Planned in countries with High GDP
# 3. Death rates also tend to decrease as the GDP grows
# 4. Infant Mortality has been excessively controlled leading to less than 10 in 1000 in the Countries with high GDP
# 5. As the Country's GDP grows, the Countries are transforming themselves from Agriculture oriented to Industry Oriented economy
# 6. Climate doesn't give out any dsitinct relation to GDP
# 7. Countries with High GDP score more with respect to Service Sector growth and Phone Usage of it's population

# In[ ]:


#Histogram
data.GDP.plot(kind = 'hist',bins = 50,figsize = (14,5))
plt.title("Histogram")
plt.show()


# **The Above Histogram shows that most countries are still developing to provide good GDP for it's population as most of it is having GDP less than 10000**

# In[ ]:


#Population by Region
f, ax = plt.subplots(figsize = (20, 6))
sns.barplot(data = data, x = 'Region', y = 'Population')
plt.show()


# In[ ]:


#GDP by Region
data1 = data.groupby(by = ["Region"]). GDP.mean()
print(data1)
data1.plot(kind = 'barh', y  = "GDP", x="Region", figsize = (18,6))


# In[ ]:


#Calculating Mean GDP and grouping as High or Low
Mean_GDP = data.GDP.mean()
print("Mean GDP:")
print(Mean_GDP)
data["High_Low_GDP"] = ["Low" if Mean_GDP > each else "High" for each in data.GDP]
#data.head()
data1 = data.groupby(by = "High_Low_GDP"). High_Low_GDP.value_counts()
data2 = data.groupby(by = "Region"). High_Low_GDP.value_counts()
print(data1)
print(data2)
f, ax = plt.subplots(2,1, figsize = (16, 12))
data1.plot(kind = "bar", x = "High_Low_GDP", y = "High_Low_GDP", ax = ax[0])
data2.plot(kind = "barh", x= "Region", y="High_Low_GDP", ax = ax[1])


# **The No. of Countries with GDP greater than Mean GDP of the worls are marked as High GDP countries and there number is 78, while those that are Low GDP countries is 149**
# 
# **Another Interesting thing to note here is none of the Countries in Western Europe Region have their GDP to be less than Mean GDP of the World**

# In[ ]:


#Calculating Mean population and grouping as High or Low
mean_population = data.Population.mean()
print('Mean Population  :') 
print(mean_population)
data['Population_Intensity'] = [ "low" if mean_population > each else "high" for each in data.Population ]
True_Populate = "high"
data['Population_Intensity_Num'] = [1 if True_Populate == each else 0 for each in data.Population_Intensity]
#data.Population_Intensity_Num.value_counts().plot(kind = 'hist')
#plt.subplots(figsize = (20, 4))
#sns.barplot(data = data, x = 'Region', y = data['Population_Intensity_Num'].sum())
#plt.plot(data['Population_Intensity_Num'])
data2 = data.groupby(by = ["Population_Intensity"]). Population_Intensity_Num.value_counts()
print(data2)
data2.plot(kind = 'bar', y  = "Population_Intensity_Num", x="Population_Intensity", figsize = (14,6))
plt.show()


# **This shows that 38 Countries have population greater than the Mean Population of the world while 189 has population less than the mean population**

# In[ ]:


data2 = data.groupby(by = ["Region","Population_Intensity"]). Population_Intensity_Num.value_counts()
print(data2)
data2.plot(kind = 'barh', y  = "Population_Intensity_Num", x="Population_Intensity", figsize = (18,6))


# The Above graph shows the bifurcation of the Highly Populated Countries and Less populated Countries by Region

# In[ ]:


data.corr()
f, ax = plt.subplots(figsize = (20,6))
sns.heatmap(data.corr(), annot=True)
plt.show()


# **From the above heatmap it can been seen that the GDP of a country is most related to Phone Usage in the Country followed by the Services and third by Literacy**

# In[ ]:


data[data['Population_Intensity']=='high'].head().corr()


# In[ ]:


plt.subplots(figsize = (18,6))
sns.heatmap(data[data['Population_Intensity']=='high'].head().corr(),annot=True)
plt.show()


# **The Above Heatmap gives the Correlation between attributes for Countries with High population**
# 
# **It can been seen that the Correlation is completely different from the earlier observation made with respect to GDP**
# 
# **For the Countries with Population greater than the mean population, it can be seen that the GDP is most correlated with Literacy, followed by Phone Usage of the population and third by Climatic Conditions of the Country**

# In[ ]:


#Mean Infant Mortality Rates by Region
data3 = data.groupby(by = "Region").Infant_mortality.mean()
print(data3)
plt.subplots(figsize = (18, 8))
data3.plot(kind = "bar", x = "Region", y = "Infant_mortality", figsize = (16,6), title = "Mean Infant Mortality Rates by Region", legend = True, colormap = "Dark2")
#sns.barplot(data = data, x = 'Region', y = 'Infant_mortality')
plt.legend(loc = "upper right", bbox_to_anchor=(1.15,1))
plt.show()
#sns.barplot(data = data, x = 'Region', y = 'Infant_mortality')
#sns.catplot(data = data, x = 'Region', y = 'Infant_mortality')


# **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**
