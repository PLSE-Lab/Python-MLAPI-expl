#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# * ![](http://)![](http://)![](http://) 

# In[ ]:


data_15=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
data_16=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
data_17=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
data_18=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data_19=pd.read_csv("/kaggle/input/world-happiness/2019.csv")


# In[ ]:


data15=data_15.copy()
data16=data_16.copy()
data17=data_17.copy()
data18=data_18.copy()
data19=data_19.copy()  #We copy original data


# we need only Country,Region,(happiness.score=score),Health,Freedom,Corruption,Generosity

# # #2015 report

# In[ ]:


data15.columns


# In[ ]:


data15.drop(['Happiness Rank','Standard Error','Economy (GDP per Capita)','Family','Dystopia Residual'],axis=1,inplace=True)
#we deleted unnecessary columns


# In[ ]:


data15.rename(columns={"Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Corruption"},inplace=True)
#We rename columns what we want.


# In[ ]:


data15['Year'] = '2015' #We add column Year 
data15 = data15[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']] 
#we have determined the columns order


# In[ ]:


data15  #final version of the report 2015.


# # #2016 report

# In[ ]:


data16.columns


# In[ ]:


data16.drop(['Happiness Rank','Lower Confidence Interval','Upper Confidence Interval','Economy (GDP per Capita)','Family','Dystopia Residual'],axis=1,inplace=True)
#we deleted unnecessary columns


# In[ ]:


data16.rename(columns={"Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Corruption"},inplace=True)
#We rename columns what we want.


# In[ ]:


data16['Year'] = '2016' #We add a column 'Year'.And order columns
data16 = data16[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


data16 #Final version of the report 2016


# # #2017 report

# In[ ]:


data17.columns


# In[ ]:


data17.drop(['Happiness.Rank','Whisker.high','Whisker.low','Economy..GDP.per.Capita.','Family','Dystopia.Residual',],axis=1,inplace=True)
#we delete unnecessary columns,


# In[ ]:


data17.rename(columns={"Happiness.Score":"Happiness Score","Health..Life.Expectancy.":"Health","Trust..Government.Corruption.":"Corruption"},inplace=True)
#We remane columns.


# In[ ]:


data17['Year'] = '2017' #We add a new column and order columns
data17 = data17[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


data17 #Final version of the report 2017


# # #2018 report

# In[ ]:


data18.columns


# In[ ]:


data18.rename(columns={"Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Health","Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"},inplace=True)
#We rename columns,


# In[ ]:


data18.drop(['Overall rank','GDP per capita','Social support'],axis=1,inplace=True) #We delete unnecessary columns.


# In[ ]:


data18['Year'] = '2018' #We add columns 'Year'.And order columns
data18 = data18[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


data18 ##Final version of the report 2018


# # #2019 report

# In[ ]:


data19.columns


# In[ ]:


data19.drop(['Overall rank','GDP per capita','Social support'],axis=1,inplace=True) #We delete unnecessary columns


# In[ ]:


data19.rename(columns={"Country or region":"Country","Score":"Happiness Score","Healthy life expectancy":"Health","Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"},inplace=True)
#We rename columns what we want.


# In[ ]:


data19['Year'] = '2019' #Add a new columns and order all columns
data19 = data19[['Year','Country','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


data19 #Final version of the report 2019


# In[ ]:


a=list(data16.Country.unique()) #we look at sorted countries.
sorted(a)


# # 1.Merge all the data into a single dataframe

# In[ ]:


alldata=pd.concat([data15,data16,data17,data18,data19],axis=0,ignore_index=True)
alldata =alldata[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


alldata #we combined all reports into a single dataframe


# In[ ]:


# print(alldata[alldata.Country=='Trinidad & Tobago'])
# print(alldata[alldata.Country=='Taiwan Province of China'])
# print(alldata[alldata.Country=='Hong Kong S.A.R., China'])
# print(alldata[alldata.Country=='Northern Cyprus'])
# print(alldata[alldata.Country=='North Macedonia']) 
#Changing different names for the same countries for consistency.


# In[ ]:


alldata.loc[507,'Country'] = 'Trinidad and Tobago'
alldata.loc[664,'Country'] = 'Trinidad and Tobago'
alldata.loc[385,'Country'] = 'Hong Kong'
alldata.loc[527,'Country'] = 'North Cyprus'
alldata.loc[689,'Country'] = 'North Cyprus'
alldata.loc[709,'Country'] = 'Macedonia'
alldata.loc[347,'Country'] = 'Taiwan'
#Changing different names for the same countries for consistency.


# In[ ]:


alldata.columns


# In[ ]:



d = data15.iloc[:,1:3] 
alldata = pd.merge(alldata,d, on = 'Country', how = 'left')

#Take the country and region information.


# In[ ]:


alldata


# In[ ]:


alldata.drop(["Region_x"],axis=1,inplace=True)
alldata.rename(columns={"Region_y":"Region"},inplace=True) #We delete unnecessary columns.And order columns
alldata =alldata[['Year','Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']]


# In[ ]:


alldata #Final version of alldata(2015,2016,2017,2018,2019)


# # #2. Count the NaN values.

# In[ ]:


alldata.isna().sum() #There are 18 NaN values of region and one value of corruption


# 
# # #3. Some datasets do not have the region information of the countries, search these countries and add them to the missing places in the region column.

# In[ ]:


alldata[alldata.Region.isna()].Country


# In[ ]:


#We find missing region informations
alldata.loc[233,'Region'] = 'Sub-Saharan Africa'
alldata.loc[407,'Region'] = 'Sub-Saharan Africa'
alldata.loc[567,'Region'] = 'Sub-Saharan Africa'
alldata.loc[737,'Region'] = 'Sub-Saharan Africa'
alldata.loc[270,'Region'] = 'Sub-Saharan Africa'
alldata.loc[425,'Region'] = 'Sub-Saharan Africa'
alldata.loc[588,'Region'] = 'Sub-Saharan Africa'
alldata.loc[738,'Region'] = 'Sub-Saharan Africa'
alldata.loc[300,'Region'] = 'Sub-Saharan Africa'
alldata.loc[461,'Region'] = 'Sub-Saharan Africa'
alldata.loc[623,'Region'] = 'Sub-Saharan Africa'
alldata.loc[781,'Region'] = 'Sub-Saharan Africa'
alldata.loc[254,'Region'] = 'Sub-Saharan Africa'
alldata.loc[745,'Region'] = 'Sub-Saharan Africa'
alldata.loc[364,'Region'] = 'Latin America and Caribbean'  
alldata.loc[518,'Region'] = 'Latin America and Caribbean'
alldata.loc[172,'Region'] = 'Latin America and Caribbean'
alldata.loc[209,'Region'] = 'Latin America and Caribbean'


# # #4. Fill NaN values with the average of that variable

# In[ ]:


alldata["Corruption"].mean()


# In[ ]:


alldata[alldata.Corruption.isna()]


# In[ ]:


alldata["Corruption"].fillna(alldata["Corruption"].mean(),inplace=True) #We write missing corruption as mean.


# In[ ]:


alldata.isna().sum() #We havent NaN values now.


# # #5. Get the info() and describe() information on the data.

# In[ ]:


alldata.info() #alldata information


# In[ ]:


alldata.describe().T #alldata describe.


# # #6. Calculate how many different regions exist.

# In[ ]:


alldata["Region"].value_counts() 


# In[ ]:


alldata.Region.unique()


# In[ ]:


len(alldata.Region.unique())


# # #7. Which is the variable that affects happiness most. Check if there is a difference by years.

# In[ ]:


koralasyon=alldata.corr() #We find correlation of alldata.
koralasyon


# In[ ]:


koralasyon[(koralasyon.abs()>0.5)&(koralasyon.abs()<1)]  #health and feedom affect happiness most.


# In[ ]:


corr_years=alldata.groupby("Year").corr()
corr_years


# In[ ]:


corr_years[(corr_years.abs()>0.5)&(corr_years.abs()<1)] #Generally,every year health and freedom affect happiness score most.


# # #8. What is the number of countries by region

# In[ ]:


alldata["Region"].value_counts()


# # #9. Find the happiest and the most unhappy 3 countries by taking the average of HAPPINESS for 5 years.

# In[ ]:


happ_averages=alldata.groupby('Country')['Happiness Score'].mean()


# In[ ]:


happ_averages


# In[ ]:


happ_averages.sort_values(ascending=False).head(3)


# In[ ]:


happ_averages.sort_values(ascending=False).tail(3)


# # #10. Find the best and worst countries by taking the average of 5 years CORRUPTION.

# In[ ]:


corrup_averages=alldata.groupby('Country')['Corruption'].mean()
corrup_averages


# In[ ]:


corrup_averages.sort_values(ascending=False).head()


# In[ ]:


corrup_averages.sort_values(ascending=False).tail()


# # #11. whats is the highest and lowest freedom average by region

# In[ ]:


freedom_averages=alldata.groupby('Region')['Freedom'].mean()
freedom_averages.sort_values(ascending=False).head(1)


# In[ ]:


freedom_averages.sort_values(ascending=False).tail(1)


# # #12. Find out which region is the most unhealthy.

# In[ ]:


health_average = alldata.groupby('Region')['Health'].mean()
health_average.sort_values(ascending=False).tail(1)


# # #13. By grouping the countries according to their regions, take the average of the variables Happiness, Freedom and Corruption.

# In[ ]:


alldata.groupby("Region").aggregate({'Happiness Score':'mean','Freedom':'mean', 'Corruption':'mean'}).T


# In[ ]:




