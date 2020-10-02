#!/usr/bin/env python
# coding: utf-8

# # Analysis Of Indian Statrtup Funding Data

# <img src="http://jrhassociates.net/wp-content/uploads/2014/09/7.png" title="Source : www.jrhassociates.net">

# ## Introduction  
# 
# India has been very active in creating a healthy startup ecosystem, and the growth in the number of startups is increasing year on year. In fact, it is among the top five startup communities in the world.  
# Being so startup friendly the country has attracted numerous numbers of investors, both national and international. Therefore a large amount of money is poured into the startup ecosystem. Also due to government support, technology boon and rise of tier-2 and tier-3 cities has boosted the startup ecosystem.   
# Events like launch of Startup India initiative, US elections and the Indian banknote demonetization had a huge impact on the startup community. The dataset acquired via kaggle.com will help to analyze the startup trends from start of 2015 to mid 2017 and how to above events have affected the trend.  

# ## Scope of the Analysis

# The followings are the scope for this report
# *	The important investor of the Indian ecosystem.
# *	The amount of funds does startup generally gets in India.
# *	The roles of cities in funding.
# *	Various sectors or industries which are more favored by investors for funding.
# *	How funding has changed overtime.
# *	How events like US elections and Indian banknote demonetization affected the funding.
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil
import squarify
import os


# In[ ]:


fund_data = pd.read_csv('../input/startup_funding.csv')


# #### Knowning more about the dataset

# In[ ]:


fund_data.head()


# #### Shape and Info 

# In[ ]:


fund_data.shape


# In[ ]:


fund_data.info()


# #### Columns of the table

# In[ ]:


fund_data.columns 


# #### Identifiying Null values and percentage of null values

# In[ ]:


fund_data.isnull().sum().sort_values(ascending =False)


# Now Percentage

# In[ ]:


missing = fund_data.isnull().sum().sort_values(ascending=False)
percent = (missing/fund_data.isnull().count())*100
print("Percentage of missing data")
percent


# We can see that Remarks has the highest amount of null values therefore the columns has to be dropped.

# In[ ]:


fund_data.drop(['Remarks'],axis=1,inplace=True)


# #### Fixing Dates and other data

# In[ ]:


fund_data['Date']=fund_data['Date'].replace({"12/05.2015":"12/05/2015"})
fund_data['Date']=fund_data['Date'].replace({"13/04.2015":"13/04/2015"})
fund_data['Date']=fund_data['Date'].replace({"15/01.2015":"15/01/2015"})
fund_data['Date']=fund_data['Date'].replace({"22/01//2015":"22/01/2015"})
fund_data['StartupName'] = fund_data['StartupName'].replace({"Flipkart.com":"Flipkart"})
fund_data['IndustryVertical']=fund_data['IndustryVertical'].replace({"ECommerce":"eCommerce"})
fund_data['IndustryVertical']=fund_data['IndustryVertical'].replace({"ecommerce":"eCommerce"})
fund_data['IndustryVertical']=fund_data['IndustryVertical'].replace({"Ecommerce":"eCommerce"})
fund_data['InvestmentType']=fund_data['InvestmentType'].replace({"Crowd funding":"Crowd Funding"})
fund_data['InvestmentType']=fund_data['InvestmentType'].replace({"SeedFunding":"Seed Funding"})
fund_data['InvestmentType']=fund_data['InvestmentType'].replace({"PrivateEquity":"Private Equity"})
fund_data['StartupName']=fund_data['StartupName'].replace({"practo":"Practo"})
fund_data['StartupName']=fund_data['StartupName'].replace({"couponmachine.in":"Couponmachine"})
fund_data['StartupName']=fund_data['StartupName'].replace({"Olacabs":"Ola Cabs"})
fund_data['StartupName']=fund_data['StartupName'].replace({"Ola":"Ola Cabs"})


# #### Replace in ',' to '' in AmountInUSD

# In[ ]:


fund_data['AmountInUSD'] = fund_data['AmountInUSD'].apply(lambda x:float(str(x).replace(",","")))


# ### Some indepth insights on the amont of investements

# #### Minimum Investments

# In[ ]:


print("Minimum Investment")
fund_data['AmountInUSD'].min()


# Details of the minimum investment of $ 16000.0

# In[ ]:


fund_data[fund_data['AmountInUSD']==16000.0]


# All the above startups were funded at Startup Heroes Event.

# #### Maximum Investment

# In[ ]:


print("Maximum Investment")
fund_data['AmountInUSD'].max()


# Details of the maximum investment of $ 1400000000.0

# In[ ]:


fund_data[fund_data.AmountInUSD == 1400000000.0]


# So __Paytm__ and __Flipkart__ were the startups that had the maximum investments

# Let's look how many times Flipkart and Paytm was funded.

# In[ ]:


fund_data[fund_data['StartupName'] == "Flipkart"]


# In[ ]:


fund_data[fund_data['StartupName'] == "Paytm"]


# #### Mean Investment

# In[ ]:


fund_data['AmountInUSD'].mean()


# #### Total Investment From 1/1/2015 To 28/7/2017
# 

# In[ ]:


fund_data['AmountInUSD'].sum()


# #### Number of Investment per month

# In[ ]:


fund_data["yearmonth"] = (pd.to_datetime(fund_data['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(fund_data['Date'],format='%d/%m/%Y').dt.month)
temp = fund_data['yearmonth'].value_counts().sort_values(ascending = False)
print("Number of funding per month in decreasing order (Funding Wise)\n\n",temp)
year_month = fund_data['yearmonth'].value_counts()


# ### Now plotting some graphs

# ####  Year-Month - Number of Funding Distribution

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(year_month.index, year_month.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Year-Month - Number of Funding Distribution", fontsize=16)
plt.show()


# #### Conclusion from above :    
# * July and August of  2015 had the highest investment that year because of the __Digital India Campaign__  
# * January 2016 had the higest investment of 2016 and also in the dataset because of the __Startup India Initaitive__
# * As expected Demonitization lowered the investement per month from __Nov '16__ to __July '17__
# * __Interseting note : __ July '16 saw a decrease of investment and the lowest in the year , may be beacuse of the __Surgical strike__ that happed that month.

# #### Year-Month - Amount of Funding distribution

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(fund_data['yearmonth'], fund_data['AmountInUSD'], alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('YearMonth', fontsize=12)
plt.ylabel('Amonut Of Investments', fontsize=12)
plt.title("YearMonth - Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusion from the above :
# * March '17 and May '17 had the maximum investment because  __Flipkart__ and __Paytm__ were funded then.

# ## Basic analysis of startups

# #### Total number of startups

# In[ ]:


len(fund_data['StartupName'])


# #### Unique startups

# In[ ]:


len(fund_data['StartupName'].unique())


# #### Startups that got funding more than 1 times

# In[ ]:


tot = (fund_data['StartupName'].value_counts()).values
c=0
for i in tot:
    if i > 1:
        c=c+1
print("Startups that got funding more than 1 times = ",c)


# In[ ]:


fund_count  = fund_data['StartupName'].value_counts()
fund_count = fund_count.head(20)
print(fund_count)


# #### Plot for top 20 companies that secured 4 or more than 4 fundings

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(fund_count.index, fund_count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Startups', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Startups-Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusion from above 
# * Most of the companie sthat were funded 4 or more than 4 times were __Consumer Internet__ companies with some exceptions.

# ## Industry Verticals

# #### Unique Industry Verticals

# In[ ]:


len(fund_data['IndustryVertical'].unique())


# In[ ]:


IndustryVert = fund_data['IndustryVertical'].value_counts().head(20)
print(IndustryVert)


# #### Plot for Industry Vertical

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(IndustryVert.index, IndustryVert.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Industry Verticals', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Industry Verticals-Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusion from above
# * Consumer Internet startups are the most with 32.5 % in  total.
# * Technology with the second most of 13.2%
# * And Ecommerce with 9 %

# ## Subvertical

# #### Unique Subverticals

# In[ ]:


sub_vertical = fund_data['SubVertical']
print("Total number of subverticals : ",len(sub_vertical.unique()))


# #### Top 25 Subverticals

# In[ ]:


sub_vertical=sub_vertical.value_counts().head(25)
print(sub_vertical)


# #### Plot for Subverticals

# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(sub_vertical.index, sub_vertical.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Industry Sub Verticals', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Industry Sub Verticals-Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusions from above
# *  Online Pharmacy leads the way with 9 investments

# ## Investment Types

# In[ ]:


Investment_Type = fund_data['InvestmentType'].value_counts()
print(Investment_Type)


# In[ ]:


plt.figure(figsize=(12,5))
sns.barplot(Investment_Type.index, Investment_Type.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Investment Type', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investment Type - Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusions from above
# * Seed Funding and Private funding is the most preferred way of investments by Investors

# ## Cities

# #### Unquie Locations

# In[ ]:


len(fund_data['CityLocation'].unique())


# In[ ]:



fund_data['CityLocation'].value_counts().head(10)


# In[ ]:


fund_city = fund_data['CityLocation'].value_counts()
plt.figure(figsize=(16,9))
sns.barplot(fund_city.index, fund_city.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('City', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("City - Number of fundings distribution", fontsize=16)
plt.show()


# #### Conclusions from above
# * Banglore attracted the most investors with a total of 627 investors. (26.4%)
# * Mumbai with 446 investors. (18.8%)
# * New Delhi with 381 investors. (16.1%)
# * __Intersting note : __  Few II Tier and III cities like Varanasi, Indore, Siliguri, Karur, Nagpur, Belgaum, Kozhikode have also attracted some investors. 

# ## Investors

# ### Processing Investors Column

# The Investors column need to be pre processed because the cells having more the one investors will also be treated ad a single identity,  therfore they need to be seperated corrected and combine for give the right results

# In[ ]:


fund_data['InvestorsName'] = fund_data['InvestorsName'].fillna("No info Available")
names = fund_data["InvestorsName"][~pd.isnull(fund_data["InvestorsName"])]
print(names.head())


# In[ ]:


Investor_list = fund_data['InvestorsName'].str.split(',').apply(pd.Series)


# #### Seperating each cell

# In[ ]:


Investor_list.head()


# In[ ]:


print("Combining all columns into one")
df = Investor_list.stack(dropna=False).reset_index(drop=True).to_frame('newinvest')
print(df.head(10))


# In[ ]:


InvestorsName = df.dropna(axis=0, how='all')


# In[ ]:


# Correcting typos
InvestorsName=InvestorsName.replace({" Sequoia Capital":"Sequoia Capital"})
InvestorsName=InvestorsName.replace({"Undisclosed investors":"Undisclosed Investors"})
InvestorsName=InvestorsName.replace({"undisclosed investors":"Undisclosed Investors"})
InvestorsName=InvestorsName.replace({"undisclosed Investors":"Undisclosed Investors"})
InvestorsName=InvestorsName.replace({"Undisclosed":"Undisclosed Investors"})
InvestorsName=InvestorsName.replace({"Undisclosed Investor":"Undisclosed Investors"})
InvestorsName=InvestorsName.replace({" Accel Partners":"Accel Partners"})
InvestorsName=InvestorsName.replace({" Blume Ventures":"Blume Ventures"})
InvestorsName=InvestorsName.replace({" SAIF Partners":"SAIF Partners"})
InvestorsName=InvestorsName.replace({" Kalaari Capital":"Kalaari Capital"})


# #### Total investments

# In[ ]:


len(InvestorsName['newinvest'])


# In[ ]:


# Now all investors are sperated into individual rows
InvestorsName.head(10)


# #### Unique Investors

# In[ ]:


InvestorsName['newinvest'] = InvestorsName['newinvest'].str.strip()
print(len(InvestorsName['newinvest'].unique()))


# In[ ]:


Investors_top50 = InvestorsName['newinvest'].value_counts().head(50)
# Top 50 investors
print(Investors_top50)


# #### Plot for Investors

# In[ ]:


plt.figure(figsize=(16,9))
sns.barplot(Investors_top50.index, Investors_top50.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Investors', fontsize=12)
plt.ylabel('Number of Startups Invested In', fontsize=12)
plt.title("Investors - Investment distribution", fontsize=16)
plt.show()


# #### Squarify plot for Investors

# In[ ]:


plt.figure(figsize=(16,9))
squarify.plot(sizes=Investors_top50.values,label=Investors_top50.index, value=Investors_top50.values,color=["#FF6138","#FFFF9D","#BEEB9F", "#79BD8F","#684656","#E7EFF3"], alpha=0.6)
plt.title('Distribution of Investors and Investments Done')


# #### Conclusions from above
# * The plot shows that __Undisclosed Investors__ have done the most investments.
# * Followed by Sequoia Capitals with 64 Investments
# * Individuals like __Ratan Tata( Former chairman of Tata Sons)__ and __Rajan Anandan( VP,Google SE Asia and India)__ have invested in 30 and 25 companies respectively, most by any individuals

# ### So this was my analysis on the Indian Startup Funding. More to come.
# ### Any remarks are welcomed. 
# ### Please upvote if you like the kernal.
# 
# 
# ## Thanks You!!

# In[ ]:





# In[ ]:




