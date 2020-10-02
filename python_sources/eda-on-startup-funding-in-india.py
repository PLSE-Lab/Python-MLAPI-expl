#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv('../input/startup_funding.csv')
print(data)


# In[ ]:


print(data['Remarks'].unique())


# In[ ]:


data.isnull().any()


# In[ ]:


data['Remarks'].fillna(0, inplace=True)


# In[ ]:


ct=0
for i in data['Remarks']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in Remarks column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


Nan_cells_count_percentage=(1953*100)/2372
print(Nan_cells_count_percentage)


# So here we can see that 82.33% data has NaN values so we can ignore this Column for out prediction

# In[ ]:


data=data.drop(['Remarks'], axis=1)


# In[ ]:


ct=0
data['IndustryVertical'].fillna(0, inplace=True)


# In[ ]:


for i in data['IndustryVertical']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in IndustryVertical column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


data['IndustryVertical'].unique().shape


# Here there are 2372 columns among which 743 are unique and 171 are NaN so we drop all these rows as large number of unique categories are there as compared to the total no. of NaN cells

# In[ ]:


data=data[data['IndustryVertical'] != 0]


# In[ ]:


data.shape


# In[ ]:


ct=0
data['SubVertical'].fillna(0, inplace=True)


# In[ ]:


for i in data['SubVertical']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in SubVertical column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


data['SubVertical'].unique().shape


# In[ ]:


import seaborn as sns
data2= data[data['SubVertical']!=0]
c2=data2.groupby('SubVertical')
d2=c2.describe()
countlist=[]
print(type(d2))
d2=d2[d2.iloc[:,0].values>2]
for i in (d2.iloc[:,0]):
    countlist.append(i)
l=np.asarray(d2.index)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.barplot(l,countlist)
plt.xticks(rotation='45')
plt.xlabel('SubVertical of Startup ')
plt.ylabel('No. of Fundings')
plt.show() 


# Online Pharmacy and food startups get most no. of fundings.

# In[ ]:


ct=0
data['CityLocation'].fillna(0, inplace=True)

for i in data['CityLocation']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in CityLocation column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


data['CityLocation'].unique().shape


# In[ ]:


data['CityLocation'].mode()


# In[ ]:



data=data.replace({'CityLocation': 0},'Bangalore')


# Thus we replace all 8 places containing NaN with 'Bangalore' which is a mode value

# In[ ]:


ct=0
data['InvestorsName'].fillna(0, inplace=True)

for i in data['InvestorsName']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in InvestorName column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


data['InvestorsName'].unique().shape


# In[ ]:


data=data[data['InvestorsName']  != 0]
data.shape


# In[ ]:


ct=0
data['InvestmentType'].fillna(0, inplace=True)

for i in data['InvestmentType']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in InvestorType column is ',ct)
print('Dimension of data is',data.shape)


# In[ ]:


data['InvestmentType'].unique().shape


# In[ ]:


data['InvestmentType'].mode()


# In[ ]:



data=data.replace({'InvestmentType': 0},'Seed Funding')


# In[ ]:


ct=0
data['AmountInUSD'].fillna(0, inplace=True)

for i in data['AmountInUSD']:
    if i==0:
        ct=ct+1
print('Total no. of NaN cells in AmountInUSD column is ',ct)
print('Dimension of data is',data.shape)


# Here we cannot fill the missing values based on mean,median,mode or any predictive model as we cannot predict the funding based on Vertical,StartupLocation etc.So we will ignore these columns when we do the data analysis on basis of Amount

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
column_1=data.iloc[:,1]
date_data=data.Date.str.split(pat=r'[/]', expand=True)
date_data.columns=['date','month','year']


# In[ ]:


data2=data.iloc[:,2:]


# In[ ]:


data_revised=pd.concat([data2,date_data], axis=1)


# In[ ]:


data_for_amt_not_zero=data_revised[data_revised['AmountInUSD'] != 0]
data_for_amt= data_revised  


# In[ ]:


data_for_amt['month'].replace(['1'],['01'],inplace=True)
data_for_amt['month'].replace(['2'],['02'],inplace=True)
data_for_amt['month'].replace(['3'],['03'],inplace=True)
data_for_amt['month'].replace(['4'],['04'],inplace=True)
data_for_amt['month'].replace(['5'],['05'],inplace=True)
data_for_amt['month'].replace(['6'],['06'],inplace=True)
data_for_amt['month'].replace(['7'],['07'],inplace=True)
data_for_amt['month'].replace(['8'],['08'],inplace=True)
data_for_amt['month'].replace(['9'],['09'],inplace=True)
data_for_amt['month'].replace(['04.2015'],['04'],inplace=True)
data_for_amt['month'].replace(['05.2015'],['05'],inplace=True)


# In[ ]:


c=data_for_amt.groupby('month')


# In[ ]:


d=c.describe()


# In[ ]:


countlist=[]
for i in (d.iloc[:,0]):
    countlist.append(i)
   


# In[ ]:


import matplotlib.pyplot as plt
months=['jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec']
#months=[1,2,3,4,5,6,7,8,9,10,11,12]
plt.figure(figsize=(15,10))
sns.barplot(months,countlist)

plt.xlabel('months')
plt.ylabel('No. of startups_funded')
plt.show()


# Here we can see that most of the investments were made in June month and between period from april to july month saw most number of Fundings.

# In[ ]:


c1=data_for_amt.groupby('year')


# In[ ]:


d1=c1.describe()


# In[ ]:


countlistyear=[]
for i in (d1.iloc[:,0]):
    countlistyear.append(i)
import matplotlib.pyplot as plt
years=[2015,2016,2017]
plt.figure(figsize=(15,10))
#sns.barplot(years,)
colors=['red','green','orange']
plt.pie(countlistyear, labels=years, colors=colors, autopct='%1.1f%%')
plt.title('Year wise no. of Startup fundings',color = 'blue',fontsize = 15)
#plt.xlabel('years')
#plt.ylabel('No. of startups_funded')
plt.show()   


# Here the investments were made mostly in 2016 and less than its half fundings were made in 2017 .Thus we can conclude that no. of fundings have decreased after a sudden increase in 2016

# In[ ]:


data_for_amt_not_zero['month'].unique()


# In[ ]:


data_for_amt_not_zero['month'].replace(['1'],['01'],inplace=True)
data_for_amt_not_zero['month'].replace(['2'],['02'],inplace=True)
data_for_amt_not_zero['month'].replace(['3'],['03'],inplace=True)
data_for_amt_not_zero['month'].replace(['4'],['04'],inplace=True)
data_for_amt_not_zero['month'].replace(['5'],['05'],inplace=True)
data_for_amt_not_zero['month'].replace(['6'],['06'],inplace=True)
data_for_amt_not_zero['month'].replace(['7'],['07'],inplace=True)
data_for_amt_not_zero['month'].replace(['8'],['08'],inplace=True)
data_for_amt_not_zero['month'].replace(['9'],['09'],inplace=True)
data_for_amt_not_zero['month'].replace(['04.2015'],['04'],inplace=True)
data_for_amt_not_zero['month'].replace(['05.2015'],['05'],inplace=True)


# In[ ]:


ulist=[]
data_for_amt_hidden=data_revised[data_revised['AmountInUSD'] == 0]
ulist=data_revised['InvestmentType'].unique()
ulist.sort()
print(ulist)


# In[ ]:


sumlist=[]

for j in ulist:
    a=(len(data_for_amt_hidden[data_for_amt_hidden.InvestmentType == j]))
    sumlist.append(a)         


# In[ ]:




plt.figure(figsize=(15,10))
#sns.barplot(years,)
colors=['red','green','orange']
plt.pie(sumlist, labels=ulist, colors=colors, autopct='%1.1f%%')
plt.title('Investment Type of  Startup fundings when Amount was not disclosed',color = 'blue',fontsize = 15)


# Here we can see that when amount was not disclosed the  most no. of startups got seed funding where as very less got private Equity Funding and no startup got debt funding.

# In[ ]:


ulist=[]
ulist=data_for_amt_not_zero['InvestmentType'].unique()
ulist.sort()
print(ulist)
sumlist=[]
for j in ulist:
    a=(len(data_for_amt_not_zero[data_for_amt_not_zero.InvestmentType == j]))
    sumlist.append(a)     

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
colors=['red','green','orange']
plt.pie(sumlist, labels=ulist, colors=colors, autopct='%1.1f%%')
plt.title('Investment Type of  Startup fundings when Amount was  disclosed',color = 'blue',fontsize = 15)
plt.show() 


# Here we can see that Private Equity funded startups are more in disclosed category of startups

# In[ ]:


ulist=[]
ulist=data_revised['InvestorsName'].unique()
print(len(ulist))


# In[ ]:


c2=data_for_amt.groupby('InvestorsName')
d2=c2.describe()
countlist=[]
print(type(d2))
d2=d2[d2.iloc[:,0].values>3]
for i in (d2.iloc[:,0]):
    countlist.append(i)
   


# In[ ]:


l=np.asarray(d2.index)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.barplot(l,countlist)

plt.xlabel('Funding Organisation')
plt.ylabel('No. of Fundings')
plt.xticks(rotation='vertical')
plt.show()     


# Here we can see that  undisclosed investors lead the tally of investing.
# Indian Angel Investors,Ratan Tata,Kalaari Capital,Sequoia Capital are other leading investors.

# In[ ]:



ulist=[]
ulist=data_revised['CityLocation'].unique()
ulist.sort()
sumlist=[]
for j in ulist:
    a=(len(data_revised[data_revised.CityLocation == j]))
    sumlist.append(a)     

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.barplot(ulist,sumlist)
plt.xticks(rotation='vertical')
plt.xlabel('City of which Startup is based on')
plt.ylabel('No. of Fundings')
plt.show()     


# Here we can see that bangalore leads the city which maximum number of startup fundings with huge margin from Mumbai and New Delhi which are second and third respectively.
# Gurgaon,Pune and Noida are other cities which are ahead. Hyderabad being such high IT hub still has less no. of fundings.

# In[ ]:


c1=data_for_amt.groupby('IndustryVertical')
d2=c1.describe()
d2=d2[d2.iloc[:,0].values>3]
 


# In[ ]:


countlist=[]
for i in (d2.iloc[:,0]):
    countlist.append(i)
l=np.asarray(d2.index)   


# In[ ]:



import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.barplot(l,countlist)
plt.xticks(rotation='45')
plt.xlabel('IndustryVertical Startup is based on')
plt.ylabel('No. of Fundings')
plt.show()     


# Here we can see that how  more No. of Consumer internet startups have got  fundings than Technology and Ecommerce Startups where as Online Education,Real Estate and food Delivery startups have struggled to get fundings.

# In[ ]:


data_for_amt_not_zero=data_revised[data_revised['AmountInUSD'] != 0]
data_for_amt_not_zero["AmountInUSD"]=data_for_amt_not_zero["AmountInUSD"].apply(lambda x: float(str(x).replace(",","")))
data_for_amt_not_zero["AmountInUSD"] = pd.to_numeric(data_for_amt_not_zero["AmountInUSD"])
print("Maximumm amount of funding given is ",data_for_amt_not_zero.AmountInUSD.max())
print("Minimum amount of funding given is ",data_for_amt_not_zero.AmountInUSD.min())
print("Average funding given to startups is ",data_for_amt_not_zero.AmountInUSD.mean())


# In[ ]:


data_for_amt_not_zero[data_for_amt_not_zero.AmountInUSD==1400000000.0]


# Here we can see that Paytm and Flipkart get the maximum funding.

# In[ ]:


data_for_amt_not_zero[data_for_amt_not_zero.AmountInUSD==18000.0]


# In[ ]:


Here we can see that Maptags got the lowest funding which is 18000.


# In[ ]:


c2=data_for_amt.groupby('StartupName')
d2=c2.describe()
countlist=[]
print(type(d2))
d2=d2[d2.iloc[:,0].values>3]
for i in (d2.iloc[:,0]):
    countlist.append(i)
l=np.asarray(d2.index) 
  


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.barplot(l,countlist)
plt.xticks(rotation='45')
plt.xlabel('IndustryVertical Startup is based on')
plt.ylabel('No. of Fundings')
plt.show()   


# Here we cans see that Swiggy got max no. of funding followed by UrbanClap,Jugnoo and Medinfi .

# In[ ]:


from wordcloud import WordCloud

names = data["InvestorsName"][~pd.isnull(data["InvestorsName"])]
print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()


# Thus we after  EDA we come to below conclusions:
#     
#     1.)The Startups were most invested in 2016 and after that sudden downfall is seen in investments.
#     
#     2.)Startups which had undisclosed amount were majorly invested by private equity where as disclosed one where Seed Funded
#     
#     3.)Startups in Consumer Internet got most no. of fundings where as sector like education and food couldnt attract more              investors.
#     
#     4.)Startups coming from the metro cities like Bangalore,Delhi,Mumbai had more no. of fundings compared to tier 2 cities.
#     
#     5.)Most no. of startups where funded between the period of April and July.
#     
#     6.)Paytm and Flipkart get the maximum funding.
#     
#     7.)Maptags got the least funding.
#     
#     8.)Swiggy got max no. of funding followed by UrbanClap,Jugnoo and Medinfi .
#     
#     9.)Based on Subvertical Online Pharmacy and food startups get most no. of fundings.
#     

# In[ ]:




