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


# * [Knowing your data](#1)
# * [Getting Rid of superfluity](#2)
#     * [Removing Superflous Features](#3) 
#     * [Removing Duplicates](#4)
# * [Fetching the Reqiured Data for our Analysis](#5)
# * [Exploratory Data Analysis](#6)
#     * [Feature creation for better understanding of data](#7)
#     * [Top Localities in Chennai and Bangalore](#8)
#     * [Top Restaurant Types in Chennai and Bangalore](#9)
#     * [Top Brands based on Number of outlets](#10)
#     * [Major Types of Restaurants](#11)
#     * [Major Cuisines in Chennai and Bangalore](#12)
#     * [Average Cost for each of the Top Establishment types](#13)
#     * [Price Range in Top classified Localities](#14)
#     * [Price Range in Top classified Establishments](#15)
#     * [Top Quick Bites Restaurants](#16)
#     * [Top Casual Dining Restaurants](#17)
#       

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
zomato = pd.read_csv("../input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv")


# # [Knowing your data]()<a id="1"></a><br>

# # knowing your Data

# In[ ]:


#checking for the recodrs and features
print('Number of Features:',zomato.shape[1])
print('Number of Records:',zomato.shape[0])


# In[ ]:


#Checking for Duplicated values
zomato[zomato.duplicated()].count()[0]


#  # [Getting Rid of superfluity]()<a id="2"></a><br>

# ## Getting Rid of superfluity is very important as the superfluous features will not add value to our analysis
# 

# # [Removing Superflous Features]()<a id="2"></a><br>

# In[ ]:


zomato.columns


# In[ ]:


redundant=['highlights','rating_text','res_id','url','address','city_id','country_id','zipcode','longitude','latitude','currency','photo_count','delivery','takeaway','locality_verbose','timings','opentable_support']


# In[ ]:


print('Number of Features before Pruning:',zomato.shape[1])
zomato.drop(redundant, axis=1, inplace=True)
print('Number of Features after Pruning:',zomato.shape[1])


# ## The next step is getting rid of Duplication in the dataset
# 

#  # [Removing Duplicates]()<a id="4"></a><br>

# In[ ]:


print('Number of Hotels before Removal of Duplicates:',zomato.shape[0])
zomato.drop_duplicates(inplace=True)
print('Number of Hotels after Removal of Duplicates:',zomato.shape[0])


# In[ ]:


zomato.head()


# 
#  # [Fetching the Reqiured Data for our Analysis]()<a id="5"></a><br>

# ## Our aim is to compare the Foodchain industry,quality and pricing of chennai and bangalore localities, let's get the required data
# 

# In[ ]:


zc=zomato[(zomato['city']=='Chennai')|(zomato['city']=='Bangalore')]
zc.to_csv('zomato_chn_blr.csv')


# In[ ]:


import pandas as pd
zom_comp = pd.read_csv("../input/test-data/zomato_chn_blr.csv")


# 
#  # [Exploratory Data Analysis]()<a id="6"></a><br>

# # EDA

# In[ ]:


zom_comp.head(2)


# #### Since we saved the file into another csv and reloaded it, the old index has been named as Unnamed in our new data

# In[ ]:


zom_comp.drop('Unnamed: 0', axis=1, inplace=True)
zom_comp.head(2)


# In[ ]:


## checking for null values
zom_comp.isnull().sum()


# In[ ]:


# Removing unwanted characters from establishment column
zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace('[',''))
zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace(']',''))
zom_comp['establishment']=zom_comp['establishment'].apply(lambda x :str(x).replace("'",''))


# 
#  # [Feature creation for better understanding]()<a id="7"></a><br>

# # Feature Creation

# In[ ]:


#Creating a New feature for better understanding of ratings
l=[]
for i in range(0,zom_comp.shape[0]):
    if zom_comp.iloc[i,7]<=1:
        l.append('Poor')
    elif zom_comp.iloc[i,7]>1 and zom_comp.iloc[i,7]<=2:
        l.append('Average')
    elif zom_comp.iloc[i,7]>2 and zom_comp.iloc[i,7]<=3:
        l.append('Good')
    elif zom_comp.iloc[i,7]>3 and zom_comp.iloc[i,7]<=4:
        l.append('Very Good')
    elif zom_comp.iloc[i,7]>4 and zom_comp.iloc[i,7]<=5:
        l.append('Excellent')
        
rat=pd.Series(l, name='Word_rating')


# In[ ]:


#concating with dataframe
zom_comp=pd.concat([zom_comp,rat], axis=1,join='outer')


# In[ ]:


# Naming the Price_ratings
dic={1:'Low',2:'Average',3:'High',4:'Very High'}
zom_comp['price_type']=zom_comp['price_range'].map(dic)


# In[ ]:


zom_comp.isnull().sum()
# Feature creation has not affected out data_set


# In[ ]:


zom_comp.loc[:,['Word_rating','price_type']]


# ## Number of Restaurants In Chennai and Bangalore[](http://)

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(6,6))
fig.suptitle('Number of Restaurants', fontsize=15)
zom_comp.groupby('city')['cuisines'].count().plot(kind='bar',color = 'orange', ax=ax)
for i in range(2):
    plt.text(x = i-0.08 , y=zom_comp.groupby('city')['cuisines'].count()[i]+15, s = zom_comp.groupby('city')['cuisines'].count()[i], size =15)


# #### The number of hotles in bangalore is  greater than that of in chennai by few hundreds
# 

# # Number Of Localities

# In[ ]:


chn=zom_comp[zom_comp['city']=='Chennai']
blr=zom_comp[zom_comp['city']=='Bangalore']
a=len(list(chn.locality.unique()))
b=len(list(blr.locality.unique()))
v=[]
v.append(b)
v.append(a)
fig,ax=plt.subplots(1,1,figsize=(6,6))
fig.suptitle('Number of Localities', fontsize=15)
sns.barplot(x=zom_comp.city.unique(), y =v, ax=ax)
for i in range(2):
    plt.text(x = i-0.08 , y=v[i]+1, s = v[i], size =15)


# ### when compared,chennai has a greater partition of localities than Bangalore, this means density of distibution of hotles is better in Bangalore

# 
#  # [Top Localities in Chennai and Bangalore]()<a id="8"></a><br>

# # Top Localities

# ### We are going to see the Top 10 localities in both the cities having Majority of the Restaurants

# In[ ]:


chn_10=(zom_comp[zom_comp['city']=='Chennai'])['locality'].value_counts().head(10)
blr_10=(zom_comp[zom_comp['city']=='Bangalore'])['locality'].value_counts().head(10)
fig,ax=plt.subplots(1,2,figsize=(30,8))
a=sns.barplot(chn_10.index,chn_10.values, ax=ax[0])
a.set_xlabel('Locality')
a.set_ylabel('Count')
b=sns.barplot(blr_10.index,blr_10.values, ax=ax[1])
b.set_xlabel('Locality')
b.set_ylabel('Count')
fig.suptitle('Major localities of hotels', fontsize=20)
a.title.set_text('Chennai')
b.title.set_text('Bangalore')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(10):
    a.text(x = i-0.08 , y=list(chn_10.values)[i]+1, s = list(chn_10.values)[i], size =15)
    b.text(x = i-0.08 , y=list(blr_10.values)[i]+1, s = list(blr_10.values)[i], size =15)

print('Major Localities of Chennai \n',chn_10)  
print('Major Localities of Bangalore \n',blr_10)    


# ### From the graph we can see that the Total number of hotels in top localitites of Bangalore is always greater than that of chennai which can be correlated with the Total Number of hotels in these cities(Bangalore>Chennai)

# # Top Establishments

# 
#  # [Top Restaurant Types in Chennai and Bangalore]()<a id="9"></a><br>

# In[ ]:


chn_10_typ=(zom_comp[zom_comp['city']=='Chennai'])['establishment'].value_counts().head(10)
blr_10_typ=(zom_comp[zom_comp['city']=='Bangalore'])['establishment'].value_counts().head(10)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(10,5))
c=sns.barplot(chn_10_typ.index,chn_10_typ.values, ax=ax[0])
c.set_xlabel('Type of Restaurant')
c.set_ylabel('Count')
d=sns.barplot(blr_10_typ.index,blr_10_typ.values, ax=ax[1])
d.set_xlabel('Type of Restauran')
d.set_ylabel('Count')
c.title.set_text('Major Types of hotels in chennai')
d.title.set_text('Major Types of hotels in bangalore')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(10):
    c.text(x = i-0.4 , y=list(chn_10_typ.values)[i]+1, s = list(chn_10_typ.values)[i], size =10)
    d.text(x = i-0.4 , y=list(blr_10_typ.values)[i]+1, s = list(blr_10_typ.values)[i], size =10)


# ### 1. Almost top 5 types of restaurants are similar in both the cities and the banglore has higher counts as expected in each of the Top 5.
# ### 2. Getting to the Next 5 we can see some minor changes.
# ### 3. In chennai there more number of FINE DINING restaurants, where as the spot 6 in bangalore was grabbed by 'Sweet shops'
# ### 4. From this we can infer that bangalore being a hub for IT professionals,they prefer more of casual dining than high priced fine dining
# ### 5. In Bangalore there more food courts than chennai, where as in chennai bakeries are more established

# ### Tabular Visualization
# 

# In[ ]:


pd.crosstab(zom_comp['city'], zom_comp['establishment']).loc[['Chennai'],list(chn_10_typ.index)]


# In[ ]:


pd.crosstab(zom_comp['city'], zom_comp['establishment']).loc[['Bangalore',],list(blr_10_typ.index)]


# # Top Brands

# 
#  # [Top Brands in Chennai and Bangalore]()<a id="10"></a><br>

# In[ ]:


chn_10_brands=(zom_comp[(zom_comp['city']=='Chennai')])['name'].value_counts().head(10)
blr_10_brands=(zom_comp[(zom_comp['city']=='Bangalore')])['name'].value_counts().head(10)
fig,ax=plt.subplots(1,2, figsize=(10,5))
e=sns.barplot(chn_10_brands.index,chn_10_brands.values, ax=ax[0])
f=sns.barplot(blr_10_brands.index,blr_10_brands.values, ax=ax[1])
fig.suptitle('Top Brands based on Number of Outlets', fontsize=20)
e.title.set_text('Chennai')
f.title.set_text('Bangalore')
e.set_xlabel('Brand')
f.set_xlabel('Brand')
e.set_ylabel('Number of Outlets')
f.set_ylabel('Number of Outlets')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(10):
    e.text(x=i-0.3, y=list(chn_10_brands.values)[i]+0.5, s=list(chn_10_brands.values)[i])
    f.text(x=i-0.3, y=list(blr_10_brands.values)[i]+0.5, s=list(blr_10_brands.values)[i])

    
    


# ### 1.From the above graph we can see that the Brand markets of two cities are totally different from each other.
# ### 2.Both of the city have it's unique and Hometown brands like "A2B" and "Hotel Saravana Bhavan" in Chennai and Kanti Sweets and Government aided "Indira Canteen" in Bangalore
# 
# ### 3.Similarly Chennai have "ibaco" and "Cream and Fudge" as its major outlet for icecream, where as Bangalore again has it's own Hometown product "Corner House Icecream" and "Polar Bear"
# 
# ### 4.For Pizzas chennai has "Domino's" and Bangalore with "onesta"

# # Establishment Vs. Locality

# 
#  # [Major Establisments of Restaurants in Top Localities]()<a id="11"></a><br>

# In[ ]:


chn_loc_est=pd.crosstab(zom_comp['locality'], zom_comp['establishment']).loc[list(chn_10.index),list(chn_10_typ.index)]
blr_loc_est=pd.crosstab(zom_comp['locality'], zom_comp['establishment']).loc[list(blr_10.index),list(blr_10_typ.index)]
fig,ax=plt.subplots(1,2, figsize=(20,10))
g=chn_loc_est.plot(kind='bar',stacked=True, ax=ax[0])
h=blr_loc_est.plot(kind='bar',stacked=True,ax=ax[1])
plt.legend()
fig.suptitle('Spread of Major Types of restaurants in each of the Top 10 Localities in both the cities', fontsize=20)


# # Major Cuisines

# 
#  # [Major Cuisines in Chennai and Bangalore]()<a id="12"></a><br>

# In[ ]:


mcc=zom_comp[(zom_comp['city']=='Chennai')]['cuisines'].value_counts().head(10)
mcb=zom_comp[(zom_comp['city']=='Bangalore')]['cuisines'].value_counts().head(10)
fig,ax=plt.subplots(1,2, figsize=(10,5))
l=sns.barplot(mcc.index,mcc.values, ax=ax[0])
m=sns.barplot(mcb.index,mcb.values, ax=ax[1])
fig.suptitle('Major cuisines', fontsize=20)
l.title.set_text('Chennai')
m.title.set_text('Bangalore')
l.set_xlabel('cuisines')
m.set_xlabel('cuisines')
l.set_ylabel('Number of each cuisines')
m.set_ylabel('Number of each cuisines')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(10):
    l.text(x=i-0.3, y=list(mcc.values)[i]+0.5, s=list(mcc.values)[i])
    m.text(x=i-0.3, y=list(mcb.values)[i]+0.5, s=list(mcb.values)[i])

    
    


# ### 1. From the Graph it can be seen that there are more South Indian Cuisines in Bangalore than Chennai
# ### 2. In chennai the trend seems to be like people are preferring snacks and desserts after sounth indian cuisine, where as in Bangalore the top 3 prference seems to be for Mainly food and not snacks and desserts

# 
#  # [Average cost per each establishments types]()<a id="13"></a><br>

# In[ ]:


avg_chn=pd.DataFrame(chn.groupby('establishment')['average_cost_for_two'].median().sort_values(ascending=False))
avg_blr=pd.DataFrame(blr.groupby('establishment')['average_cost_for_two'].median().sort_values(ascending=False))
av_c=avg_chn.loc[chn_10_typ.index].sort_values(by=['average_cost_for_two'],ascending=False)
av_b=avg_blr.loc[blr_10_typ.index].sort_values(by=['average_cost_for_two'],ascending=False)


# In[ ]:


print(av_c)
print(av_b)


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(20,7))
avc=av_c.plot(kind='bar',ax=ax[0])
avb=av_b.plot(kind='bar',ax=ax[1])
fig.suptitle('Average price',fontsize=20)
avc.title.set_text('Chennai')
avb.title.set_text('Bangalore')
avc.set_xlabel('Establishment Type')
avb.set_xlabel('Establishment Type')
avc.set_ylabel('Price')
avb.set_ylabel('price')
avc.legend('')
avb.legend('')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(10):
    avc.text(x=i-0.3, y=list(av_c.values)[i]+30, s=list(av_c.values)[i])
    avb.text(x=i-0.3, y=list(av_b.values)[i]+30, s=list(av_b.values)[i])


# ### From the Above Graph it is evident that the average cost in Bangalore is greater than Chennai in most of the establishments

# # Price Range

# 
#  # [Price Range in Top classified Localities]()<a id="14"></a><br>

# In[ ]:


chn_pri_loc=pd.crosstab(zom_comp['locality'], zom_comp['price_type']).loc[list(chn_10.index)]
blr_pri_loc=pd.crosstab(zom_comp['locality'], zom_comp['price_type']).loc[list(blr_10.index)]
chn_pri_loc


# In[ ]:


blr_pri_loc


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(10,5))
n=chn_pri_loc.plot(kind='bar',stacked=True, ax=ax[0])
o=blr_pri_loc.plot(kind='bar',stacked=True, ax=ax[1])
fig.suptitle('Price Range in Top classified Localities',fontsize=20)
n.title.set_text('Chennai')
o.title.set_text('Bangalore')
n.set_xlabel('Locality')
o.set_xlabel('Locality')
n.set_ylabel('count')
o.set_ylabel('count')
n.legend(bbox_to_anchor=[-0.15,1])
o.legend(bbox_to_anchor=[1.5, 1])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)


# ### 1. From the above plots and Tabulation it is evident that all the localities in Bangalore except 'Electronic city'and 'jp nagar', where as in chennai only 'Nungambakkam', 'Alwarpet' and 'Phoenix Market city' has high priced restaurants.

# 
#  # [Price Range in Top classified Establishments]()<a id="15"></a><br>

# In[ ]:


chn_pri_est=pd.crosstab(chn['establishment'], chn['price_type']).loc[list(chn_10_typ.index),:]
blr_pri_est=pd.crosstab(blr['establishment'], blr['price_type']).loc[list(blr_10_typ.index),:]
chn_pri_est


# In[ ]:


blr_pri_est


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(10,5))
n=chn_pri_est.plot(kind='bar',stacked=True, ax=ax[0])
o=blr_pri_est.plot(kind='bar',stacked=True, ax=ax[1])
fig.suptitle('Price Range in Top classified Restaurants',fontsize=20)
n.title.set_text('Chennai')
o.title.set_text('Bangalore')
n.set_xlabel('Restaurant Type')
o.set_xlabel('Restaurant Type')
n.set_ylabel('count')
o.set_ylabel('count')
n.legend(bbox_to_anchor=[-0.15,1])
o.legend(bbox_to_anchor=[1.5, 1])

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)


# ### In both the cities only casual and fine dinings have high priced restaurants

# In[ ]:


print('Number of Low priced hotels in chennai:',chn_pri_est['Low'].sum())
print('Number of Low priced hotels in Bangalore:',blr_pri_est['Low'].sum())


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(10,5))
n=chn_pri_est.sum().plot(kind='bar', ax=ax[0])
o=blr_pri_est.sum().plot(kind='bar', ax=ax[1])
fig.suptitle('Price Range in Top classified Restaurants',fontsize=20)
n.title.set_text('Chennai')
o.title.set_text('Bangalore')
n.set_xlabel('Restaurant Type')
o.set_xlabel('Restaurant Type')
n.set_ylabel('count')
o.set_ylabel('count')
n.legend(bbox_to_anchor=[-0.15,1])
o.legend(bbox_to_anchor=[1.5, 1])
n.legend('')
o.legend('')
chl=list(chn_pri_est.sum().values)
bl=list(blr_pri_est.sum().values)
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
for i in range(4):
    n.text(x=i-0.15, y=chl[i]+10, s=chl[i])
    o.text(x=i-0.15, y=bl[i]+10, s=bl[i])


# ### From The above plot it is clear that chennai has more number of low priced Restaurants than Bangalore  
# * [Low:200-450]
# * [Medium : 500-950]
# * [High : 1100-1900]
# * [LVery High : 2000-5500]
# 

# # Quick Bites

# 
#  # [Top Quick Bites]()<a id="16"></a><br>

# ## Chennai

# In[ ]:


chn_qb=chn[(chn['establishment']=='Quick Bites')]
chn_qb=chn_qb.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)
chn_qb['rank']=chn_qb['votes'].rank(ascending=False,method='dense')


# In[ ]:


chn_qb.sort_values(by=['rank'])


# ### 1. I have ranked the restaurants based on no.of votes and agg rating, bcoz more the number of votes more people visit the place and still manages to have higher ratings,
# ### 2. even if we rank it based on agg_rating there won't be any change in the top_10 Brands but position will vary
# ### 3. In this case kailash kitchen will grab the first rank and fusili we pulled down to 2nd position
# 

# ## Bangalore

# In[ ]:


blr_qb=blr[(blr['establishment']=='Quick Bites')]
blr_qb=blr_qb.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)
blr_qb['rank']=blr_qb['votes'].rank(ascending=False,method='dense')
blr_qb.sort_values(by=['rank'])


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(20,5))
sns.barplot(chn_qb['name'], chn_qb['price_range'],ax=ax[0],hue=chn_qb['Word_rating'])
sns.barplot(blr_qb['name'], blr_qb['price_range'],ax=ax[1],hue=blr_qb['Word_rating'])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)


# ### All the qucik bites are Low priced restaurants in both the cities

# # Casual Dining

# 
#  # [Top Casual Dinings]()<a id="17"></a><br>

# In[ ]:


# Top Casual dinings in chennai
chn_cd=chn[(chn['establishment']=='Casual Dining')]
chn_cd=chn_cd.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)
chn_cd['rank']=chn_cd['votes'].rank(ascending=False,method='dense')
chn_cd.sort_values(by=['rank'])

#Top casual dinings in bangalore
blr_cd=blr[(blr['establishment']=='Casual Dining')]
blr_cd=blr_cd.sort_values(by=['aggregate_rating','votes'], ascending=False).head(10)
blr_cd['rank']=blr_cd['votes'].rank(ascending=False,method='dense')


# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(20,5))
sns.barplot(chn_cd['name'], chn_cd['price_range'],ax=ax[0],hue=chn_cd['Word_rating'])
sns.barplot(blr_cd['name'], blr_cd['price_range'],ax=ax[1],hue=blr_cd['Word_rating'])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)


# ### 1. In chennai except 'onesta' and 'bhangra' all other restaurents are high priced
# ### 2. In Bangalore all the Top Casual Dining restaurants are High prices

# In[ ]:


fig,ax=plt.subplots(1,2, figsize=(10,5))
a1=sns.countplot(chn_cd['name'], ax=ax[0],hue=chn_cd['price_range'])
b1=sns.countplot(blr_cd['name'], ax=ax[1],hue=blr_cd['price_range'])
fig.suptitle('Top Casual Dinings',fontsize=20)
a1.title.set_text('Chennai')
b1.title.set_text('Bangalore')
a1.set_xlabel('Restaurant Name')
b1.set_xlabel('Restaurant Name')
a1.set_ylabel('count')
b1.set_ylabel('count')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)

print(chn_cd.groupby('name')['name'].count())
print(blr_cd.groupby('name')['name'].count())


# ### 1. From the above plot it is evident that top ten spots are dominated by BBQ and Buffet in both the cities,espescially  AB's has a great base in bangalore, than compared to any other casual dinings in top 10
# ### 2. In chennai both coal BBQ and AB's BBQ goes hand in hand both in rating as well as Number of outlets.
# ### 3. In chennai except Onesta and Bhangra(average priced) all other casual dinings are high priced, in bangalore all are High priced
# 
