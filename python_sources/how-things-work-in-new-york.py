#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# ## WHere host name is missing:

# In[ ]:


df[df['host_name'].isnull()]


# # Where hotel name is missing :

# In[ ]:


df[df['name'].isnull()]


# # Hotel owners with most hotels in NY :

# In[ ]:


df['host_name'].value_counts().head(10)


# #### Where do they deal:

# In[ ]:


most_hotel_owners = df['host_name'].value_counts().head(10)

name_list = list(df['host_name'].value_counts().head(10).index)

temp_dict = {}


# In[ ]:


for name, group in df.groupby(by = 'host_name') :
    if name in name_list:
        if name not in temp_dict:
            temp_dict[name] = group['neighbourhood_group']
        else :
            temp_dict[name].append(group['neighbourhood_group'])

del name_list


# In[ ]:


dealing_area = pd.DataFrame(temp_dict).mode().T.join(pd.DataFrame(most_hotel_owners))

dealing_area.columns=['most_frequent_area','total_hotels_owned']

del temp_dict,most_hotel_owners

dealing_area.sort_values(by='total_hotels_owned',inplace=True)

dealing_area


# In[ ]:





# # Neighbourhood group with number of hotel listings:

# In[ ]:


df['neighbourhood_group'].value_counts()


# In[ ]:


df['neighbourhood_group'].value_counts().plot(kind='bar')


# # Number of listings by room type

# In[ ]:


df['room_type'].value_counts()


# In[ ]:


df['room_type'].value_counts().plot(kind='bar')


# # Most expensve hotels in NY City

# In[ ]:


df.sort_values(by='price',ascending=False).head(10)


# ### Number of nights exceeding 30 could be people who want to put their house on rent.

# In[ ]:


df[df['minimum_nights']>=30].groupby(by='host_name')['id'].count().sort_values(ascending=False).head(10)


# #### above are the people with most houses on rent listed on airbnb.

# In[ ]:





# ### we should add an attribute 'price per night' to get the ture most expensive hotels.

# In[ ]:


df['price_per_night'] = df['price']/df['minimum_nights']
df.drop('price',axis=1,inplace=True)


# In[ ]:


df.sort_values(by='price_per_night',ascending=False).head(10)


# ### Above are the details of most expensive hoetl listings in NY City.

# In[ ]:





# ### Let's see average price for room types and how room types are distributed across neighbourhood groups.

# In[ ]:


plt.figure(figsize=(15,4))
for name,group in df.groupby(by='neighbourhood_group'):
    sns.distplot(group['price_per_night'])
plt.title('Distribution neighbourhood_group-price')


# In[ ]:


plt.figure(figsize=(15,4))
for name,group in df.groupby(by='room_type'):
    sns.distplot(group['price_per_night'])
plt.title('Distribution room_type-price')


# ### Above graph shows the distribution of price_per_night in listings across the neighbourhood_groups. This is a skewed distribution. SO, taking mean won't give us correct answers as means are sensitive to outliars. We'll analyse mean while considering these distribution.

# In[ ]:


df.groupby(by='room_type')['price_per_night'].median()


# In[ ]:


pd.pivot_table(data=df, index='neighbourhood_group', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar')


# - As we can see from the distribution, manhattan has the most hotel listings whereas, bronx and staten island have least listings. 
# - Every neighbourhood except manhattan seem to follow a parren in distribution of hotels that is : private rooms have most listings followed by entire apartment and shared rooms.
# - In mahattan entire apartments have more listiings than private rooms.

# In[ ]:


pd.pivot_table(data=df, index='neighbourhood_group', columns='room_type', values='price_per_night' ,aggfunc='median')


# ### From Above Table we can interpret that :
# - Manhattan is the most expensive hotel listings in NY City.
# - One can get cheapest rooms in brooklyn if he/she is comfortable to share the room.
# - Staten Island has comparitively lesser listings as we'd seen before. But, they are fairly expensive.
# - After Manhatten, Queens have the most expensive listings in NY City.

# In[ ]:





# #### Since i have special interest in Manhattan, Brooklyn and Queens due to their history. (spiderman -> queens, Cap. America-> Brooklyn) Let's see further distribution in them.

# In[ ]:


manhattan_data = df[df['neighbourhood_group']=='Manhattan']
brooklyn_data = df[df['neighbourhood_group']=='Brooklyn']
queens_data = df[df['neighbourhood_group']=='Queens']


# # Manhattan :

# In[ ]:


fig,ax = plt.subplots(1,figsize=(15,4))
pd.pivot_table(data=manhattan_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)


# ### Above distributiion shows :
# - Every Neighbourhood except Harlem has specific distribution that is : most listings of entire apartment followed by private room and shared room.
# - In Harlem, Private room listings exceed Entire apartment listings in numbers
# - Harlen, Hell's Kitchen, East Village, Midtown, Upper East Side and Upper West Side are top 6 neighbourhoods with most listings in Manhattan.

# In[ ]:


temp_table = pd.pivot_table(data=manhattan_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)
temp_table


# ### From Above Table we can interpret that :
# - NoHo, Tribeca, Flatrion District, Two Bridges, Civic Center and Marble Hill are the neighbourhoods that have no listings of shared rooms.
# - NoHo, Battery Park City, Theater District, Midtown, Tribeca, Murray Hill, Chelsea are some localities which have an strange price range. Surprisingly these neighbourhoods have Entire apartment for a price lesser than that of a private room. This means the neighbourhoods have More Expensive hotel listings than apartmenty listings.
# - NoHo does not have any Shared room listings but has most expensive private rooms and entire apartments.
# - One can have the cheapest stay in Greenwich Village by sharing the room.
# - Significant neighbourhoods have median price for shared room higher than that of private rooms.

# In[ ]:


color = ['y','g','b']
flag=0
for i in temp_table.columns :
    sns.distplot(temp_table[i].dropna(),color=color[flag])
    flag+=-1
del color,flag


# #### Above graph shows distribution of room types in Manhatten.

# # Brooklyn :

# In[ ]:


fig,ax = plt.subplots(1,figsize=(15,4))
pd.pivot_table(data=brooklyn_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)


# ### Above distribution shows :
# - Bedford-Stuyvesant, Bushwick, Williamsburg, Crown Heights and GreenPoint have significantly large number of listings than those of the other neighbourgoods in brooklyn.
# - In contrast of what we've been seeing before, neighbourhoods in brooklyn have mixed distribution of number of listings. Though shared rooms are always least in number. Some neighbourhoods have more listings of Entire Apartment while some have Private room listings leading all.

# In[ ]:


temp_table = pd.pivot_table(data=brooklyn_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)
temp_table


# ### From above table we can interpret that :
# - Significant number of neighbourhoods don't have shared rooms' listings in brooklyn.
# - Mill Baasin has only Entire House listings.
# - Navy Yard, Coney Island, Brooklyn Heights and Boerum Hill have more Expensive hotel listings than entire house listings.
# - Most Expensive stay in brooklyn could be in Navy Yard where one can stay in East Flashbush by sharing room.
# - Significant neighbourhoods have median price for shared room higher than that of private rooms.

# In[ ]:


color = ['y','g','b']
flag=0
for i in temp_table.columns :
    sns.distplot(temp_table[i].dropna(),color=color[flag])
    flag+=-1
del color,flag


# ### All three; Entire apartment, Private room and shared rooms have skewed distribution across brooklyn.

# In[ ]:





# # Queens :

# In[ ]:


fig,ax = plt.subplots(1,figsize=(15,4))
pd.pivot_table(data=queens_data, index='neighbourhood', columns='room_type', values='id' ,aggfunc='count').plot(kind='bar',ax=ax)


# ### Above Distribution shows that :
# - Astoria, Ditmars Steinway, Flushing, Long Island City and Ridgewood have highest number of listings in Queens. Where Astoria leads them all with a huge margin.
# - Queens has a general distribution where Private rooom leads in number of listings across neighbourhoods folowed by entire home and shared room.

# In[ ]:


temp_table = pd.pivot_table(data=queens_data, index='neighbourhood', columns='room_type', values='price_per_night' ,aggfunc='median').sort_values(by='Private room',ascending = False)
temp_table


# ### From Above table we can interpret that;
# - significant number of neighbourhoods don't have shared rooms listings.
# - Neponsit has only Entire Home listings.
# - Holliswood and Breezy Point have only Private room listings.
# - Shared room in Ridgewood are the cheapest across Queens.
# - in Queens we have Belle harbour and Ozone Park are the neighbourhood having high end hotel listings that are expensive than the Entire Apartment listing of the area.
# - Significant neighbourhoods have median price for shared room higher than that of private rooms.

# In[ ]:





# In[ ]:


color = ['y','g','b']
flag=0
for i in temp_table.columns :
    sns.distplot(temp_table[i].dropna(),color=color[flag])
    flag+=-1
del color,flag


# ### Alll three room types have skewed distribution.

# In[ ]:





# In[ ]:


sns.scatterplot(data = df, x = 'longitude',y = 'latitude',hue = 'neighbourhood_group')


# In[ ]:


sns.scatterplot(data = df, x = 'longitude',y = 'latitude',hue = 'room_type')


# ### Above graph show distribution of room types across NY City.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Prediction :

# In[ ]:


df1=df.copy()


# In[ ]:


df1.head()


# In[ ]:


df1.describe(include='object')


# In[ ]:


df1.drop(columns=['name','host_name','last_review'],inplace=True)


# In[ ]:


df1.describe()


# In[ ]:


df1.drop('reviews_per_month',axis=1,inplace=True)


# In[ ]:


sns.heatmap(df1.corr(),cmap='Greens')


# In[ ]:


df1 = pd.get_dummies(df1)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


# In[ ]:


from sklearn.feature_selection import RFE

selector = RFE(rfr,10)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[ ]:


df1.shape


# In[ ]:


x = df1.drop('price_per_night',axis=1)
y = df1['price_per_night']


# In[ ]:


x = scaler.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)

rfr.fit(x_train,y_train)
pred = rfr.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,pred)


# In[ ]:





# In[ ]:





# In[ ]:




