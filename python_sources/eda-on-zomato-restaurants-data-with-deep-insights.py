#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dataset=pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")


# In[ ]:


dataset.columns


# In[ ]:


dataset2=pd.read_excel('../input/Country-Code.xlsx')


# In[ ]:


get_ipython().system('pip install xlrd')


# In[ ]:


dataset2


# In[ ]:


data=pd.merge(dataset,dataset2,on='Country Code')


# In[ ]:


data


# In[ ]:


data.info()


# Here we can see that there are 9551 rows and Cuisines is the row which has some null values ,as of now while analysing we will ignore those rows.

# In[ ]:


c2=data.groupby('Country')
d2=c2.describe()
country_count=d2.iloc[:,0].values
country_index=d2.index


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
sns.barplot(country_index,country_count)

plt.xlabel('Country Name')
plt.ylabel('No. of Restarurants')
plt.xticks(rotation='vertical')
plt.show()     


# Here we can see that india has most no. of restaurants which is far more than other countries,United States is on second and other countries like UAE,UK,SA have very less no. of restaurants
# 

# In[ ]:


data.groupby('City').describe().iloc[:,0].values.mean()


# In[ ]:


c1=data.groupby('City')
d2=c1.describe()
d2=d2[d2.iloc[:,0].values>20]
city_count=d2.iloc[:,0].values


# In[ ]:


city_index=d2.index
plt.figure(figsize=(12,8))
sns.barplot(city_index,city_count)

plt.xlabel('City_Name')
plt.ylabel('No. of Restarurants')
plt.xticks(rotation='vertical')
plt.show()  


# Here we can see that New Delhi has most no. of restaurants ,gurgaon,noida and Faridabad are behind it but by very huge margin,and other cities like Ahmedabad ,Amritsar,Bhubaneshwar have least no. of restaurants.It is a noticable point that there is not even a single city outside india to be in top 10 in no. of restaurants.

# In[ ]:


ax=data['Cuisines'].value_counts().head(15).plot.bar(figsize =(12,6))
plt.xlabel('Cuisine Name')
plt.ylabel('No. of Restaurants')
for i in ax.patches:
    ax.annotate(i.get_height(),(i.get_x() * 1.005, i.get_height() * 1.005))#here form is ax.annotate(a,b) where a is valus which 
    #you want to keep and b is location of value,we multiply by 1.005 so that it looks good n we get some space from our plot


# Here we can see that North Indian restaurants are more in number than North Indian Chinese,which are followed by Chinese and Fastfood

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = (WordCloud(width=500, height=300, relative_scaling=0.5, stopwords=stopwords).generate_from_frequencies(data['Cuisines'].value_counts().head(100)))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Here we can clearly see that North Indian,North Indian Chinese,Fast Food,North Indian Mughlai,Chinese lead the cuisines.

# In[ ]:


ax=data['Restaurant Name'].value_counts().head(15).plot.bar(figsize =(12,6))
plt.xlabel('Restaurant Chain Name')
plt.ylabel('No. of Restaurants')
for i in ax.patches:
    ax.annotate(i.get_height(),(i.get_x() * 1.005, i.get_height() * 1.005))#here form is ax.annotate(a,b) where a is valus which 
    #you want to keep and b is location of value,we multiply by 1.005 so that it looks good n we get some space from our plot


# Here we can see that Cafe Coffee day has most no. of outlets,domino's Pizza has almost equal number of outlets.We can see it better way in wordcloud.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
# height width represent normal dimension of wordcloud ,relative_scaling gives amount of scaling we apply while compairing two restaurants
wordcloud = (WordCloud(width=500, height=300, relative_scaling=0.75, stopwords=stopwords).generate_from_frequencies(data['Restaurant Name'].value_counts().head(100)))
fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Here word cloud clearly represents the restaurants with most no. of outlets.

# In[ ]:


#for average cost for two we will take data of india only because if we take other countries we will get currency differences.
data2=data.loc[data['Country']=='India']
ax=data2['Average Cost for two']
ay=data2['Aggregate rating']


# In[ ]:


plt.hist(ax,bins=100)
plt.xlabel('average cost for two')
plt.ylabel('no of restaurants')
l=[i for i in range(0,8000,750)]
plt.xticks(l)


# From the above graph we can clearly see that indian restaurants have price range majorly between 0-750 and then 750 to 1500 is other range where many restaurant are found.There are very less restaurants above price of 2250.

# In[ ]:


az=data2['Has Online delivery']


# In[ ]:


az.shape


# In[ ]:


import seaborn as sns
sns.lmplot( x="Aggregate rating", y="Average Cost for two", data=data2, fit_reg=False, hue='Has Online delivery', legend=False)

plt.xlabel('Cost for two')
plt.ylabel('Average Rating')


# Here we can see that restaurants upto cost of about 2000 have rating between range 2.5-4.5.
# Restaurants having price greater than 3000 have rating above 2.9 approx and Hotels having Prize above 3000 do not deliver online.

# In[ ]:


plt.hist(ay,bins=75)
plt.xlabel('average ratings')
plt.ylabel('no of restaurants')
l=[i for i in range(0,5,1)]
plt.xticks(l)


# Majority of restaurants have rating 0 and after that reviews are distributes as per bayesian right skewed curve,many restaurants have rating between 3-4

# In[ ]:


a=data['Restaurant Name'].unique()


# In[ ]:


a.shape


# In[ ]:


a


# In[ ]:


data3=data.rename(columns={"Restaurant Name":"RestaurantName"})


# In[ ]:


vote_count=[]
for i in a:
    vote_count_i=data3[data3.RestaurantName==i].iloc[:,-2].sum()
    vote_count.append(vote_count_i)


# In[ ]:


print(a)


# In[ ]:


import pandas as pd
i=[i for i in range(7446)]
data4=pd.DataFrame(vote_count,index=a,columns={"city_count"})


# In[ ]:


data4['city_count'].values.mean()


# In[ ]:


data4=data4[data4.city_count>4000]


# In[ ]:


ax=data4['city_count'].plot.bar(figsize =(12,6))
plt.xlabel('Restaurant Name')
plt.ylabel('No. of Votes')


# Here we can see that Barbeque Nation,AB's Absolute Barbecues,Big Chill ,Farzi Cafe,Toit have recieved most no. of votes

# In[ ]:


ax=data['Restaurant Name'].value_counts().head(15)


# In[ ]:


ax.index


# In[ ]:


import folium
train_data = data.loc[data['Country Code'] ==1,['Latitude', 'Longitude']]


# In[ ]:


train_data.shape


# Here we have 8652 unique hotels in india so we cannot plot them in map so we round of the lattitudes and longitudes and then plot the locations,so that restaurants near each other are plotted at the same place ,so we get its latitude and longitude same and then we drop the duplicates and we are narrowed tp 85 locations which we map in our map,

# In[ ]:


train_data=train_data.round(2)


# In[ ]:


ab=train_data.drop_duplicates(subset=['Latitude'],keep=False,inplace=True)


# In[ ]:


map_F = folium.Map(location = [train_data['Latitude'].mean(),train_data['Longitude'].mean()], zoom_start = 5)
for i, (lat, lon) in enumerate(train_data.values): folium.Marker([lat, lon]).add_to(map_F)
map_F


# There are very less Zomato restaurants in states Rajasthan,Andra Pradesh,Karnataka,Tamil Nadu,Orissa.No or negligible Restaurants in Himachal Pradesh,Chattisgarh and North Eastern States

# In[ ]:


map_F = folium.Map(location = [train_data['Latitude'].mean(),train_data['Longitude'].mean()], zoom_start = 5,      tiles = "Stamen Terrain",)
for i, (lat, lon) in enumerate(train_data.values): folium.Marker([lat, lon]).add_to(map_F)
from folium.plugins import FloatImage
map_F


# In[ ]:


map_F = folium.Map(location = [train_data['Latitude'].mean(),train_data['Longitude'].mean()], zoom_start = 5,      tiles = "Stamen Toner",)
for i, (lat, lon) in enumerate(train_data.values): folium.Marker([lat, lon]).add_to(map_F)
map_F


# In[ ]:


a=data2['Cuisines'].unique()


# In[ ]:


a.shape


# In[ ]:


alist=[]
for i in a:
    d=data2[data2['Cuisines']==i]['Average Cost for two'].mean()
    alist.append(d)


# In[ ]:


len(alist)


# In[ ]:


data_cuisine_price=pd.DataFrame(alist,index=a,columns={"Average_cost_of_cuisine"})


# In[ ]:


a=data_cuisine_price.sort_values(by=['Average_cost_of_cuisine'], ascending=False).head(20)


# In[ ]:


ax2=a['Average_cost_of_cuisine'].plot.bar(figsize =(12,6))
plt.xlabel('Cuisine Type')
plt.ylabel('Price')


# Asian,Thai,Chinese,Koran,Japanese,Malysian,Vietnamese,Srilankan,French,Italian,European,Japanese are the cuisines which dominate the costliest food list here in indian Zomato Restaurants.It is noticable to see that all the foods which are not from india but foreign countries are costliest.

# In[ ]:


data_cuisine_price=data_cuisine_price[data_cuisine_price['Average_cost_of_cuisine']!=0]
a1=data_cuisine_price.sort_values(by=['Average_cost_of_cuisine'], ascending=True).head(20)
ax2=a1['Average_cost_of_cuisine'].plot.bar(figsize =(12,6))
plt.xlabel('Cuisine Type')
plt.ylabel('Price')


# The price of Beverages,Street Food,Bakery,Mithai,South Indian,North Indian are cheapest and they are available below 100 INR also.

# In[ ]:


a4 = sns.boxplot(x="Has Online delivery", y="Average Cost for two", data=data)
a4 = sns.stripplot(x="Has Online delivery", y="Average Cost for two", jitter=True, data=data, edgecolor="black")


# Here we can clearly see that the restaurants having very high prizes dont provide Online Delivery.

# In[ ]:


import seaborn as sns
sns.lmplot( x="Aggregate rating", y="Average Cost for two", data=data2, fit_reg=False, hue='Has Table booking', legend=False)

plt.xlabel('Cost for two')
plt.ylabel('Average Rating')


# Here we can see that Restaurants with high prices have more  Table booking available facilities than restaurants with lower prices

# In[ ]:


import seaborn as sns
sns.lmplot( x="Aggregate rating", y="Average Cost for two", data=data2, fit_reg=False, hue='Is delivering now', legend=False)

plt.xlabel('Cost for two')
plt.ylabel('Average Rating')


# Here role of 'is delivering now' feature is not known but it is surely showing some trends with having cost in range from 500-1500

# Thus from Above findings we come to below conclusions:
#     
#    1.)India has most no. of restaurants which is far more than other countries,United States is on second and other countries          like UAE,UK,SA have very less no. of restaurants
#    
#    2.)New Delhi has most no. of restaurants ,gurgaon,noida and Faridabad are behind it but by very huge margin,and other cities        like Ahmedabad ,Amritsar,Bhubaneshwar have least no. of restaurants.It is a noticable point that there is not even a            single city outside india to be in top 10 in no. of restaurants.
#    
#    3.)North Indian restaurants are more in number than North Indian Chinese,which are followed by Chinese and Fastfood.
#    
#    4.)North Indian,North Indian Chinese,Fast Food,North Indian Mughlai,Chinese lead the cuisines.
#   
#   5.)Cafe Coffee day has most no. of outlets,domino's Pizza has almost equal number of outlets.We can see it better way in            wordcloud.
#   
#   6.)Indian restaurants have price range majorly between 0-750 and then 750 to 1500 is other range where many restaurant are          found.There are very less restaurants above price of 2250.
#   
#   7.)Restaurants upto cost of about 2000 have rating between range 2.5-4.5. Restaurants having price greater than 3000 have          rating above 2.9 approx.
#   
#   8.)Hotels having Prize above 3000 do not deliver online.
#   
#   9.)Majority of restaurants have rating 0 and after that reviews are distributes as per bayesian right skewed curve,many            restaurants have rating between 3-4.
#   
#   10.)Barbeque Nation,AB's Absolute Barbecues,Big Chill ,Farzi Cafe,Toit have recieved most no. of votes
#   
#   11.)There are very less Zomato restaurants in states Rajasthan,Andra Pradesh,Karnataka,Tamil Nadu,Orissa.No or negligible           Restaurants in Himachal Pradesh,Chattisgarh and North Eastern States.
#   
#   12.)Asian,Thai,Chinese,Koran,Japanese,Malysian,Vietnamese,Srilankan,French,Italian,European,Japanese are the cuisines which         dominate the costliest food list here in indian Zomato Restaurants.It is noticable to see that all the foods which are           not from india but foreign countries are costliest.
#   
#   13.)The price of Beverages,Street Food,Bakery,Mithai,South Indian,North Indian are cheapest and they are available below 100         INR also.
#   
#   14.)Restaurants with high prices have more Table booking available facilities than restaurants with lower prices

# In[ ]:




