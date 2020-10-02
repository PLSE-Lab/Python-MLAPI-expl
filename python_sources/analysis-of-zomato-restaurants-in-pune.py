#!/usr/bin/env python
# coding: utf-8

# This is my first Analysis at Kaggle. Any feedback or suggestions would be appreciated.
# 
# **Dataset** : For this analysis I'll be using the Zomato Pune Restaurants Dataset which consists of information like the name, address, location, rate, votes, approx cost(for two), cuisines and reviews about the different restaurants in Pune.
# 
# **Overview Of Analysis** : Restaurants have become one of the most important parts of our daily routine whether it is casual dining or visit a club. Below is the data visualization of zomato restaurant in Pune.
# 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import urllib
import requests
import pandas as pd


# # 1. **** Preparing Data:****
# *       Dataset consists of restaurant details of many cities, only data of Pune is extracted.
# *       Unnecessary data was dropped
# *       Removing the duplicate data
# *       Displaying the data

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import urllib
import requests
import pandas as pd


# In[ ]:


data = pd.read_csv('/kaggle/input/zomato-restaurants-in-pune/zomato_restaurants_in_India.csv')
data=data[data.city=="Pune"]
data=data.drop(['res_id','url','address','latitude','longitude','country_id','zipcode','city','city_id','locality_verbose','currency'],axis=1)
data.index = range(len(data))
data=data.drop_duplicates(subset=None, keep='first')
data.shape
data.head(5)


# # 2. Establishment in Pune
# * Various establishment has been shown using a bar graph.
# * It is observed that Pune has a sufficient number of 'quick bites' and 'casual dining'.
# * Pune need some 'Microbrewery' and some 'Club'  

# In[ ]:


est_count = data['establishment'].value_counts()
est_count = est_count.sort_values(ascending=True, axis=0)


# In[ ]:


plt.figure(figsize=(35,25))
plt.xlabel('number of restaurants')
plt.ylabel('establishments')
plt.barh(est_count.index,est_count.values, color='green')


# # 3. Localities with maximum number of restaurant
# *     Only the top 5 localities have been shown using a graph.
# *     It can be observed that Baner has a maximum number of restaurants followed by Viman Nagar.
# 

# In[ ]:


#top 10 locality
loc_count = data['locality'].value_counts() 
loc_count = loc_count.sort_values(ascending=False, axis=0)
loc_count=loc_count.head()
loc_count = loc_count.sort_values(ascending=True, axis=0)
loc_count


# In[ ]:


plt.figure(figsize=(35,10))
plt.xlabel('number of restaurants')
plt.ylabel('localities')
plt.barh(loc_count.index,loc_count.values, color='blue')


# # Word cloud for Cuisines
# * In the word cloud below the cuisines have been represented using the word cloud.
# * The word cloud has been represented using an image of apple which is imported externally.
# 

# In[ ]:


words=data["cuisines"]
words=words.str.cat(sep=', ')
#words


# In[ ]:


mask = np.array(Image.open("/kaggle/input/applepng/apple.png"))


# In[ ]:


# This function takes in your text and your mask and generates a wordcloud. 
def generate_wordcloud(words, mask):
    word_cloud = WordCloud(width = 512, height = 512, background_color='black', stopwords=STOPWORDS, mask=mask).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
#Run the following to generate your wordcloud
generate_wordcloud(words, mask)


# #  5. Stacked Graph
# * It is clear from the graph that almost 85 percent of the restaurant in Pune have a digital payment  system.
# * Observed that 40 percent of restaurants sever alcohol and pre-booking facility is available only in 20 percent restaurants.

# In[ ]:


total=data.shape[0]
digital=data[data['highlights'].str.contains('Card|Digital')]
alcohol=data[data['highlights'].str.contains('Alcohol')]
booking=data[data['highlights'].str.contains('book|table')]


# In[ ]:


dig=(digital.shape[0]/total*100, (total-digital.shape[0])/total*100)
alc=(alcohol.shape[0]/total*100, (total-alcohol.shape[0])/total*100)
book=(booking.shape[0]/total*100, (total-booking.shape[0])/total*100)


# In[ ]:


N = 3
yes = (digital.shape[0]/total*100,alcohol.shape[0]/total*100   ,booking.shape[0]/total*100)
no = ((total-digital.shape[0])/total*100,(total-alcohol.shape[0])/total*100  , (total-booking.shape[0])/total*100 )


# In[ ]:


ind = np.arange(N)    
width = 0.30   
plt.figure(figsize=(10,5))
p1 = plt.bar(ind, yes, width)
p2 = plt.bar(ind, no, width, bottom=yes)
plt.ylabel('Percentage')
plt.title('Scores for Restaurants')
plt.xticks(ind, ('Digital', 'Alcohol','Booking'))
plt.yticks(np.arange(0, 101, 10))
plt.legend((p1[0], p2[0]), ('Yes', 'No'))

plt.show()


# # 6.Relation between rating of restaurant and avg cost for two people
# * It is clearly seen that the restaurants in the range of 4.5 to 5 ratings are worth their money.
# * Restaurants in the range of 3 to 3.5 rating have a relatively lower prices as compared to the price 
# * of restaurants having rating 2-3.

# In[ ]:


rating1=data.average_cost_for_two[data.aggregate_rating>4.5]
rating1=rating1.mean(axis=0)

rating2=data.average_cost_for_two[(data.aggregate_rating>4) & (data.aggregate_rating<=4.5)]
rating2=rating2.mean(axis=0)

rating3=data.average_cost_for_two[(data.aggregate_rating>3.5) & (data.aggregate_rating<=4)]
rating3=rating3.mean(axis=0)

rating4=data.average_cost_for_two[(data.aggregate_rating>3) & (data.aggregate_rating<=3.5)]
rating4=rating4.mean(axis=0)

rating5=data.average_cost_for_two[(data.aggregate_rating>2.5) & (data.aggregate_rating<=3)]
rating5=rating5.mean(axis=0)

rating6=data.average_cost_for_two[(data.aggregate_rating>2) & (data.aggregate_rating<=2.5)]
rating6=rating6.mean(axis=0)

rating7=data.average_cost_for_two[(data.aggregate_rating<2)]
rating7=rating7.mean(axis=0)


# In[ ]:


height = [rating7,rating6,rating5,rating4,rating3,rating2,rating1]
bars = ('1-2','2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5','4.5-5')
y_pos = np.arange(len(bars))

plt.rc('xtick', labelsize=15)     
plt.rc('ytick', labelsize=15)
plt.xlabel('rating')
plt.ylabel('avg cost for two people(rupees)')
plt.bar(y_pos, height, color=['black', 'red', 'green', 'blue', 'cyan','yellow','orange'])
plt.xticks(y_pos, bars)
plt.show()


# Reference:https://www.kaggle.com/manzoormahmood/analysis-of-zomato-restaurants-in-bangalore

# This is my first analysis at Kaggle.
# So if you like my work then upvote this notebook.
# Thanks

# 

# 

# 

# 
