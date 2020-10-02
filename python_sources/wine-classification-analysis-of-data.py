#!/usr/bin/env python
# coding: utf-8

# ## Text Classification using NLP for Various types of Wines -- Part1

# <p> This is the wine review dataset. And this part of then notebook is concerned with various parameters of analysing the reviews dataset. And plotting graphs and getting the basic understanding of the dataset .Then Next part will comprise of Making Word Clouds for this dataset. And then in the final part, I will be using NLTK for  Natural Language Processing and use various ML Techniques to find out the accuract of predictions for various algos and find out which is better.  </p>
# <h3> So lets get started </h3>

# ### Importing libraries and dataset

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
# Importing Natural language Processing toolkit.
from PIL import Image
# from python imaging library
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[ ]:


wines = pd.read_csv("../input/winemag-data-130k-v2.csv",index_col = 0)
wines = wines.dropna(subset = ['points' , 'price'])
print(wines.shape)
wines.head(2)


# ## Basic Information about the dataset

# In[ ]:


print("There are {} countries and these are {}\n".format(len(wines.country.unique()), ", ".join(wines.country.unique()[0:5])))
print("There are {} varities of wine which are {}\n".format(len(wines.variety.unique()), ", ".join(wines.variety.unique()[0:5])))


# ### Finding out how many entries are there per country

# In[ ]:


wines_country = pd.crosstab(index = wines['country'], columns = 'count')
plt.rcParams["figure.figsize"][0] = 10
plt.rcParams["figure.figsize"][1] = 4
wines_country['count'].plot(kind = 'bar',width= 0.7)
plt.title("Graph showing total number of entries of each country", fontsize = 17)
plt.xlabel("Countries", fontsize = 14)
plt.ylabel("Frequency of entries", fontsize = 14)


# <p> So, we find that there are almost all the countries with less than 10000 mark. There are only 3 countries which are more than 20000 out of which USA has more than 50000.The rest 2 countries are France and Italy with values somewhat around 20000, While Armenia seems to get the minimum entries in the dataset</p>
# <p> Next we find out the varities of wine on the basis of number of entries. So we select the values of varities for only more than 1000 entries are available which proves that these wines are popular than others. Then We plot the bar graph for it to find their frequency. </p>

# ### Getting the varities of wine and then plotting their barchart

# In[ ]:


variety = pd.crosstab(index = wines['variety'], columns = 'count').sort_values('count' , ascending = False)
wines_var = variety[variety['count'] > 1000]
wines_var['count'].plot(kind = 'bar', width = 0.7)
plt.title("The number of each variety of wine present", fontsize= 17)
plt.xlabel("The variety of wines that are present", fontsize = 14)
plt.ylabel("The number of wine variety that are present", fontsize = 14)


# ## Analysing the parameters of price and points for classification

# <p> So, now we have already plotted the graphs for the countries and wine varities. Next what we need to know is the 2 important parameters that will actually decide the classification for us. These features are points and price. The points as defined gives the points from 1 to 100 which the reviewer has given to the wine and price defines the price at which it is sold. </p>
# <p> So, it is obviously preassumed that the higher the rating points of the wine the better will be its quality and somewhat same relation is applied with the price. So let us find out the prices and points in the dataset </p> 
# 

# In[ ]:


wines['points'].hist(grid = False, bins = 40,color = 'purple')


# In[ ]:


wines['points'].describe()


# <p> So from here we can clearly see that the graph so formed is a normal graph. It has its peak somewhere in the middle at around 88-89. So we can found all the important points for the graph. And what we found out was the mean of the points given to the varieties were around 88.45. And maximum was 100 which was very few and minimum was 80 which was again very few. So None of the wine quality was ever given a below 80 mark which proves most of the wines were above average for the reviewer. And the Standard deviation of 3.03 suggests that the points given to each wine was 88 +/- 3 which means anything ranging from 85 to 91 which is a very good score.</p>
# <p> Now points is 1 parameter which we can find out with the help of description. but we need to divide it into ranges like the 4 ranges 
#      <ol> 
#          <li> 80 to 85 points </li>
#          <li> 85 to 90 points </li>
#          <li> 90 to 95 points </li>
#          <li> 95 to 100 points </li>
#     </ol>
# This way we can easily do the analysis of the text and classify the wines in according to the points </p>

# In[ ]:


price = pd.crosstab(index = wines['price'], columns = 'count').sort_values(by = 'count', ascending = False)
plt.rcParams['figure.figsize'][0] = 4
plt.rcParams['figure.figsize'][1] = 4
price['count'].hist(grid = False)


# <p> So, what we find here is a completely skewed graph which has prices lying just mainly from 0 to maybe 400. And there are very very little prices above 2000 and maybe negligible like 1 or 2 above 4000. So now let us get the prices which have the value of just 100 or below them in the dataset so that we can have a much more closer look at the data.</p>

# In[ ]:


price_max = wines[wines['price'] < 100]['price']
price_max.hist(grid = False, color = 'green')


# <p> So now this gives a much broader look at the dataset And we find that mostly the prices are from 20 to 40. 100 is also something which the price really touches. So now this means that the costly wines are lesser in number in this dataset.</p>
# <p> So Now for the text classification of the data here we will have to divide the groups into non-uniform groups. Because mainly the wines are below 100 and that too we have mainly below 50. So division can be more clearly understood by further finding the mean and 25th and 75th percentile etc for this graph. </p>

# In[ ]:


price_max.describe()


# <p> So here we find that the mean is around 30 which is much below 50 but we have a very high standard deviation i.e. 18 so the prices actually vary from 48 to 12 and this can also be seen from the graph as it is highly skewed. We have hardly the prices going to 100's So we can divide up the groups as follows : 
#    <ol> 
#        <li> 0-10 </li>
#        <li> 10-20</li>
#        <li> 20-30</li>
#        <li> 30-50</li>
#        <li> 50-100</li>
#        <li>Above 100</li>
#    </ol>
#   <p> Now for the further information like Word Clouds Formation, Text Analysis using NLTK, Various ML Algos for Predictions visit the following link</p>
#   <p> <a href = 'https://www.kaggle.com/bhaargavi/wine-review-classification-making-word-clouds'><h3>Link To Text Classification using NLP for Various types of Wines -- Part 2</h3></a> </p>
#   <p> <a href = 'https://www.kaggle.com/bhaargavi/wine-reviews-classification-using-nlp-and-ml '><h3>Link To Text Classification using NLP for Various types of Wines -- Part 3</h3></a> </p>
# 

# In[ ]:




