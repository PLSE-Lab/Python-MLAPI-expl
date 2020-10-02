#!/usr/bin/env python
# coding: utf-8

# # *Explore the New World Whisky*
# 
# ### Introduction
# As a whisky fan, I always wanted to check out datasets on all the whisky distilleries around the world. Thus I scraped data from a whisky information website called WhiskyDatabase.
# 
# In this report, I am gpoing to do simple EDA of two datasets, one of which is on distilleries and the other on is on whisky brands. These two datasets are unique datasets since data objects came from all around the world. When people talk about whisky, they mostly refer to whisky made in 5 main countries such as Scotland, Ireland, United States, Canada, and Japan. This is because they were the pioneer countries in the whisky industry and have been leading the industry in terms of producing and selling.
# 
# #### What is "New World Whisky"?
# These year, however, whisky lovers are paying attentions to countries apart from the 5 main countries in order to seek different flavor. Thus the popularity of new countries is rapidly growing, and they are now called "New World Whisky".
# So, in this report I am going to focus on new world whisky and hopefully you found this analysis insightful and interesting.
# 
# ![](https://s3-ap-southeast-2.amazonaws.com/koki25ando/Photos/imasia_14333957_S.jpg)
# 
# ### Data Source
# [WhiskyDatabase](https://www.whiskybase.com/):
# 
# ### Research Questions
# - What Coutries are producing whisky the most?
# - Which new world whisky should I try and recommend others?

# In[ ]:


# Preparation
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Data Import & Cleaning
Distillery = pd.read_csv("../input/world-whisky-distilleries-brands-dataset/Distillery.csv")
Distillery = Distillery.iloc[:, 1:]

Country_Code = pd.read_csv("../input/country-code/country_code.csv")
Country_Code = Country_Code.iloc[:, 1:]

Brand = pd.read_csv("../input/world-whisky-distilleries-brands-dataset/Whisky_Brand.csv")
Brand = Brand.iloc[:, 1:7]


# In[ ]:


# Data Content
Distillery.head()


# In[ ]:


Distillery.info()


# In[ ]:


type(Distillery)


# In[ ]:


Distillery.shape


# In[ ]:


Distillery['Rating'] = pd.to_numeric(Distillery['Rating'], errors='coerce')
Distillery = Distillery.dropna(subset=['Rating'])
sns.distplot(Distillery['Rating'], kde = False, bins = 90)


# As you noticed, there are so many ditilleries whose ratings are 0.
# This suggests that there are so many tiny or newly constructed distilleries. However, in this report, I am not going to focus on the distilleries. Thus I am going to omit them.

# In[ ]:


Distillery = Distillery[(Distillery['Rating'] > 0)]
fig, ax =plt.subplots(1,2, figsize=(20,8))
sns.distplot(Distillery['Rating'], kde = True, bins = 90, ax = ax[1])
sns.boxplot(x = Distillery['Rating'], ax = ax[0])
plt.suptitle("Distribution of Rating Score", fontsize = 30)
plt.show()


# As the 2 plots show, rating scores are distributed around 80.

# In[ ]:


Country = Distillery['Country'].value_counts()
Country = pd.DataFrame({'Country':Country.index,'Count':(Country.values)})
barplot1 = [go.Bar(
            x=Country['Country'],
            y=Country['Count']
    )]
layout = go.Layout(
    title = 'Distillery Number of Each Country'
)
fig = go.Figure(data = barplot1, layout = layout)
iplot(fig)


# The plot is showing the number of distilleries located in each country around the world.
# United States has by far the most distilleries among all the countries. It is not surprising that the country is one of the main whisky producing countries and their whisky are well known as "Bourbon Whisky".
# One thing we notice from the plot is that Germany ranks the 2nd place in the world. However, if you know familiar with how whiskies are made, this result would not be surprising at all. I am not going to talk about this topic deeply but it is pretty well known that the process of producing whisky is pretty similar to the one of producing beer. Since they both have common techniques and obviously Germany is a country which is famous for the people love beer a lot, this is why the number of distilleries is high and most of them could be beer breweries as well.
# 
# From now on, to focus a little bit more on new world whisky, I am going to delete data on distilleries in 5 main countries.

# In[ ]:


NW_Distillery = Distillery[(Distillery['Country'] != "Japan") & (Distillery['Country'] != "Scotland") & 
                                     (Distillery['Country'] != "United States") & (Distillery['Country'] != "Canada") & 
                                     (Distillery['Country'] != "Ireland")]
Country = NW_Distillery['Country'].value_counts()
Country = pd.DataFrame({'Country':Country.index,'Count':(Country.values)})

barplot2 = [go.Bar(
            x=Country['Country'],
            y=Country['Count']
    )]
layout = go.Layout(
    title = 'Distillery Number of Each New World Whisky Country'
)
fig = go.Figure(data = barplot2, layout = layout)
iplot(fig)


# As we saw previous bar chart, German is the leading country. 
# Other top countries are mostly european countries such as Austria, France, Netherland, etc...

# In[ ]:


Country_Rating = NW_Distillery.groupby("Country").mean()["Rating"]
Country_Rating = Country_Rating.to_frame()
Country_Rating['Country'] = Country_Rating.index
Country_Rating = Country_Rating.sort_values(by = 'Rating', ascending=False)
fig = plt.figure(figsize=(10,10))
sns.barplot(
    data = Country_Rating,
    x = "Rating",
    y = "Country"
)
plt.axvline(60, color = 'r')
plt.title("Average Rating Scores of Whisky Disitilleries for each Country", fontsize = 20)
fig.show()


# Based on the average rating scores for each country, I am going to focus on countries whose ratings are above 60 points. I chose them because one of purposes of this analysis is finding out good whiskies I would suggest people.

# In[ ]:


Top_NW_Country = Country_Rating[Country_Rating['Rating'] > 60]
Top_NW_Country = Top_NW_Country.reset_index(drop = True)
Top_NW_Country = pd.DataFrame(Top_NW_Country, columns = ['Country', 'Rating'])
Top_NW_Country_name = Top_NW_Country['Country']
Top_NW_Brand = Brand.loc[Brand['Country'].isin(['Taiwan','Israel', 'Mexico', 'United Kingdom', 
                                                'Australia', 'South Africa', 'Finland', 'Liechtenstein', 
                                                'Bhutan', 'Sweden', 'Sweden', 'Denmark', 'Switzerland', 
                                                'Czech Republic','Norway', 'Slovakia', 'France', 
                                                'Indonesia', 'Spain', 'Germany', 'Hungary', 'Italy',
                                                'India', 'Austria', 'Netherlands', 'Belgium',
                                                'Luxembourg', 'Iceland', 'Turkey', 'New Zealand', 
                                                'Poland', 'Netherlands Antilles', 'Serbia And Montenegro'])]
Top_NW_Brand.info()


# Now we have data of 742 whisky brands produced by top rated countries.
# Since I would like to find out recommendable whisky based on rating scores, I am going to remove objects without rating record. 

# In[ ]:


Top_NW_Brand = Top_NW_Brand[np.isfinite(Top_NW_Brand['Rating'])]
Top_NW_Brand = Top_NW_Brand[Top_NW_Brand['Votes'] > 10]
fig = plt.figure(figsize = (10,6))
sns.boxplot(
    data = Top_NW_Brand,
    x = 'WB Ranking',
    y = 'Rating'
)
plt.axhline(80, color = 'red')
plt.suptitle("Rating Band", fontsize = 20)
fig.show()


# In[ ]:


Top_NW_Brand = Top_NW_Brand[Top_NW_Brand['Rating'] > 80].sort_values(by = ['Rating'], ascending = False)
fig = plt.figure(figsize = (10,10))
sns.barplot(x="Rating", y="Brand", hue="Country",
                 data=Top_NW_Brand, dodge=False)
plt.title("Top New World Whisky Brands", fontsize = 20)


# So, after all, here are my new world whisky recommendation list.
# Let me take a closer look at top 5 brands.

# In[ ]:


Top_NW_Brand.head(5)


# A brand called "The Alrik" has marked the best rating score among the other. And it is the only brand from "B" WB ranking.
# ![](https://s3-ap-southeast-2.amazonaws.com/koki25ando/Photos/Alrik.png)

# ### Conclusion
# 
# In this report, I have found 
# - the number of distilleries around the world
# - leading countries of whisky producer
# - new world whisky countries
# - New World whisky brands I would recommend others.
# 
# Although this was my first time python kernel, I have really enjoyed exporing the interesting datasets I have collected.
# If you have found this analysis interesting, please try your own analysis and share. Your contributions are always welcome.
# 
# Also, if you are interested in, make sure to check out other whisky-related datasets I have uploaded before.
# - [22,000+ Scotch Whisky Reviews](https://www.kaggle.com/koki25ando/22000-scotch-whisky-reviews)
# - [Japanese Whisky Review Dataset](https://www.kaggle.com/koki25ando/japanese-whisky-review)
# - [Scotch Whisky Dataset](https://www.kaggle.com/koki25ando/scotch-whisky-dataset)
