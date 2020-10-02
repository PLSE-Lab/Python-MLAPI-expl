#!/usr/bin/env python
# coding: utf-8

# ##### Dataset column description
# <b>country:</b> The country that the wine is from<br>
# <b>description:</b> The review given by the taster<br>
# <b>designation:</b> The vineyard within the winery where the grapes that made the wine are from<br>
# <b>points:</b> The number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)<br>
# <b>price:</b> The cost for a bottle of the wine<br>
# <b>province:</b> The province or state that the wine is from<br>
# <b>region_1:</b> The wine growing area in a province or state (ie Napa)<br>
# <b>region_2:</b> Sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank<br>
# <b>taster_name</b><br>
# <b>taster_twitter_handle</b><br>
# <b>title: </b>The title of the wine review, which often contains the vintage if you're interested in extracting that feature<br>
# <b>variety:</b> The type of grapes used to make the wine (ie Pinot Noir)<br>
# <b>winery:</b> The winery that made the wine<br>

# #### <font color="red">Note: I have first done some data munging and then the visualization.</font>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# The dataset has 2 different types of data files available
# Let's see what is the difference between them before we proceed..
wine_130k = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
wine_150k = pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)


# In[ ]:


wine_130k.info()


# In[ ]:


wine_150k.info()


# In[ ]:


# So the data file with 130k reviews has 3 more columns than the 150k file, let's see what extra columns are there..
for column in wine_130k.columns:
    if(column not in wine_150k.columns):
        print(column)


# Since the 150k dataset has 3 less columns and therefore has less information on wine reviews, so for the sake of visual analysis I am using only the 130k dataset for further process.

# In[ ]:


wine_130k.head()


# In[ ]:


wine_130k.describe()
# We have only two numerical values in the dataset: points, price


# Before we proceed, let's deal with missing values first. <br>
# Also the names of a few columns are a bit misleading, let's change that too..

# In[ ]:


wine_130k.rename(index=str, columns={"designation":"vineyard", "variety":"grape_variety"}, inplace=True)
wine_130k.info()


# # <font color="red">Data Munging</font>

# ### Missing values in 'country'

# In[ ]:


wine_130k.loc[wine_130k.country.isnull()]


# In[ ]:


wine_130k.groupby('country').winery.value_counts()


# <b>I'm creating a list of mappings to country based on winery.
# I'll use that to fill the missing country values.</b>

# In[ ]:


country_by_winery = wine_130k.groupby('winery').country.unique().to_json()
import json
country_by_winery = json.loads(country_by_winery)


# In[ ]:


country_by_winery


# In[ ]:


# Function for getting country respective to the winery
def GetCountry(w):
    cntry = country_by_winery[w]
    if(None in cntry):
        cntry = cntry.remove(None)
    return cntry[0] if cntry else np.NaN


# In[ ]:


#GetCountry('Famiglia Meschini')
country_by_winery['Ross-idi']


# <b>Here, we can see in the above line that some wineries don't have a single country value. <br>I'll use the country that has max number of wineries as the value to fill the remaining null values..</b>

# In[ ]:


wine_130k.country.value_counts()


# In[ ]:


wine_130k['country_by_winery'] = wine_130k.winery.map(GetCountry)
# US has the most number of wineries
wine_130k.country_by_winery.fillna('US', inplace=True)
wine_130k.country.fillna(wine_130k.country_by_winery, inplace=True)
wine_130k.drop('country_by_winery', axis=1, inplace=True)


# ## Missing values in 'price'

# In[ ]:


wine_130k.price.value_counts()


# In[ ]:


log_price = np.log(wine_130k.price)
log_price.plot.hist(title='Price distribution histogram', bins=20, color='tomato', edgecolor='black', figsize=(10,7));


# In[ ]:


sns.set(style="whitegrid")
plt.figure(figsize=(7,7))
sns.boxplot(wine_130k.price);


# <b>From the two plots above, we can see that price is extremely right skewed and many price values lie outside the 1.5*IQR. <br>
# Since, mean is a bad choice in case of skewed data, I'm using median of price to fill the missing values.</b>

# In[ ]:


wine_130k.groupby('country').price.median().plot.line(title='Median distribution of price by country', fontsize=60, figsize=(100,30), linewidth=3.3, color='red');


# <b>The above line plot shows that the median values of price are more or less similar for the countries <br>
# Filling missing values by medians based on country</b>

# In[ ]:


median_price = wine_130k.groupby('country').price.transform('median')
wine_130k.price.fillna(median_price, inplace=True)


# In[ ]:


wine_130k[wine_130k.price.isnull()]


# In[ ]:


# 'Egypt' has only one review and has no price value, using overall price median to fill
wine_130k.price.fillna(wine_130k.price.median(), inplace=True)


# ## Missing values in 'taster_name'

# In[ ]:


wine_130k[wine_130k.taster_name.isnull()]


# In[ ]:


wine_130k['taster_name'].value_counts()


# In[ ]:


# List of countries with taster that has maximum reviews in that specific country
taster_by_country = wine_130k.groupby('country').apply(lambda x: x['taster_name'].value_counts().index[0])
taster_by_country


# In[ ]:


wine_130k['taster_by_country'] = wine_130k[wine_130k.taster_name.isnull()].country.map(taster_by_country)
wine_130k.taster_name.fillna(wine_130k['taster_by_country'], inplace=True)
wine_130k.drop('taster_by_country', axis=1, inplace=True)


# ## Missing values in 'grape_variety'

# In[ ]:


wine_130k[wine_130k.grape_variety.isnull()]


# In[ ]:


wine_130k[wine_130k.winery=='Carmen'].grape_variety.value_counts()


# In[ ]:


# Filling with the max number of variety
wine_130k.grape_variety.fillna('Cabernet Sauvignon', inplace=True)


# <b> Since I am doing only visual analysis here, I won't be dealing with any other missing values further<br>

# <br><br>
# # <font color="red">Visualizations, analysis and inferences</font>

# In[ ]:


# How the dataset looks like now:
wine_130k.info()


# In[ ]:


wine_130k.describe()


# In[ ]:


print('Avegare Wine price: ', wine_130k.price.mean())
print('Median Wine price: ', wine_130k.price.median())


# From the mean and median price values, we can see that there is quite a difference. This means that the price data is skewed and mean is generally not a good measure of centrality value. So I might consider using median as the price centrality measure. Let's confirm this by plotting some graphs ahead.

# In[ ]:


wine_130k.taster_name.value_counts()


# In[ ]:


reviews_per_taster = wine_130k['taster_name'].value_counts().plot.bar(title='Review Distribution by Taster', fontsize=14, figsize=(12,8), color='tomato');
plt.xlabel('Taster')
plt.ylabel('Number of reviews')
plt.show()


# From the above plot, we can see that Virgine Boone and Roger Voss are the top two most active wine reviewers

# In[ ]:


reviews_per_country = wine_130k['country'].value_counts().plot.bar(title='Review Distribution by Countries', fontsize=14, figsize=(17,10), color='c');
plt.xlabel('Country')
plt.ylabel('Number of reviews')
plt.show()


# In[ ]:


plt.figure(figsize=(25,10))
wine_130k.groupby('country').max().sort_values(by="points",ascending=False)["points"].plot.bar(fontsize=17)
plt.xticks(rotation=70)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()


# In[ ]:


price_box = wine_130k.price.plot.box(title='Price Boxplot', figsize=(10,7));


# The price boxplot shows us that there are a few high priced wines, but since the median is quite low(near to 0 in figure), maximum number of wines have low-average price.

# In[ ]:


price_hist = wine_130k.price.plot.hist(title='Price Distribution by intervals', bins=30 ,figsize=(7,7));
plt.xlabel('Price')
plt.ylabel('Reviews')
plt.show()


# The price distribution is highly right skewed, so we need some sort of transformation to better visualize it

# In[ ]:


print("Skewness of Price: %f" % wine_130k['price'].skew())


# In[ ]:


log_price_hist = log_price.plot.hist(title='Price(log) distribution histogram', bins=20, color='orange', edgecolor='black', figsize=(10,7));
plt.xlabel('Price')
plt.ylabel('Reviews')
plt.show()


# In[ ]:


points_bar = wine_130k['points'].value_counts().sort_index().plot.bar(title='Points Distribution', figsize=(15,10), color='firebrick')
plt.xlabel('Points')
plt.ylabel('Number of reviews')
plt.show()


# In[ ]:


print('Min points:', wine_130k.points.min())
print('Max points:', wine_130k.points.max())


# Every wine is allotted an overall score between 80 and 100

# In[ ]:


sns.violinplot(wine_130k['price'], wine_130k['taster_name'], figsize=(15,15)) #Variable Plot
sns.despine()


# In[ ]:


fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(wine_130k['points'],wine_130k['price'], alpha=0.15)
plt.title('Points vs Price')
plt.xlabel('Points')
plt.ylabel('Price')
plt.show()


# Using the alpha property trick in the above scatterplot, we can really see where most of the data points are situated.

# In[ ]:


plt.figure(figsize=(15,10))
price_points_box = sns.boxplot(x="points", y="price", data = wine_130k);
plt.title('Points Boxplot')
plt.show()


# In[ ]:


plt.rc('xtick', labelsize=20)     
plt.rc('ytick', labelsize=20)
fig = plt.figure(figsize=(30,24))
ax = fig.add_subplot(1,1,1)
province_not_null = wine_130k[wine_130k.province.notnull()]
ax.scatter(province_not_null['points'],province_not_null['country'], s=province_not_null['price'])
plt.xlabel('Points')
plt.ylabel('Country')
plt.show()


# In[ ]:


df1= wine_130k[wine_130k.grape_variety.isin(wine_130k.grape_variety.value_counts().head(5).index)]
plt.figure(figsize=(15,10))
sns.boxplot(
    x = 'grape_variety',
    y = 'points',
    data = df1,     
);


# In[ ]:


sns.set()
columns = ['price', 'points']
plt.figure(figsize=(20,20))
sns.pairplot(wine_130k[columns],size = 10 ,kind ='scatter',diag_kind='kde')
plt.show()


# In[ ]:


wine_130k.pivot_table(index='country', columns='taster_name', values='points', aggfunc='mean')


# In[ ]:


from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import requests
import urllib


# In[ ]:


wine_mask = np.array(Image.open(requests.get("http://www.clker.com/cliparts/4/7/5/6/11949867422027818929wine_glass_christian_h._.svg.hi.png", stream=True).raw))


# In[ ]:


title_text = " ".join(t for t in wine_130k['title'])


# In[ ]:


title_wordcloud = WordCloud(width = 1024, height = 1024, background_color='#D1310F', mask=wine_mask, contour_width=1, contour_color='black').generate(title_text)


# In[ ]:


plt.figure( figsize=(30,15), facecolor = 'white' )
plt.imshow(title_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


# title_wordcloud.to_file("title.png")


# In[ ]:


review_text = " ".join(review for review in wine_130k.description)
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])


# In[ ]:


reviews_wordcloud = WordCloud(width=1024, height=1024, stopwords=stopwords, max_words=1000, background_color="#D8F2EE").generate(review_text)


# In[ ]:


plt.figure( figsize=(30,15), facecolor = 'white' )
plt.imshow(reviews_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


#reviews_wordcloud.to_file("reviews.png")


# In[ ]:


mean_points = wine_130k.groupby('country').points.mean()
mean_points


# In[ ]:


mean_points.plot.line(title='Mean points per Country', figsize=(15,7));


# In[ ]:


top_vineyards = wine_130k.groupby('vineyard').points.median().nlargest(15)


# In[ ]:


top_vineyards.plot.line(rot=80, title='Top rated Wines', figsize=(10,7), fontsize=12, color='tomato');


# In[ ]:


costly_vineyards = wine_130k.groupby('vineyard').price.median().nlargest(15)
costliest_wine = costly_vineyards.plot.bar(figsize=(10,7), fontsize=12, color='firebrick')
plt.title('Costliest Wines by Vineyard')
plt.show()


# In[ ]:


costly_province = wine_130k.groupby('province').price.median().nlargest(15)
costliest_wine_by_province = costly_province.plot.bar(figsize=(10,7), fontsize=12, color='c')
plt.title('Costliest Wines by Province')
plt.show()


# In[ ]:


costly_region = wine_130k.groupby(['region_1', 'region_2']).price.median().nlargest(15)
costliest_wine_by_region = costly_region.plot.bar(figsize=(10,7), fontsize=12, color='skyblue')
plt.title('Costliest Wines by Region')
plt.show()


# In[ ]:


costly_title = wine_130k.groupby(['title']).price.median().nlargest(15)
costliest_wine_by_title = costly_title.plot.area(rot=80, figsize=(10,7), fontsize=12, color='pink')
plt.title('Costliest Wines by Title')
plt.show()


# In[ ]:


# f, (ax1,ax2) = plt.subplots(1, 2, figsize=(50,20))

top_grape_points = wine_130k.groupby('grape_variety').points.median().nlargest(15)
pl1 = top_grape_points.plot.bar(figsize=(10,7), fontsize=12, color='firebrick')
plt.title('Top Rated Wine by Grape Variety')
plt.ylabel('Median Points')
plt.show()


# In[ ]:


top_grape_price = wine_130k.groupby('grape_variety').price.median().nlargest(15)
pl2 = top_grape_price.plot.bar(figsize=(10,7), fontsize=12, color='orange')
plt.title('Top Rated Wine by Grape Variety')
plt.ylabel('Median Price')
plt.show()


# In[ ]:


top_winery_vineyard = wine_130k.groupby(['winery', 'vineyard']).price.median().nlargest(15)
costliest_wine_winery_vineyard = top_winery_vineyard.plot.line(rot=90, figsize=(10,7))
plt.title('Top Priced Wine by Winery and Vineyard')
plt.ylabel('Median Price')
plt.show()


# In[ ]:


top_winery_vineyard

