#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Project-Background" data-toc-modified-id="Project-Background-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Project Background</a></span></li><li><span><a href="#Data-Understanding" data-toc-modified-id="Data-Understanding-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Understanding</a></span><ul class="toc-item"><li><span><a href="#Describe-the-dataset" data-toc-modified-id="Describe-the-dataset-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Describe the dataset</a></span></li><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Data Cleaning</a></span><ul class="toc-item"><li><span><a href="#Drop-the-missing-value" data-toc-modified-id="Drop-the-missing-value-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Drop the missing value</a></span></li><li><span><a href="#Remove-the-duplicate-items" data-toc-modified-id="Remove-the-duplicate-items-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Remove the duplicate items</a></span></li><li><span><a href="#Remove-outliers" data-toc-modified-id="Remove-outliers-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Remove outliers</a></span></li><li><span><a href="#Correct-the-data-type" data-toc-modified-id="Correct-the-data-type-2.2.4"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>Correct the data type</a></span></li><li><span><a href="#Save-to-database" data-toc-modified-id="Save-to-database-2.2.5"><span class="toc-item-num">2.2.5&nbsp;&nbsp;</span>Save to database</a></span></li></ul></li></ul></li><li><span><a href="#Recommendation-Analysis" data-toc-modified-id="Recommendation-Analysis-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Recommendation Analysis</a></span><ul class="toc-item"><li><span><a href="#Descriptive-statistics" data-toc-modified-id="Descriptive-statistics-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Descriptive statistics</a></span><ul class="toc-item"><li><span><a href="#Histogram-of-numerical-columns" data-toc-modified-id="Histogram-of-numerical-columns-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Histogram of numerical columns</a></span></li><li><span><a href="#Top-10-popular-breweries" data-toc-modified-id="Top-10-popular-breweries-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Top 10 popular breweries</a></span></li><li><span><a href="#Top-10-brewery-with-the-most-beer-types" data-toc-modified-id="Top-10-brewery-with-the-most-beer-types-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Top 10 brewery with the most beer types</a></span></li><li><span><a href="#Top-10-popular-beers" data-toc-modified-id="Top-10-popular-beers-3.1.4"><span class="toc-item-num">3.1.4&nbsp;&nbsp;</span>Top 10 popular beers</a></span></li><li><span><a href="#Top-10-beers-with-highest-rating" data-toc-modified-id="Top-10-beers-with-highest-rating-3.1.5"><span class="toc-item-num">3.1.5&nbsp;&nbsp;</span>Top 10 beers with highest rating</a></span></li><li><span><a href="#The-Top-10-popular-beer-styles" data-toc-modified-id="The-Top-10-popular-beer-styles-3.1.6"><span class="toc-item-num">3.1.6&nbsp;&nbsp;</span>The Top 10 popular beer styles</a></span></li></ul></li><li><span><a href="#Get-the-two-recommendations" data-toc-modified-id="Get-the-two-recommendations-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Get the two recommendations</a></span></li><li><span><a href="#Additional-exploration-of-the-dataset" data-toc-modified-id="Additional-exploration-of-the-dataset-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Additional exploration of the dataset</a></span><ul class="toc-item"><li><span><a href="#The-correlation-of-review-features" data-toc-modified-id="The-correlation-of-review-features-3.3.1"><span class="toc-item-num">3.3.1&nbsp;&nbsp;</span>The correlation of review features</a></span></li><li><span><a href="#The-time-series-line-chart-of-reviews" data-toc-modified-id="The-time-series-line-chart-of-reviews-3.3.2"><span class="toc-item-num">3.3.2&nbsp;&nbsp;</span>The time-series line chart of reviews</a></span></li></ul></li></ul></li></ul></div>

# # Project Background
# Question definition:
# - To get two recommendations using the dataset, and report to the HR department for weekly report.
# - To discorver interesting insights from the dataset
# 

# #  Data Understanding
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly
import plotly.figure_factory as ff


# In[ ]:


df = pd.read_csv("../input/beerreviews/beer_reviews.csv")
df.head(5)


# ## Describe the dataset
# 

# In[ ]:


print("types of each columns: \n\n",df.dtypes)
print("\ninformation of the columns: \n")
print(df.info())


# In[ ]:


print("Count of unique breweries, by brewery_id: " ,df.brewery_id.nunique())
print("Count of unique breweries, by brewery_name: " ,df.brewery_name.nunique())


# In[ ]:


print("Count of unique beers, by beer_id: " ,df.beer_beerid.nunique())
print("Count of unique beers, by beer_name: " ,df.beer_name.nunique())


# In[ ]:


print("Count of unique users, by review_profilename: " ,df.review_profilename.nunique())


# From the information above, we can understand the follows:
# - Columns of the dataset:
#   - Information of the brewery: brewery_id, brewery_name
#   
#   - Information of the beer: beer_name,beer_beerid,beer_abv,beer_style
#   
#   - Information of the reviewer: review_profilename
#   
#   - All about review: review_time review_overall, review_aroma, review_appearance, review_palate, review_taste
#   
#   
#   
#   
# 
#   
#   
# - This dataset contains 1.5 million+ reviews of around 60,000 beers from 5000+ wineries.
# 
# 
# 
# 
# 
# 
# - The review_time column needs to convert the data type from int to datetime.

# ## Data Cleaning
# ### Drop the missing value 
# 

# In[ ]:


print("Overview of missing values in the dataset: /n",df.isnull().sum())


# In[ ]:


df=df.dropna()
print("After dropping the missing value",df.info())


# ### Remove the duplicate items

# In[ ]:


print("a user review the same beer more than one time, by beer_beerid: \n",df.loc[df.duplicated(['review_profilename','beer_beerid'],keep=False)][['review_profilename','beer_name','beer_beerid','review_overall']])


# In[ ]:


print("a user review the same beer more than one time, by beer_name: \n",df.loc[df.duplicated(['review_profilename','beer_name'],keep=False)][['review_profilename','beer_beerid','beer_name',"review_overall"]])


# In this case, some users reviewed the same beer more than one time. We would keep the highest rating that user gave.

# In[ ]:


df = df.sort_values("review_overall",ascending=False)
df = df.drop_duplicates(subset = ['review_profilename','beer_beerid'],keep='first')
df = df.drop_duplicates(subset = ['review_profilename','beer_name'],keep='first')


# In[ ]:


df.info()


# After dropping the duplicated reviews, there are still 1496263 reviews left in the dataset.

# ### Remove outliers

# In[ ]:


round(df.describe(),2)


# The rating score should be scaled between 1 to 5,however,in the data frame described above, we can find that there are reviews with a score below 1 in the column review_overall and review_apperance. Therefore, we should remove the outliers.

# In[ ]:


df = df.loc[(df.review_overall>=1) & (df.review_appearance>=1)]
round(df.describe(),2)


# ### Correct the data type

# In[ ]:


df.review_time = pd.to_datetime (df.review_time,unit = 's')


# In[ ]:


df.dtypes


# # Recommendation Analysis

# ## Descriptive statistics

# ### Histogram of numerical columns

# In[ ]:


df.hist(figsize=(15,15),color='#007399')


# - beer_abv(right long-tailed distribution),most of the beers are less than 20% abv.
# - review_appearance (normal distribution), most beers are rated between 3.5 and 4.5.
# - review_aroma (normal distribution), most beers are rated between 3.5 and 4.5.
# - review_plate (normal distribution), most beers are rated between 3.5 and 4.5.
# - review_taste (normal distribution), most beers are rated between 3.5 and 4.5.
# - review_overall (normal distribution), As review_overall is the mean of review_appearance, review_aroma, review_plate and review_taste, the overall rate of beers are between 3.5.to 4.5
# - beer_beerid (right long-tailed distribution), beers with lower id own more reviews.
# - brewery_id (right long-tailed distribution), breweries with lower id own more reviews.

# ### Top 10 popular breweries

# In[ ]:


bar = go.Bar(x=df.brewery_name.value_counts().head(10).sort_values(ascending=True),
             y=df.brewery_name.value_counts().head(10).sort_values(ascending=True).index,
             hoverinfo = 'x',
             text=df.brewery_name.value_counts().head(10).sort_values(ascending=True).index,
             textposition = 'inside',
             orientation = 'h',
             opacity=0.75, 
             marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='The Top 10 popular breweries',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar, layout=layout)

# Plot it
plotly.offline.iplot(fig)


# ###  Top 10 brewery with the most beer types

# In[ ]:


brewery_type = df.groupby('brewery_name')
brewery_type = brewery_type.agg({"beer_name":"nunique"})
brewery_type = brewery_type.reset_index()


bar2 = go.Bar(x=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).beer_name,
              y=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).brewery_name,
              hoverinfo = 'x',
              text=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).brewery_name,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.75, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='Top 10 brewery with the most beer types',
                   xaxis=dict(title="Count of beer types",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar2, layout=layout)

# Plot it
plotly.offline.iplot(fig)


# ### Top 10 popular beers

# In[ ]:


bar3 = go.Bar(x=df.beer_name.value_counts().head(10).sort_values(ascending=True),
              y=df.beer_name.value_counts().head(10).sort_values(ascending=True).index,
              hoverinfo = 'x',
              text=df.beer_name.value_counts().head(10).sort_values(ascending=True).index,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.75, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='Top 10 popular beers',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar3, layout=layout)

plotly.offline.iplot(fig)


# ### Top 10 beers with highest rating

# In[ ]:


rate_beer = df[['beer_name','review_overall']].groupby('beer_name').agg('mean')

rate_beer = rate_beer.reset_index()

rate_beer
bar4 = go.Bar(x=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).review_overall,
              y=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).beer_name,
              hoverinfo = 'x',
              text=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).review_overall,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.75, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='Top 10 beers with highest rating',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar4, layout=layout)

plotly.offline.iplot(fig)


# In[ ]:


aa=list(rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).beer_name)
df[df['beer_name'].isin(aa)].groupby("beer_name").agg("count").reset_index().beer_name.value_counts()


# ### The Top 10 popular beer styles

# In[ ]:


bar5 = go.Bar(x=df.beer_style.value_counts().head(10).sort_values(ascending=True),
              y=df.beer_style.value_counts().head(10).sort_values(ascending=True).index,
              hoverinfo = 'x',
              text=df.beer_style.value_counts().head(10).sort_values(ascending=True).index,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.75, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='The Top 10 popular beers styles',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar5, layout=layout)

# Plot it
plotly.offline.iplot(fig)


# The bar plots list the Top 10 popular breweries, beer, and beer types, as well as the Top 10 breweries who produce the most types of beer.
# 
# It is worth noting that the list of the TOP 10 popular breweries and the TOP 10 breweries that produce the most beer types do not overlap.
# 
# Is there a correlation between the popularity of the brewery and the type of beer produced by the brewery?

# In[ ]:


popular= df.brewery_name.value_counts().sort_index()
popular = popular.reset_index()
print("The correlation of reviews and beer types: ",brewery_type.beer_name.corr(popular.brewery_name))


# In[ ]:


### Distribution of beer_abv


# In[ ]:


plt.figure(figsize=(10,8))
plt.title("Distribution of beer abv")
sns.distplot(df.beer_abv)
plt.xlabel("beer abv %")


# ## Get the two recommendations

# In[ ]:


print("Review count of each beer \n ",df.beer_name.value_counts().describe())


# In[ ]:


sns.distplot(df.beer_beerid.value_counts(),kde=False)
plt.xlabel("beer_id")
plt.ylabel("count of reviews")
plt.title("distribution of beer's reviews")
plt.show()


# In[ ]:


reshape=df[['review_overall','beer_name']].groupby("beer_name").agg(['count','mean'])
print("Beers with review_overall more than 4: \n",reshape[reshape['review_overall',  'mean']>4])
print("Beers with review_overall more than 4, and number of review less than 30: \n",reshape[reshape['review_overall',  'mean']>4][reshape[reshape['review_overall',  'mean']>4]['review_overall',  'count']<30])


#  From the information above we can understand:
# - Although there are some beers with 3000+ reviews, however, half of the beers are with review less than 3.
# 
# 
# - There are more than 9000 beers with a rating of more than 4, however, 7754 of them are with less than 30 reviews.
# 
# 
# 
# As we assume that when the number of reviews is less than 30, the beer's rating is biased.
# Therefore, in the following steps, I will filter out beers with less than 30 reviews, then pick the top 2 beers with the highest rate.

# Based on the TOP rating we got from the last section, I will choose the two beer with the highest average review_overall from the TOP 10 popular beer types list and produced by TOP 10 popular breweries for recommendation.

# In[ ]:


top10_breweries=df.brewery_name.value_counts().head(10).reset_index()
top10_styles=df.beer_style.value_counts().head(10).reset_index()
subset = df[df['brewery_name'].isin(top10_breweries['index'])& df['beer_style'].isin(top10_styles['index'])]


# In[ ]:


reshaped_subset = subset[['review_overall','beer_name']].groupby("beer_name").agg(['count','mean'])
reshaped_subset = reshaped_subset[reshaped_subset['review_overall',  'count']>30]
reshaped_subset.columns
reshaped_subset.sort_values(('review_overall',  'mean'),ascending=False).head(2)


# In[ ]:


categories=['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']
r1=df[df.beer_name=="Founders CBS Imperial Stout"]
r2=df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"]
r1_value=[r1.review_overall.mean(),r1.review_aroma.mean(),r1.review_appearance.mean(),r1.review_palate.mean(),r1.review_taste.mean()]
r2_value=[r2.review_overall.mean(),r2.review_aroma.mean(),r2.review_appearance.mean(),r2.review_palate.mean(),r2.review_taste.mean()]

mean_value=[df.review_overall.mean(),df.review_aroma.mean(),df.review_appearance.mean(),df.review_palate.mean(),df.review_taste.mean()]


# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=r1_value,
      theta=categories,
      fill='toself',
      name='Founders CBS Imperial Stout'
))

fig.add_trace(go.Scatterpolar(
      r=mean_value,
      theta=categories,
      fill='toself',
      name='Overall_mean'
))

fig.update_layout(title="Radar chart of review features - Founders CBS Imperial Stout",
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),
  showlegend=True
)

fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r=r2_value,
      theta=categories,
      fill='toself',
      name='Founders KBS (Kentucky Breakfast Stout)'
))
fig.add_trace(go.Scatterpolar(
      r=mean_value,
      theta=categories,
      fill='toself',
      name='Overall_mean'
))
fig.update_layout(title="Radar chart of review features - Founders KBS (Kentucky Breakfast Stout)",
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 5]
    )),
  showlegend=True
)

fig.show()


# In[ ]:


print("Breweries of the beer recommendations: ",df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"].brewery_name.unique(),df[df.beer_name=="Founders CBS Imperial Stout"].brewery_name.unique())
print("Styles of the recommendations: ",df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"].beer_style.unique(),df[df.beer_name=="Founders CBS Imperial Stout"].beer_style.unique() )


# In[ ]:


df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts()


# In[ ]:


bar5 = go.Bar(x=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True),
              y=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True).index,
              hoverinfo = 'x',
              text=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True).index,
              textposition = 'inside',
              orientation = 'h',
              opacity=0.75, 
              marker=dict(color='rgb(1, 77, 102)'))

layout = go.Layout(title='The Top 15 bestsellers of Founders Brewing Company',
                   xaxis=dict(title="Count of reviews",),
                   margin = dict(l = 220),
                   font=dict(family='Comic Sans MS',
                            color='dark gray'))

fig = go.Figure(data=bar5, layout=layout)

# Plot it
plotly.offline.iplot(fig)


# ## Additional exploration of the dataset

# ### The correlation of review features
# 
# The review_overall is calculated by review_appearance, review_aroma, review_palate, review_taste. We use these features to measure whether a beer is worth recommending. 
# 
# However, we are not sure whether review_overall is the mean of the other four features. Maybe some of these features are more important and deserve a higher weight in the review_overall. Therefore, I will do a correlation heatmap of the review features and learn something from it.

# In[ ]:


corr= df[["review_appearance","review_aroma","review_palate","review_taste", "review_overall"]].corr()
corr
x=list(corr.index)
y=list(corr.columns)

fig = ff.create_annotated_heatmap(x=x,y=y,z=corr.values.round(2),colorscale=[[0, 'navy'], [1, 'plum']],font_colors = ['white', 'black'])

fig.show()


# The heatmap above shows the correlation of review features.
# The lighter the color, the higher the correlation between the two features.
# 
# We can get the information as follows:
# - review_taste has the highest correlation with review_overall, at 0.79, followed by review_palate, at 0.7. 
# - the palate and taste of a beer show a high relevance at 0.73.
# - Although all of the four features show a strong correlation with the overall rating, due to the different correlation coefficients, different weights need to be given to a beer when calculating the overall rating.

# ### The time-series line chart of reviews
# 
# In the above analysis, we ignored the column "review_time". Time-series data always contains a lot of information describing the trend. Here, I will make a line chart to see how the number of reviews changes over time.

# In[ ]:


time=df["review_time"].groupby(df.review_time.dt.date).agg('count')
fig = go.Figure(data=go.Scatter(x=time.index, y=time.values))
fig.update_layout(title='The time-series line chart of reviews',
                   xaxis_title='Date',
                   yaxis_title='Count of reviews')
fig.show()


# The beer review data was first generated in 1998, but the number of daily beer reviews only began to increase in 2002, and reached a peak in early 2011, about 1,000 reviews per day.

# In[ ]:




