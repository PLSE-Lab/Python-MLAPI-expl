#!/usr/bin/env python
# coding: utf-8

# If you visit this kernel, and interested in this topic, please spare a moment to appreciate my effort by having a look at the research paper. You may find some very interesting and useful results: https://dx.doi.org/10.14569/IJACSA.2019.0100644

# This notebook is divided into following sections:
# 
# 1. Preprocessing of data
# 2. EDA (Exploratory data analysis)
# 3. Useful Insights from Google Play Store Info file
# 4. Useful Insights from Google Play Store Reviews file

# Necessary libraries needed:
# * Numpy is needed for mathematical operations
# * Pandas is used for data set manipulation and analysis
# * Matplotlib and seaborn is used for graphical representation

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# **Preprocessing**

# Import googleplaystore.csv file into *df_info* data frame.

# In[ ]:


df_info=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df_info.head()


# In[ ]:


df_info.sample(5)


# In[ ]:


df_info.tail()


# In[ ]:


df_info.columns


# In[ ]:


print('App:')
print(df_info['App'].describe())
print()
print('Category:')
print(df_info['Category'].describe())
print()
print('Rating:')
print(df_info['Rating'].describe())
print()
print('Reviews:')
print(df_info['Reviews'].describe())
print()
print('Size:')
print(df_info['Size'].describe())
print()
print('Installs:')
print(df_info['Installs'].describe())
print()
print('Type:')
print(df_info['Type'].describe())
print()
print('Price:')
print(df_info['Price'].describe())
print()
print('Content Rating:')
print(df_info['Content Rating'].describe())
print()
print('Genres:')
print(df_info['Genres'].describe())
print()
print('Last Updated:')
print(df_info['Last Updated'].describe())
print()
print('Current Ver:')
print(df_info['Current Ver'].describe())
print()
print('Android Ver:')
print(df_info['Android Ver'].describe())
print()


# Upon manual analysis i found some garbage records, so i dropped them

# In[ ]:


df_info.loc[df_info.App=='Tiny Scanner - PDF Scanner App']
df_info[df_info.duplicated(keep='first')]
print(len(df_info))
df_info.drop_duplicates(subset='App', inplace=True)
if(df_info.App=='Life Made WI-Fi Touchscreen Photo Frame').any():
    df_info.drop(10472,inplace=True)
if(df_info.App=='Command & Conquer: Rivals').any():
    df_info.drop(9148,inplace=True)
print(len(df_info))


# For analysis we need to pre process data, encode and normalize attributes, change their data types.
# * NaN and null records of "Ratings" attributes has been removed.
# * All Apps "Size" has been converted to KB and sign has been removed to allow numerical analysis.
# * Classes are defined for "Content Rating" attribute.

# In[ ]:


#App
df_info.App = df_info.App.apply(lambda x: str(x))
#Category
df_info.Category = df_info.Category.apply(lambda x: str(x))
#Rating
df_info.Rating = df_info.Rating.apply(lambda x: float(x))
print('NaN Ratings:')
print(len(df_info.loc[pd.isna(df_info.Rating)]))
#Reviews
df_info.Reviews = df_info.Reviews.apply(lambda x: int(x))
#Size : Remove 'M', Convert 'k'
df_info.Size = df_info.Size.apply(lambda x: str(x))
print('Apps having Varies with device as size:')
print(len(df_info.loc[df_info.Size=='Varies with device']))
df_info.Size = df_info.Size.apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df_info.Size = df_info.Size.apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df_info.Size = df_info.Size.apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df_info.Size = df_info.Size.apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df_info.Size = df_info.Size.apply(lambda x: round(float(x),2))
#Installs: Remove + and ,
df_info.Installs = df_info.Installs.apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df_info.Installs = df_info.Installs.apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df_info.Installs = df_info.Installs.apply(lambda x: int(x))
#Type
df_info.Type = df_info.Type.apply(lambda x: str(x))
#Price
df_info.Price = df_info.Price.apply(lambda x: x.replace('$', '') if '$' in str(x) else x)
df_info.Price = df_info.Price.apply(lambda x: int(round(float(x))))
#Content Rating
df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: str(x))
df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Everyone 10+', '10+') if 'Everyone 10+' in str(x) else x)
df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Teen', '13+') if 'Teen' in str(x) else x)
df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Mature 17+', '17+') if 'Mature 17+' in str(x) else x)
df_info['Content Rating'] = df_info['Content Rating'].apply(lambda x: x.replace('Adults only 18+', '18+') if 'Adults only 18+' in str(x) else x)
df_info.Genres.astype('str')
pd.to_datetime(df_info['Last Updated'])
print('Data shape:')
print(df_info.shape)
df_info.sample(5)


# Basic Info using describe()

# In[ ]:


df_info.describe(include=[np.object]).round(1).transpose()


# Import googleplaystore_user_reviews.csv file into *df_reviews* data frame.

# In[ ]:


df_reviews=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
df_reviews.head()


# In[ ]:


df_reviews.sample(5)


# In[ ]:


df_reviews.tail()


# In[ ]:


print('App:')
print(df_reviews['App'].describe())
print()
print('Translated_Review:')
print(df_reviews['Translated_Review'].describe())
print()
print('Sentiment:')
print(df_reviews['Sentiment'].describe())
print()
print('Sentiment_Polarity:')
print(df_reviews['Sentiment_Polarity'].describe())
print()
print('Sentiment_Subjectivity:')
print(df_reviews['Sentiment_Subjectivity'].describe())
print()


# In[ ]:


df_reviews.columns


# Remove *NaN Translated_Review*

# In[ ]:


print(df_reviews.shape)
print('NaN Translated_Review:')
print(len(df_reviews.loc[pd.isna(df_reviews.Translated_Review)]))


# Remove *null* records

# In[ ]:


df_reviews=df_reviews.dropna()
print(df_reviews.shape)


# In[ ]:


df_reviews.sample(5)


# Basic info using *describe()*

# In[ ]:


round(df_reviews.describe(),0)


# In[ ]:


df_reviews.describe(include=[np.object]).round(1)


# In[ ]:


import math
import scipy.stats as stats
import plotly
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plotly.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')


# In[ ]:


df=df_info['Category'].value_counts()
df


# In[ ]:


df_info['Category'].value_counts().plot(kind='barh',figsize=(4,8))
plt.title('No of apps in each category')
plt.xlabel('No of apps')
plt.ylabel('Category')
plt.show()


# In[ ]:


df.describe()


# In[ ]:


df_info['Category'].describe()


# In[ ]:


df_info['Rating'].hist()
plt.xlabel('Rating')
plt.show()


# In[ ]:


df_info['Rating'].describe()


# In[ ]:


mean=df_info.Rating.mean()
variance=df_info.Rating.var()
stdDev = math.sqrt(variance)
x = np.linspace(mean - stdDev, mean + stdDev, 100)
plt.plot(x, stats.norm.pdf(x, mean, stdDev))
plt.show()


# In[ ]:


df_info['Reviews'].hist()


# In[ ]:


df=df_info[df_info['Reviews']>=10000000]
df['Reviews'].hist()


# In[ ]:


df_info['Size'].describe()


# In[ ]:


df_info.Size.hist()


# In[ ]:


df_info.Size.value_counts(bins=[0,20,40,60,80,100])


# In[ ]:


df_info.Installs.hist()


# In[ ]:


df_info.Installs.value_counts(bins=[0,100000000,500000000,1000000000])


# In[ ]:


df_info.Type.value_counts()


# In[ ]:


df=df_info[df_info['Price']>0]
df.Price.describe()


# In[ ]:


df.Price.hist()


# In[ ]:


df_info['Content Rating'].value_counts()


# In[ ]:


df_info.Genres.value_counts()


# In[ ]:


df_info['Android Ver'].value_counts()


# **Useful Insights from Google Play Store Info file**

# *Basic correlogram shows:*
# 
# * Free apps has higher reviews and ratings
# * Paid apps with higher size has higher ratings
# * Paid Apps has very less reviews
# * Free apps with higher size also has higher reviews
# * No of installs are higher for free apps
# * Higher no of installs has higher reviews
# * Size of paid apps is much more than free apps
# * Free apps also has higher size

# In[ ]:


np.warnings.filterwarnings('ignore')
sns.set()
sns.pairplot(df_info,hue='Type')
plt.show()


# *Top categories with most apps:*
# 1. Family
# 2. Game
# 3. Tools
# 4. Business
# 5. Medical

# *Categories with least apps:*
# 5. Arts & Design
# 4. Events
# 3. Parenting
# 2. Comics
# 1. Beauty

# In[ ]:


number_of_apps_in_category = df_info['Category'].value_counts().sort_values(ascending=True)

data = [go.Pie(
        labels = number_of_apps_in_category.index,
        values = number_of_apps_in_category.values,
        hoverinfo = 'label+value'    
)]
plotly.offline.iplot(data, filename='active_category')


# * Category vise mean is *292*.
# * So catyegories with no of apps greater than 292 are considered.

# In[ ]:


groups = df_info.groupby('Category').filter(lambda x: len(x) > 292).reset_index()
array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))


# In[ ]:


df=df_info.groupby(['Category']).filter(lambda x: len(x) > 292).reset_index()
df=df.groupby(['Category']).mean()
df


# In[ ]:


groups = df_info.groupby('Category').filter(lambda x: len(x) >= 292).reset_index()
print('Average rating = ', np.nanmean(list(groups.Rating)))
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 720, len(set(groups.Category)))]

layout = {'title' : 'App ratings across major categories',
        'xaxis': {'tickangle':-40},
        'yaxis': {'title': 'Rating'},
          'plot_bgcolor': 'rgb(250,250,250)',
          'shapes': [{
              'type' :'line',
              'x0': -.5,
              'y0': np.nanmean(list(groups.Rating)),
              'x1': 19,
              'y1': np.nanmean(list(groups.Rating)),
              'line': { 'dash': 'dashdot'}
          }]
          }

data = [{
    'y': df_info.loc[df_info.Category==category]['Rating'], 
    'type':'violin',
    'name' : category,
    'showlegend':False,
    #'marker': {'color': 'Set2'},
    } for i,category in enumerate(list(set(groups.Category)))]

plotly.offline.iplot({'data': data, 'layout': layout})


# In[ ]:


df['Rating'].sort_values(ascending=True).plot(kind='barh',figsize=(4,4))
plt.title('Average Rating in each category')
plt.xlabel('Average Rating')
plt.ylabel('Category')
plt.show()


# In[ ]:


df['Price'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))
plt.title('Average Price in each category')
plt.xlabel('Average Price')
plt.ylabel('Category')
plt.show()


# In[ ]:


df['Size'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))
plt.title('Average Size in each category')
plt.xlabel('Average Size')
plt.ylabel('Category')
plt.show()


# In[ ]:


df['Installs'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))
plt.title('Average Installs in each category')
plt.xlabel('Average Installs')
plt.ylabel('Category')
plt.show()


# In[ ]:


df['Reviews'].sort_values(ascending=True).plot(kind='barh',figsize=(5,5))
plt.title('Average Reviews in each category')
plt.xlabel('Average Reviews')
plt.ylabel('Category')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,8))
s=sns.scatterplot(x="Rating", y="Category", hue="Type", data=df_info, ax=ax);
plt.title('Rating in each category')
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(4,8))
s=sns.scatterplot(x="Installs", y="Category", hue="Type", data=df_info, ax=ax);
plt.title('No of Installs in each category')
plt.xlabel('No of Installs')
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
s=sns.scatterplot(x="Price", y="Rating", hue="Type", data=df_info, ax=ax)
plt.title('Price vs Rating')
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
s=sns.scatterplot(x="Size", y="Rating", hue="Type", data=df_info);
plt.title('Size vs Rating')
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,3))
s=sns.scatterplot(x="Rating", y="Reviews", hue="Type", data=df_info, ax=ax);
plt.title('Reviews vs Rating')
plt.draw()
plt.show()


# In[ ]:


df_info_rating=df_info[df_info.Rating<=2.5]
number_of_apps_in_category = df_info_rating['Category'].value_counts().sort_values(ascending=True)

data = [go.Pie(
        labels = number_of_apps_in_category.index,
        values = number_of_apps_in_category.values,
        hoverinfo = 'label+value'    
)]

plotly.offline.iplot(data, filename='active_category')


# In[ ]:


fig, ax = plt.subplots(figsize=(5,8))
s=sns.scatterplot(x="Reviews", y="Category", hue="Type", data=df_info, ax=ax);
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,8))
s=sns.scatterplot(x="Installs", y="Category", hue="Type", data=df_info, ax=ax);
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,3))
s=sns.scatterplot(x="Price", y="Installs", hue="Type", data=df_info, ax=ax);
plt.title('Price vs Installs')
plt.draw()
plt.show()


# In[ ]:


df_info['Size']=df_info['Size'].astype(float)
fig, ax = plt.subplots(figsize=(5,3))
s=sns.scatterplot(x="Size", y="Installs", hue="Type", data=df_info, ax=ax);
plt.title('Size vs Installs')
plt.draw()
plt.show()


# In[ ]:


df_info['Size']=df_info['Size'].astype(float)
fig, ax = plt.subplots(figsize=(5,3))
s=sns.scatterplot(x="Rating", y="Installs", hue="Type", data=df_info, ax=ax);
plt.title('Installs vs Ratings')
plt.draw()
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(5,3))
s=sns.scatterplot(x="Reviews", y="Installs", hue="Type", data=df_info, ax=ax);
plt.title('Reviews vs Installs')
plt.draw()
plt.show()


# **Useful Insights from Google Play Store Reviews file**

# In[ ]:


df_reviews['Sentiment'].value_counts()


# In[ ]:


df_reviews['Sentiment'].value_counts().plot(kind='bar',figsize=(3,3))
plt.title('Sentiment Count')
plt.xlabel('Sentiment')
plt.ylabel('Count')


# In[ ]:


df_reviews['Sentiment_Polarity'].hist()
plt.title('Sentiment_Polarity Count')
plt.xlabel('Sentiment_Polarity')
plt.ylabel('Count')
plt.show()


# In[ ]:


df_reviews['Sentiment_Subjectivity'].hist()
plt.title('Sentiment_Subjectivity Count')
plt.xlabel('Sentiment_Subjectivity')
plt.ylabel('Count')
plt.show()
plt.show()


# In[ ]:


text = " ".join(review for review in df_reviews.Translated_Review)
print ("There are {} words in the combination of all review.".format(len(text)))


# *Top 5 Words in Reviews are:*
# 1. game
# 1. good
# 1. app
# 1. time
# 1. greate

# In[ ]:


wordcloud = WordCloud(max_words=50, background_color="white").generate(text)
plt.figure(figsize=(5,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


df=df_reviews[df_reviews['Sentiment']=='Positive']
textP = " ".join(review for review in df.Translated_Review)
print ("There are {} words in the combination of all review.".format(len(textP)))


# In[ ]:


wordcloud = WordCloud(max_words=50, background_color="white").generate(textP)
plt.figure(figsize=[5,5])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


df=df_reviews[df_reviews['Sentiment']=='Neutral']
textU = " ".join(review for review in df.Translated_Review)
print ("There are {} words in the combination of all review.".format(len(textU)))


# In[ ]:


wordcloud = WordCloud(max_words=50, background_color="white").generate(textU)
plt.figure(figsize=[5,5])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


df=df_reviews[df_reviews['Sentiment']=='Negative']
textN = " ".join(review for review in df.Translated_Review)
print ("There are {} words in the combination of all review.".format(len(textN)))


# In[ ]:


wordcloud = WordCloud(max_words=50, background_color="white").generate(textN)
plt.figure(figsize=[5,5])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
s=sns.scatterplot(x="Sentiment_Subjectivity", y="Sentiment_Polarity", hue="Sentiment", data=df_reviews, ax=ax);
plt.draw()
s.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.ticklabel_format(style='plain', axis='y')
plt.show()

