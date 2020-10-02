#!/usr/bin/env python
# coding: utf-8

# # My first kernel
# This my first attempt at a Kernel, I started with trying some of the techniques in on of the Titanic Tutorial and then tried to look a bit deeper at some of the parts of the data that looked interesting before borrowing from the community and looking how other people had analysed and visualed the data.
# 
# ## Acquire data
# Getting the open dataset and importing the essential libaries to look at the data

# In[ ]:


import os
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.figure_factory as ff

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print(os.listdir("../input"))


# In[ ]:


# A nice way to tidy the notebook - borrowed from - Lavanya Gupta https://www.kaggle.com/lava18/all-that-you-need-to-know-about-the-android-market
# But quite like the default Kaggle
"""from IPython.display import HTML
HTML('''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
"""


# ## Describing the data
# ### What does the data look like

# In[ ]:


apps_df = pd.read_csv('../input/googleplaystore.csv')
print('App table columns and data \n')
print(apps_df.columns.values)
apps_df.head()


# In[ ]:


reviews_df = pd.read_csv('../input/googleplaystore_user_reviews.csv')
print('Review table columns and data \n')
print(reviews_df.columns.values)
reviews_df.head()


# In[ ]:


print('Show then describe the App and Review tables\n')
apps_df.info()
print('_'*40)
reviews_df.info()


# In[ ]:


print('Describe the numerical values of the App table')
apps_df.describe()


# In[ ]:


print('Describe the non-numberical data of the App table')
apps_df.describe(include=['O'])


# In[ ]:


print('Describe the numerical values of the Review table')
reviews_df.describe()
# Polarity is between -1 and 1 the average is 0.182
# Subjectivity is between 0 and 1 with the mean at 0.492


# In[ ]:


print('Describe the non-numberical data of the review table')
reviews_df.describe(include=['O'])


# ### In apps_df
# #### Which features are categorical?
# * Category
# * Type
# * Content Rating
# * Genres
# * Current Ver
# * Android Ver
# 
# #### Which features are numerical?
# * Rating
# 
# #### Which values are dates or mixed data?
# * Size (we can convert this to a number)
# * Reviews (we can convert this to a number)
# * Installs (we can convert this to a number)
# * Price (we can convert this to a number)
# * Last Updated (a date value)
# 
# ### In reviews_df
# #### Which features are categorical?
# * Sentiment
# 
# #### Which features are numerical?
# * Sentiment_Polarity
# * Sentiment_Subjectivity
# 
# #### Which values are dates or mixed data?
# * Translated_Review
# 
# 

# ### Which features contain blank, null or empty values?
# * apps_df -> Rating has about 10% blanks
# * apps_df -> Type has a single value missing
# * apps_df -> Content Rating has a single value missing
# * apps_df -> Current Ver has less than 10 values missing
# * apps_df -> Android Ver has less than 10 values missing
# * reviews_df ->  Translated_Review about 50% have values 37427 / 64295
# * reviews_df -> Sentiment about 50% have values 37432 / 64295
# * reviews_df -> Sentiment_Polarity about 50% have values 37432 / 64295
# * reviews_df -> Sentiment_Subjectivity about 50% have values 37432 / 64295

# ### Distribution of the numerical values
# #### Apps -> Rating
# * Rating is the only numeric in apps_df 
# * the Average is 4.19 
# * The distribution seems to be between 1 and 5
# * there appears to be incorrect values at there's a max of 19 
# 
# #### Review 
# * Polarity is between -1 and 1 the average is 0.182
# * Subjectivity is between 0 and 1 with the mean at 0.492

# ### What is the distribution of categorical features?
# #### Apps 
# * App - there are around about 500 apps with the same name / dupicate records - there are 9 ROBLOX apps
# * Category - there are 34 categories - Family is the most popular with almost 2000 family apps
# * Reviews - around 60% have reviews
# * Size - there are 462 representations of size - this could be corrected to a numeric value - there are over 1500 that have a different size for different devices so the size is not recorded
# * Installs there are 22 categories / levels of installs - most are 1,000,000+
# * Most Apps fall in the free category & there are only 3 types of app
# * There are 6 types of category rating - Most are for "everyone"
# * There are 120 Genres - the most popular is tools
# * Most apps were updated on the 3rd of August 2018... an ever moving number I am sure
# * Like with size the version often varies with device
# * Most apps are built for 4.1 and up although there are 33 other noted versions
# #### Reviews
# * 1074 Apps have reviews
# * The most common comment is "Good"
# * There are three measure of sentiment, most are positive which is just over half of them

# In[ ]:


print('Cleaning data before we go ahead and drop rows with empty cells')
# - Installs : Remove + and , from - Lavanya Gupta
apps_df = apps_df[apps_df['Installs'] != 'Free'] # Data in the wrong column
apps_df = apps_df[apps_df['Installs'] != 'Paid'] # Data in the wrong column

apps_df['Installs'] = apps_df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
apps_df['Installs'] = apps_df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
apps_df['Installs'] = apps_df['Installs'].apply(lambda x: int(x))
#print(type(apps_df['Installs'].values))


# In[ ]:


# - Size : Remove 'M', Replace 'k' and divide by 10^-3
#df['Size'] = df['Size'].fillna(0)

apps_df['Size'] = apps_df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

apps_df['Size'] = apps_df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
apps_df['Size'] = apps_df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
apps_df['Size'] = apps_df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)


apps_df['Size'] = apps_df['Size'].apply(lambda x: float(x))
apps_df['Installs'] = apps_df['Installs'].apply(lambda x: float(x))

apps_df['Price'] = apps_df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
apps_df['Price'] = apps_df['Price'].apply(lambda x: float(x))

apps_df['Reviews'] = apps_df['Reviews'].apply(lambda x: int(x))


# In[ ]:


print('App Table')
print('* Renaming the column with a space')
print('* Identifying the rows with empty cells (thanks lastnight)')
apps_df=apps_df.rename(columns = {'Content Rating':'Content_Rating'})
#missing data (borrowed from lastnight https://www.kaggle.com/tanetboss/how-to-get-high-rating-on-play-store)
total = apps_df.isnull().sum().sort_values(ascending=False)
percent = (apps_df.isnull().sum()/apps_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)


# In[ ]:


print('Removing the rows with empty cells')
apps_df.dropna(how ='any', inplace = True)


# In[ ]:


print('Review Table')
print('* Renaming the column with a space')
print('* Identifying the rows with empty cells (thanks lastnight)')
#missing data
total = reviews_df.isnull().sum().sort_values(ascending=False)
percent = (reviews_df.isnull().sum()/reviews_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)


# In[ ]:


reviews_df.dropna(how ='any', inplace = True)
print('Describe the cleaned App and Review tables\n')
apps_df.info()
print('_'*40)
reviews_df.info()


# ### Looking at the data with a pivot table
# * Very high rating alone might not tell you the whole story - the one app with a rating of 5 was downloaded once
# * Dating apps do not recieve a high rating on average
# * Tools maybe the most common category, hoever they don't rate well
# * Events, Education, arts/design and books/reference get good ratings
# * Apps with less than 50 downloads have the highest rating
# * Apps with a few thousand have a lower average rating... suggesting they may have hit their peak
# * Paid apps generally have a better rating
# * Adults only content rating has the highest average rating and unrated the lowest
# * Genres - Comic/ creative, board/pretend play have the highest average rating while parenting apps get the lowest 
# 

# In[ ]:


apps_df[['Category', 'Rating']].groupby(['Category'], as_index=False).mean().sort_values(by='Rating', ascending=False)


# In[ ]:


apps_df[['Installs', 'Rating']].groupby(['Installs'], as_index=False).mean().sort_values(by='Rating', ascending=False)


# In[ ]:


apps_df[['Type', 'Rating']].groupby(['Type'], as_index=False).mean().sort_values(by='Rating', ascending=False)


# In[ ]:


apps_df[['Content_Rating', 'Rating']].groupby(['Content_Rating'], as_index=False).mean().sort_values(by='Rating', ascending=False)


# In[ ]:


apps_df[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values(by='Rating', ascending=False)


# ### Analysing the review data with the pivot
# To do this we need to take the apps data that has the categories and apply a left join with the review data on the app names - this will allow us to look at the sentiment analysis around the views - potentially still using the rating too.
# * Comics have the most positive reviews that correlates with the high average rating, the reviews were the most subjective
# * Parents apps also have positive reviws even though they were at the bottom of the average rating
# * At the bottom is Social and Games with neutral comments
# * The business reviews were the least subjective
# * In general it appears that the more installs there are the less positive the reviews, the subjectivity tends to decrease too.
# * Paid apps have better reviews on average
# * Adults only content receives more positive reviews, teen content recieve less positive reviews
# * Comics and creative has the best reviews, Role play is the only one that recieves on average negative sentiment
# 
# My guesses at this stage.
# There's a weak correlation between rating and sentiment polatity that gets weeker with the number of installs

# 

# In[ ]:


apps_reviews_df = pd.merge(apps_df, reviews_df, on='App')


# In[ ]:


apps_reviews_df.head()


# In[ ]:


apps_reviews_df[['Category', 'Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['Category'], as_index=False).mean().sort_values(by='Sentiment_Polarity', ascending=False)


# In[ ]:


apps_reviews_df[['Category','Sentiment_Subjectivity', 'Sentiment_Polarity']].groupby(['Category'], as_index=False).mean().sort_values(by='Sentiment_Subjectivity', ascending=False)


# In[ ]:


apps_reviews_df[['Installs', 'Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['Installs'], as_index=False).mean().sort_values(by='Sentiment_Polarity', ascending=False)


# In[ ]:


apps_reviews_df[['Type', 'Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['Type'], as_index=False).mean().sort_values(by='Sentiment_Polarity', ascending=False)


# In[ ]:


apps_reviews_df[['Content_Rating', 'Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['Content_Rating'], as_index=False).mean().sort_values(by='Sentiment_Polarity', ascending=False)


# In[ ]:


apps_reviews_df[['Genres', 'Sentiment_Polarity','Sentiment_Subjectivity']].groupby(['Genres'], as_index=False).mean().sort_values(by='Sentiment_Polarity', ascending=False)


# ## Exploratory Data Analysis
# EDA - a new acronym for me - again thanks to the work by Lavanya Gupta https://www.kaggle.com/lava18/all-that-you-need-to-know-about-the-android-market
# 
# * Rating for free and paid are distrubuted similarly and skewed to high ratings
# * Most applications are small
# * More free apps are installed than paid
# * Most paid apps are cheap, however there are a couple of very expensive apps ~   $350 & $400
# * Ratings are generally better for paid apps that are small
# * There are less installs and less reviews for paid apps
# * The more installs there are... the more reviews there are
# 
# 

# 

# In[ ]:


x = apps_df['Rating'].dropna()
y = apps_df['Size'].dropna()
z = apps_df['Installs'][apps_df.Installs!=0].dropna()
p = apps_df['Reviews'][apps_df.Reviews!=0].dropna()
t = apps_df['Type'].dropna()
price = apps_df['Price']

plot = sns.pairplot(pd.DataFrame(list(zip(x, y, np.log(z), np.log10(p), t, price)), 
                        columns=['Rating','Size', 'Installs (log)', 'Reviews (log)', 'Type', 'Price']), hue='Type', palette="Set2")


# ### Violin plot
# Another lovely plot by Lavanya Gupta
# * Health and Fitness and Games perform best
# * Dating catergory does not perfomr well
# * Lifestyle, Family and Finance have some very low ratings 
# 

# In[ ]:


groups = apps_df.groupby('Category').filter(lambda x: len(x) >= 170).reset_index()
#print(type(groups.item.['BUSINESS']))
print('Average rating = ', np.nanmean(list(groups.Rating)))
#print(len(groups.loc[df.Category == 'DATING']))
c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 720, len(set(groups.Category)))]


#df_sorted = df.groupby('Category').agg({'Rating':'median'}).reset_index().sort_values(by='Rating', ascending=False)
#print(df_sorted)

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
    'y': apps_df.loc[apps_df.Category==category]['Rating'], 
    'type':'violin',
    'name' : category,
    'showlegend':False,
    #'marker': {'color': 'Set2'},
    } for i,category in enumerate(list(set(groups.Category)))]

plotly.offline.iplot({'data': data, 'layout': layout})


# ### Sentiment analysis of the categories

# In[ ]:


# by Lavanya Gupta

grouped_sentiment_category_count = apps_reviews_df.groupby(['Category', 'Sentiment']).agg({'App': 'count'}).reset_index()
grouped_sentiment_category_sum = apps_reviews_df.groupby(['Category']).agg({'Sentiment': 'count'}).reset_index()

new_df = pd.merge(grouped_sentiment_category_count, grouped_sentiment_category_sum, on=["Category"])
#print(new_df)
new_df['Sentiment_Normalized'] = new_df.App/new_df.Sentiment_y
new_df = new_df.groupby('Category').filter(lambda x: len(x) ==3)
# new_df = new_df[new_df.Category.isin(['HEALTH_AND_FITNESS', 'GAME', 'FAMILY', 'EDUCATION', 'COMMUNICATION', 
#                                      'ENTERTAINMENT', 'TOOLS', 'SOCIAL', 'TRAVEL_AND_LOCAL'])]
new_df

trace1 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[::3][6:-5],
    name='Negative',
    marker=dict(color = 'rgb(209,49,20)')
)

trace2 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[1::3][6:-5],
    name='Neutral',
    marker=dict(color = 'rgb(49,130,189)')
)

trace3 = go.Bar(
    x=list(new_df.Category[::3])[6:-5],
    y= new_df.Sentiment_Normalized[2::3][6:-5],
    name='Positive',
    marker=dict(color = 'rgb(49,189,120)')
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    title = 'Sentiment analysis',
    barmode='stack',
    xaxis = {'tickangle': -45},
    yaxis = {'title': 'Fraction of reviews'}
)

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot({'data': data, 'layout': layout})


# In[ ]:


# by Lavanya Gupta

from wordcloud import WordCloud
wc = WordCloud(background_color="white", max_words=200, colormap="Set2")
# generate word cloud

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop = stop + ['app', 'APP' ,'ap', 'App', 'apps', 'application', 'browser', 'website', 'websites', 'chrome', 'click', 'web', 'ip', 'address',
            'files', 'android', 'browse', 'service', 'use', 'one', 'download', 'email', 'Launcher']

#merged_df = merged_df.dropna(subset=['Translated_Review'])
apps_reviews_df['Translated_Review'] = apps_reviews_df['Translated_Review'].apply(lambda x: " ".join(x for x in str(x).split(' ') if x not in stop))
#print(any(merged_df.Translated_Review.isna()))
apps_reviews_df.Translated_Review = apps_reviews_df.Translated_Review.apply(lambda x: x if 'app' not in x.split(' ') else np.nan)
apps_reviews_df.dropna(subset=['Translated_Review'], inplace=True)


free = apps_reviews_df.loc[apps_reviews_df.Type=='Free']['Translated_Review'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(free)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### WORDCLOUD
# Showing some of the words that frequently appear in the reviews (by Lavanya Gupta)
