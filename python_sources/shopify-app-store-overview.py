#!/usr/bin/env python
# coding: utf-8

# # Shopify App Store overview

# Shopify is a complete commerce platform that lets clients start, grow, and manage a business. The company reported that it had more than **600,000 merchants** using its platform as of August 2017, with total gross merchandise volume exceeding **$82 billion**.
# 
# That is one of the main causes why more and more developers choose the Shopify platform and the Shopify apps marketplace as the next app store for their products.
# 
# This notebook is an attempt to describe the Shopify app store. What kind of apps are the most popular? What shop owners are looking for? What apps provide and what they fail to deliver? 
# All these questions are explained in the following blocks.

# ## Dataset import

# In[ ]:


import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

# source
apps = pd.read_csv('../input/apps.csv')
categories = pd.read_csv('../input/categories.csv')
reviews = pd.read_csv('../input/reviews.csv')


# In[ ]:


apps.head()


# In[ ]:


categories.head()


# In[ ]:


reviews.head()


# ## Data preprocessing

# The categories from the source dataset do not have the ids, the `category_id` column might be usefull for joining/grouping tables.

# In[ ]:


categories['category_id'] = categories['category'].factorize()[0]


# Prepare Dataframes with joined data

# In[ ]:


apps_and_reviews = pd.merge(apps, reviews, how='right', left_on='url', right_on='app_url')
apps_with_categories = pd.merge(apps, categories, how='right', left_on='url', right_on='app_url')
apps_and_reviews_with_categories = pd.merge(apps_and_reviews, categories, how='right', left_on='app_url', right_on='app_url')


# #### Validation
# 
# Small ammount of apps (**22**) have reviews which were published and included in the reviews count but are not available on the marketplace. This is small part of overal number (< 0.8%) so it doesn't have big impact on category statistics.

# In[ ]:


reviews_count_check = pd.merge(
    apps[['url', 'reviews_count']], 
    reviews.groupby(['app_url']).size().reset_index(name='reviews_available_count'), 
    how='left', left_on='url', right_on='app_url')

reviews_count_check[['reviews_available_count']] = reviews_count_check[['reviews_available_count']].fillna(value=0)
reviews_count_check['diff'] = reviews_count_check['reviews_available_count'] - reviews_count_check['reviews_count']
reviews_count_check.loc[reviews_count_check['diff'] != 0].drop_duplicates(subset=['url'])


# ## Shopify apps marketplace breakdown

# ### Category connections

# Each app belongs to at least one category. The chord graph shows the intersection of the existing 12 categories.

# In[ ]:


import holoviews as hv

hv.extension('bokeh')
get_ipython().run_line_magic('output', 'size=250')

category_id_df = apps_with_categories[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)

results = pd.DataFrame()

for cat in categories['category'].unique():
    app_urls = categories[categories['category'] == cat]['app_url'].values
    category_connections = categories[(categories['app_url'].isin(app_urls)) & (categories['category'] != cat)]        .groupby(['category'])         .size()         .reset_index(name='connections')
    category_connections['from'] = cat
    category_connections['to'] = category_connections['category']
    
    category_connections['source']=category_connections['from'].map(category_to_id)
    category_connections['target']=category_connections['to'].map(category_to_id)
    category_connections['value']=category_connections['connections']
    
    results = pd.concat([results, category_connections[['source', 'target', 'value']]])

source = results[['source', 'target']].min(axis=1)
target = results[['source', 'target']].max(axis=1)
results['source'] = source
results['target'] = target
results = results.drop_duplicates(subset=['source', 'target'])

nodes_data = categories[['category']].drop_duplicates()
nodes_data['index'] = categories['category'].map(category_to_id)

nodes = hv.Dataset(nodes_data, 'index')

get_ipython().run_line_magic('opts', "Chord [label_index='category' color_index='index' edge_color_index='source']")

hv.Chord((results, nodes))


# This chart shows that **'Store design'**, **'Sales and conversion optimization'** and **'Marketing'** categories have intersection. **'Reporting'** category has lower number of connections with that group but has more connections that other groups.

# Which category has the highest number of apps ?

# In[ ]:


apps_per_category = apps_with_categories    .groupby(['category'])     .size()     .reset_index(name='apps_count')     .sort_values('apps_count', ascending=False)

plotly.offline.iplot({
    'data': [go.Pie(labels=apps_per_category['category'], values=apps_per_category['apps_count'])],
    'layout': go.Layout(title='Apps per category')
})


# - 'Store design' category (29%) includes more than 1.5 times more apps than closest 'Sales and conversion optimization' category (17.5%)
# - The 'Sales and conversion optimization' and 'Marketing' categories are close

# Which category has the highest number of reviews ?

# In[ ]:


apps_and_reviews_grouped_by_category = apps_and_reviews_with_categories    .groupby(['category'])     .size()     .reset_index(name='reviews')     .sort_values('reviews', ascending=False)

plotly.offline.iplot({
    'data': [go.Pie(labels=apps_and_reviews_grouped_by_category['category'], values=apps_and_reviews_grouped_by_category['reviews'])],
    'layout': go.Layout(title='Reviews per category')
})


# - Nothing surprises here, the same top 3 categories
# - The distribution is proportional to apps per category numbers

# In[ ]:


apps_and_reviews_grouped_by_app_title = apps_and_reviews     .groupby(['title'])     .size()     .reset_index(name='reviews')     .sort_values('reviews', ascending=False)

limit = 15
plotly.offline.iplot({
    'data': [go.Bar(
        name='Reviews',
        x=apps_and_reviews_grouped_by_app_title.head(limit)['title'],
        y=apps_and_reviews_grouped_by_app_title.head(limit)['reviews']
    )],
    'layout': go.Layout(title='Apps ordered by number of reviews', margin=go.layout.Margin(b=200))
})


# ### Feedback

# Another interesting thing to investigate is the connection between category and rating. How app rating is distributed between categories, which category has the lowest/highest average rating.

# In[ ]:


reviews_in_category = (apps_and_reviews_with_categories.groupby(['category', 'title'], as_index=False)['rating_x'].mean())    .dropna(subset=['rating_x'])    .groupby(['category'], as_index=False)    .mean()    .sort_values('rating_x', ascending=True)
reviews_in_category[['mean_rating']] = reviews_in_category[['rating_x']]
reviews_in_category = reviews_in_category.drop(columns=['rating_x'])

plotly.offline.iplot({
    'data': [go.Bar(
        name='Rating',
        x=reviews_in_category['mean_rating'],
        y=reviews_in_category['category'],
        orientation = 'h'
    )],
    'layout': go.Layout(title='Average rating per category', margin=go.layout.Margin(l=250))
})


# It looks like customers are less satisfied with tools from 'Places to sell' category.

# In[ ]:


each_rating_count_in_category = pd.DataFrame({
    'category': apps_and_reviews_with_categories['category'].unique(),
    'r_count': [
        apps_and_reviews_with_categories.loc[apps_and_reviews_with_categories['category'] == cat].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_1_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 1.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_2_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 2.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_3_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 3.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_4_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 4.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_5_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 5.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
})
each_rating_count_in_category = each_rating_count_in_category.loc[each_rating_count_in_category.category.notna()]

groups = ['r_1_count', 'r_2_count', 'r_3_count', 'r_4_count', 'r_5_count']
traces = []
x_axis_lables = ['Share of 1 star ratings', 'Share of 2 star ratings', 'Share of 3 star ratings', 'Share of 4 star ratings', 'Share of 5 star ratings']

for idx, row in each_rating_count_in_category.iterrows():
    traces.append(go.Bar(
        x=x_axis_lables,
        y=list((row[groups] / row['r_count'])),
        name=row['category']
    ))

layout = go.Layout(
    title='Share of reviews per rating per category',
    yaxis = dict(
        tickformat='.2%'
    )
)

plotly.offline.iplot(go.Figure(data=traces, layout=layout))

each_rating_count_in_category


# The **'Places to sell'** category has the largest share of 1 star reviews and the smallest share of 5 star reviews. It fits with the previous assumption that users are least satisfied with apps from **'Places to sell'** category.

# ## Supply and demand

# On the next chart you can see the word combinations that appears on the apps description most frequently. Developers write down those qualities so they can be used as a proxy to 'functionality/features the app provides'.

# In[ ]:


from plotly import tools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

apps_with_categories['normalized_description'] = (apps_with_categories['description'].map(str) + apps_with_categories['pricing'])

category_id_df = apps_with_categories[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', ngram_range=(2, 5), stop_words='english')
features = tfidf.fit_transform(apps_with_categories['normalized_description']).toarray()
labels = apps_with_categories['category_id']

N = 15
fig = tools.make_subplots(rows=6, cols=2, 
                          shared_yaxes=False, shared_xaxes=False,
                          horizontal_spacing=0.5, print_grid=False, 
                          subplot_titles=["'{0}' term scores".format(entry[0]) for entry in category_to_id.items()])

for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])    
    feature_names = np.array(tfidf.get_feature_names())[indices]
    scores = features_chi2[0][indices]

    category_significant_terms = pd.DataFrame({'term': feature_names, 'score': scores})        .sort_values('score', ascending=True)        .tail(N)

    row = int(category_id / 2) + 1
    col = 1 if (category_id % 2 == 0) else 2
    bar_chart = go.Bar(
        name=category, 
        x=category_significant_terms['score'], 
        y=category_significant_terms['term'], 
        orientation='h'
    )
    fig.append_trace(bar_chart, row, col)

    
fig['layout'].update(title='Category terms', 
                     height=1024, width=1024, 
                     margin=go.layout.Margin(l=225, r=225), showlegend=False)
plotly.offline.iplot(fig)


# Let's try to understand what people in reviews are complaining about. Poor quality app might be an opportunity for the new app that solves the same problem better.

# In[ ]:


reviews_with_low_ratings = pd.DataFrame()
for category in apps_and_reviews_with_categories['category'].dropna().unique():
    reviews_in_category = apps_and_reviews_with_categories[apps_and_reviews_with_categories['category'] == category]
    reviews_with_rating_lower_than_app_rating = reviews_in_category[(reviews_in_category['rating_y'] < reviews_in_category['rating_x']) &                                                         (reviews_in_category['reviews_count'] > 0)]
    reviews_with_low_ratings = pd.concat([reviews_with_low_ratings, reviews_with_rating_lower_than_app_rating])
reviews_with_low_ratings.head()


# On the next chart you can see the word combinations that appears on the reviews with low rating. Users complain about those things so they can be used as a proxy to 'what app fail to deliver'.

# In[ ]:


from plotly import tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

reviews_with_low_ratings = reviews_with_low_ratings.dropna(subset=['body'])
reviews_with_low_ratings['negative_prob'] = reviews_with_low_ratings['body'].apply(lambda body: analyser.polarity_scores(body)["neg"])
negative_reviews_with_low_rating = reviews_with_low_ratings[reviews_with_low_ratings['negative_prob'] >= 0.5]

category_id_df = negative_reviews_with_low_rating[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, 
                        min_df=3, 
                        norm='l2',
                        ngram_range=(2, 5), 
                        stop_words='english')

features = tfidf.fit_transform(negative_reviews_with_low_rating['body']).toarray()
labels = negative_reviews_with_low_rating['category_id']

N = 15
fig = tools.make_subplots(rows=6, cols=2, 
                          shared_yaxes=False, shared_xaxes=False,
                          horizontal_spacing=0.5, print_grid=False, 
                          subplot_titles=["'{0}' term scores".format(entry[0]) for entry in category_to_id.items()])

charts = []

for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])    
    feature_names = np.array(tfidf.get_feature_names())[indices]
    scores = features_chi2[0][indices]

    category_significant_terms = pd.DataFrame({'term': feature_names, 'score': scores})        .sort_values('score', ascending=True)        .tail(N)

    if category_significant_terms.shape[0] == 0:
        next

    bar_chart = go.Bar(
        name=category, 
        x=category_significant_terms['score'], 
        y=category_significant_terms['term'], 
        orientation='h'
    )
    charts.append(bar_chart)

for number, chart in enumerate(charts):
    row = int(number / 2) + 1
    col = 1 if (number % 2 == 0) else 2
    fig.append_trace(chart, row, col)
    
fig['layout'].update(title='Terms of low rating reviews', 
                     height=1024, width=1024, 
                     margin=go.layout.Margin(l=225, r=225), showlegend=False)
plotly.offline.iplot(fig)


# These charts show that terrible customer support, constant error screens and 'app simply does not work' are the main complaints across all categories.

# ## Developers

# This part describes the marketplace from developers & reviewers point of view.

# In[ ]:


limit = 15

apps_per_developer = apps    .groupby(['developer'])     .size()     .reset_index(name='apps_count')     .sort_values('apps_count', ascending=False)

plotly.offline.iplot({
    'data': [go.Bar(
            name='Summary rating',
            x=apps_per_developer.head(limit)['developer'],
            y=apps_per_developer.head(limit)['apps_count']
    )],
    'layout': go.Layout(title='Developers ordered by the number of apps', margin=go.layout.Margin(b=100))
})


# The developer with the biggest number of published apps is the **"Webkup Software Pvt Ltd"**.
# 
# But the number of apps is not the same as the quality of apps. The top 15 developers (by sum of ratings of all their apps) can be seen on the next chart .

# In[ ]:


limit = 15

apps_and_reviews_grouped_by_developer = apps_and_reviews     .groupby(['developer'])     .agg({'rating_y': ['size', 'mean', 'sum']})     .reset_index()     .sort_values(('rating_y', 'sum'), ascending=False)


plotly.offline.iplot({
    'data': [go.Bar(
            name='Summary rating',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'sum')]
    ),go.Bar(
            name='Reviews',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'size')]
    ),go.Scatter(
            name='Rating',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'mean')],
            yaxis='y2'
    )],
    'layout': go.Layout(
        title='Developers with the highest summary rating',
        legend=dict(x=1.25, y=1),
        barmode='group',
        yaxis2=dict(
            overlaying='y',
            anchor='x',
            side='right'
        )
    )
})


# The 2 patterns can be seen between the developers from the cart:
# - Developer has multiple highly rated apps (most of the developers follow this pattern)
# - Developer has only one but incredibly successfull app
# 
# 
# Only **3** ([BOLD](https://apps.shopify.com/partners/bold), [SpurIT](https://apps.shopify.com/partners/spurit), [Booster Apps]()) devs from first chart managed to get into **'Top 15'**.

# ### Category leaderboards
# 
# The next table displays the category leaders for each of 12 categories.

# In[ ]:


apps_with_rating_scores = apps_with_categories[(apps_with_categories['reviews_count'] > 0)].copy()
apps_with_rating_scores['rating_mult_by_reviews'] = apps_with_rating_scores['rating'] * apps_with_rating_scores['reviews_count']

apps_with_highest_score = apps_with_rating_scores    .loc[apps_with_rating_scores.groupby(['category'])['rating_mult_by_reviews'].idxmax()]

apps_with_highest_score[['category', 'app_url', 'title', 'rating', 'reviews_count']]


# ---

# ## Conclusion
# 
# ### Observations
# 
# - It looks like the 'bad support' is the most frequent problem that users deal with. Maybe it can be handled on the markeplace level by providing links to forum. Specific threads for apps on [Discussion Forums](https://ecommerce.shopify.com/forums) might improve experience of those users.
# - The apps from 'Places to sell' category are the most downvoted. It also looks like it may happen because of high expectations of the users. After analyzing the reviews it can be seen that many users don't even know what they want from apps.
# - The 'Store design', 'Sales and conversion optimization' and 'Marketing' are categories with the biggest amount of apps
