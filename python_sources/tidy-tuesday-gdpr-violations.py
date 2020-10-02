#!/usr/bin/env python
# coding: utf-8

# # Copying Dave Robinson to understand Python
# 
# In language learning, there is this concept of triangulation: use your second language to learn your third. For example, use your French knoweldge to learn German. Thus, one can refresh earlier concepts **AND** learn a new language simultaneously. 
# 
# This notebook is my attempt at triangulation in programming. Having read and wrote some R, I attempt a TidyTuesday submission in Python following step by step what Dave Robinson did in his [screencast](https://www.youtube.com/watch?v=vT-DElIaKtE) with R. That is, I attempt to replicate almost every step he did in his analysis using Python. Thus, I am able to keep up with the growing tidyverse in R and at the same time I get to learn how to do advanced data analysis in Python. 
# 
# ## What is TidyTuesday?
# 
# > A weekly [data project](https://github.com/rfordatascience/tidytuesday) aimed at the R ecosystem. As this project was borne out of the R4DS Online Learning Community and the R for Data Science textbook, an emphasis was placed on understanding how to summarize and arrange data to make meaningful charts with ggplot2, tidyr, dplyr, and other tools in the tidyverse ecosystem. However, any code-based methodology is welcome - just please remember to share the code used to generate the results.
# 
# ## This week's TidyTuesday

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import altair as alt


# ## Reading the Data
# 
# Notice that `pandas` support parsing the date from the moment we read the data. 

# In[ ]:


gdpr_violations = pd.read_table('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_violations.tsv', parse_dates = ['date'])
gdpr_text = pd.read_table('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-21/gdpr_text.tsv')


# In[ ]:


gdpr_violations.rename({'name': 'country'}, inplace = True, axis = 1)
mask = gdpr_violations['date'] == pd.to_datetime('1970-01-01 00:00:00')
gdpr_violations['date'] = np.where(mask, np.datetime64("NaT") , gdpr_violations['date'])
gdpr_violations.head()


# *What's the total amount that has been fined?*

# In[ ]:


total_fine = gdpr_violations.price.sum()

print(f'Total amount fined has been: {total_fine:,}')


# Notice the lengths I had to go to in order to lump the countries whom had been least fined into a group of their own. 

# In[ ]:


fine_per_country = gdpr_violations.groupby('country')['price'].agg(np.nansum)
list_largest_countries = fine_per_country.nlargest(8).index
fine_per_country = fine_per_country.reset_index()
fine_per_country['country'] = np.where(fine_per_country.country.isin(list_largest_countries), fine_per_country['country'], 'other')
fine_per_country = fine_per_country.groupby('country')['price'].agg(np.nansum)


# In[ ]:


ax = fine_per_country.sort_values().plot(kind = 'barh')
sns.despine()
fmt = '${x:,}'
tick = mtick.StrMethodFormatter(fmt)
ax.xaxis.set_major_formatter(tick)
plt.xticks(rotation = 25)
plt.show()


# The greatest culprits tend to be France, Italy and Germany.
# 
# Next, let's plot the time series of the fines by upsampling by month

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))
ax.yaxis.set_major_formatter(tick)
gdpr_violations.set_index('date')['price'].resample('M', label = 'left').sum().plot(kind = 'bar', ax = ax)  
sns.despine()
plt.show()


# There seems to be a lot of variations. Let's include the countries in this analysis. However, stacked plots are notoriously difficult to plot in `matplotlib`. Thus, I instead switch to plot with `altair`.

# In[ ]:


by_year_by_country = gdpr_violations.set_index('date').groupby([pd.Grouper(freq='M', label = 'left'), 'country']).sum().reset_index()

list_largest_countries = by_year_by_country.nlargest(6, 'price').country
by_year_by_country = by_year_by_country.reset_index()
by_year_by_country['country'] = np.where(by_year_by_country.country.isin(list_largest_countries), by_year_by_country['country'], 'other')

alt.Chart(by_year_by_country, 
         width = 1000, height = 500).mark_bar().encode(
    x='date',
    y=alt.Y('price', title = 'Total fines'),
    color=alt.Color('country', sort = ['France', 'Italy', 'Germany'])
)


# In[ ]:


gdpr_violations[['controller', 'date', 'article_violated', 'type', 'summary', 'price']].sort_values('price', ascending = False)


# ### Which article was violated?

# In[ ]:


gdpr_violations['article_violated'].str.split('\\|', expand = True).melt()['value'].value_counts() 


# In[ ]:


gdpr_violations['article_violated'].str.split('\\|', expand = True, ).melt()['value'].str.extract('Art\\. ?(\\d+)')[0].value_counts()


# This time, it took some sidesteps to arrive at what Dave did: 

# In[ ]:


articles_wide = pd.concat([gdpr_violations, gdpr_violations['article_violated'].str.split('\\|', expand = True)], axis = 'columns')
id_vars = articles_wide.columns.difference([0, 1, 2, 3, 4])
separated_articles = articles_wide.melt(id_vars).sort_values('id').dropna(subset = ['value'], axis = 0).rename({'value': 'article_sep'}, axis = 1)
separated_articles['article_number'] = separated_articles['article_sep'].str.extract("Art\\. ?(\\d+)")
separated_articles


# In[ ]:


separated_articles['price_per_article'] = separated_articles.groupby('id')['price'].transform(lambda x: x/x.shape[0])
most_common_articles = separated_articles.groupby('article_number')['price'].agg('sum').sort_values(ascending = False)[:8].index.tolist()
most_common_articles


# In[ ]:


separated_articles['article_number'] = np.where(separated_articles['article_number'].isin(most_common_articles), separated_articles['article_number'], 'other')
separated_articles.groupby('article_number')['price_per_article'].agg([np.nansum, np.size]).sort_values('nansum', ascending = False)


# So it seems that article 5 was not only the most common but also the one that accrued more money in fines. The 58 is the huge fine that Google received in France. 
# 
# ## Distribution of fines per Type

# In[ ]:


# get 8 most profitable type of violations 
most_common_types = gdpr_violations.groupby('type')['price'].agg('sum').sort_values(ascending = False)[:8].index.tolist()
most_common_types


# In[ ]:


# lump violations that accrued less money into a category of their own
gdpr_violations['type'] = np.where(gdpr_violations['type'].isin(most_common_types), gdpr_violations['type'], 'other')
# order categories by their mean
order = gdpr_violations.query('price > 0').groupby('type')['price'].agg('median').sort_values(ascending = False).index.to_list()

# box plot
box_plot = alt.Chart(gdpr_violations.query('price>0'), width = 600, height = 500).mark_boxplot(color = "gray").encode(
    x=alt.X('price', scale=alt.Scale(type = 'log', base = 10)),
    y=alt.Y('type', sort = order))
# points
points = alt.Chart(gdpr_violations.query('price>0')).mark_circle(opacity = 0.3,
                                                                color = "red").encode(
    x=alt.X('price', scale=alt.Scale(type = 'log', base = 10)),
    y=alt.Y('type', sort = order))

box_plot + points


# The several category, where Google was included, is a huge outlier. With only to fines. Whereas the failure to implement.. and non-compliance seem to have many different observations. 

# ### Again we turn at articles
# 
# But this time connecting it with the text data. 

# In[ ]:


# get article titles
article_titles = gdpr_text[['article_title', 'article']].drop_duplicates()
# data munging to be able to merge
separated_articles['article_number'] = pd.to_numeric(separated_articles['article_number'], errors = "coerce")
# merge and get total fine per article
ax = (pd.merge(separated_articles, article_titles, left_on = 'article_number', right_on = 'article').rename({0: 'counts'}, axis = 1).
    groupby('article_title')['price_per_article'].agg('sum').sort_values().plot(kind = 'barh'))
# plot styling
plt.title('What articles got the most fines?')
ax.xaxis.set_major_formatter(tick)
plt.xticks(rotation = 25)
sns.despine()
plt.xlabel('Total Fine')
plt.ylabel('');


# In[ ]:


mask = gdpr_violations['controller'].str.contains('Vodafone')

alt.Chart(gdpr_violations[mask].groupby(['date', 'country'])['price'].agg([np.size, np.sum]).reset_index()).mark_circle().encode(
    x = 'date',
    y = alt.Y('sum', title = 'Total fines on this day'),
    color = 'country',
    size = 'size'
).properties(title = "Vodafone's GDPR violations")


# In[ ]:




