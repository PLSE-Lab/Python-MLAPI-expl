#!/usr/bin/env python
# coding: utf-8

# # Airline Tickets Search Engine Results Pages and Landing Pages 
# This is a quick exploration of the [tickets and airlines dataset](https://www.kaggle.com/eliasdabbas/flights-serps-and-landing-pages) that contains landing page data in addition to Google search results data.  
# 
# For more details on the first part of the dataset, you can check this [tutorial on SEMrush](https://www.semrush.com/blog/analyzing-search-engine-results-pages/) detailing how the keywords were developed, and how the data was imported and analyzed.  

# In[ ]:


get_ipython().system('pip install advertools')


# In[ ]:


import advertools as adv
import pandas as pd
pd.options.display.max_columns = None
import plotly.graph_objects as go
import plotly
print('advertools v' + str(adv.__version__))
print('pandas v' + str(pd.__version__))
print('plotly v' + str(plotly.__version__))


# In[ ]:


flights = pd.read_csv('/kaggle/input/flights-serps-and-landing-pages/flights_serp_scrape.csv')
flights.head(2)


# ## Column names
# 
# As you can see above, columns are prepended with `serp_` or `scrape_` to indicate the source of the data. The reason is that there is an overalp of elements from both sources. There is a title in the SERP page, and a title scraped from the landing page for example. 
# 
# ### Columns with multiple results
# In some cases, like header tags and images, pages can contain multiple items. In these cases the elements have been merged into one string, where the items are separated by two @ signs e.g. `first h2 tag@@second tag@@third tag`. You simply need to split by "@@" to get them as lists.

# ## Visual summary of SERP positions

# In[ ]:


serps_to_plot = flights.copy()
serps_to_plot.columns = flights.columns.str.replace('serp_', '').str.replace('scrape_', '')
def plot_data(serps_to_plot, num_domains=10, select_domain=None):
#     df = pd.DataFrame(serp_results, columns=serp_results[0].keys())
    df = serps_to_plot
    if select_domain:
        df = df[df['displayLink'].isin(select_domain)]
    top_domains = df['displayLink'].value_counts()[:num_domains].index.tolist()
    top_df = df[df['displayLink'].isin(top_domains)]
    top_df_counts_means = (top_df
                           .groupby('displayLink', as_index=False)
                           .agg({'rank': ['count', 'mean']}))
    top_df_counts_means.columns = ['displayLink', 'rank_count', 'rank_mean']
    top_df = (pd.merge(top_df, top_df_counts_means)
              .sort_values(['rank_count', 'rank_mean'],
                           ascending=[False, True]))
    rank_counts = (top_df
                   .groupby(['displayLink', 'rank'])
                   .agg({'rank': ['count']})
                   .reset_index())
    rank_counts.columns = ['displayLink', 'rank', 'count']
    summary = (df
               .groupby(['displayLink'], as_index=False)
               .agg({'rank': ['count', 'mean']})
               .sort_values(('rank', 'count'), ascending=False)
               .assign(coverage=lambda df: (df[('rank', 'count')]
                                            .div(df[('rank', 'count')]
                                                 .sum()))))
    summary.columns = ['displayLink', 'count', 'avg_rank', 'coverage']
    summary['displayLink'] = summary['displayLink'].str.replace('www.', '')
    summary['avg_rank'] = summary['avg_rank'].round(1)
    summary['coverage'] = (summary['coverage'].mul(100)
                           .round(1).astype(str).add('%'))
    num_queries = df['queryTime'].nunique()

    fig = go.Figure()
    fig.add_scatter(x=top_df['displayLink'].str.replace('www.', ''),
                    y=top_df['rank'], mode='markers',
                    marker={'size': 30, 'opacity': 1/rank_counts['count'].max()})

    fig.add_scatter(x=rank_counts['displayLink'].str.replace('www.', ''),
                    y=rank_counts['rank'], mode='text',
                    text=rank_counts['count'])

    for domain in rank_counts['displayLink'].unique():
        rank_counts_subset = rank_counts[rank_counts['displayLink'] == domain]
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[0], mode='text',
                        marker={'size': 50},
                        text=str(rank_counts_subset['count'].sum()))

        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-1], mode='text',
                        text=format(rank_counts_subset['count'].sum() / num_queries, '.1%'))
        fig.add_scatter(x=[domain.replace('www.', '')],
                        y=[-2], mode='text',
                        marker={'size': 50},
                        text=str(round(rank_counts_subset['rank']
                                       .mul(rank_counts_subset['count'])
                                       .sum() / rank_counts_subset['count']
                                       .sum(), 2)))

    minrank, maxrank = min(top_df['rank'].unique()), max(top_df['rank'].unique())
    fig.layout.yaxis.tickvals = [-2, -1, 0] + list(range(minrank, maxrank+1))
    fig.layout.yaxis.ticktext = ['Avg. Pos.', 'Coverage', 'Total<br>appearances'] + list(range(minrank, maxrank+1))

#     fig.layout.height = max([600, 100 + ((maxrank - minrank) * 50)])
    fig.layout.height = 600
    fig.layout.yaxis.title = 'SERP Rank (number of appearances)'
    fig.layout.showlegend = False
    fig.layout.paper_bgcolor = '#eeeeee'
    fig.layout.plot_bgcolor = '#eeeeee'
    fig.layout.autosize = True
    fig.layout.margin.r = 2
    fig.layout.margin.l = 120
    fig.layout.margin.pad = 0
    fig.layout.hovermode = False
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.zeroline = False
    return fig

fig = plot_data(serps_to_plot.query('gl == "us"'))
fig.layout.title = 'SERP Postions for USA'
fig


# In[ ]:


fig = plot_data(serps_to_plot.query('gl == "uk"'))
fig.layout.title = 'SERP Postions for UK'
fig


# ## Content Quantity (Inventory) per Query
# Of course it's about quality and not quantity. But the higher the number of results for a certain query, the more crowded and competitive it is going to be.  
# These are the queries sorted by the number of results (coupled with the `gl` (geo location) of the query)

# In[ ]:


(flights
 .drop_duplicates(subset=['serp_searchTerms', 'serp_gl'])
 [['serp_searchTerms', 'serp_gl', 'serp_totalResults']]
 .sort_values('serp_totalResults', ascending=False)[:20]
 .reset_index(drop=True)
 .style.format({'serp_totalResults': '{:,}'})
 .set_caption('Queries by number of results'))


# The numbers are clearly extremely high. A better approach might be to search for the queries quoted, to force Google to search for the phrase, and not any word in the query, "flights to singapore" for example. 
# 
# ## Header tags
# These pages were scraped on March 11, 2020, with the Coronavirus a major news story disrupting the travel business. We can quickly check how many pages are talking about this topic. As you can see below, it seems that about 4% of the landing pages contain the word "corona" in their body text.

# In[ ]:


flights['scrape_body_text'].str.contains('[kc]orona', regex=True, case=False).mean()


# ### Duplicated H1 Tags
# Are the sites using unique H1s in their pages? Or is there a lot of overlap/duplication?

# In[ ]:


top_h1_tags = flights['scrape_h1'].value_counts()
top_h1_tags[2:12]


# As you can see above, there are a few are that duplicated exactly across different landing pages.  
# We can check more to see if they are from the same sites, different site, etc.

# In[ ]:


(flights[flights['scrape_h1']=='Flights to Vienna']
 [['serp_searchTerms', 'serp_link', 'scrape_h1']]
 .sort_values('serp_link')
 .style.set_caption('Pages with "Flights to Vienna" as their H1 tag'))


# To explore further, the following are all URLs and H1 tags for all queries containing "vienna".  
# You probably want to have content that is unique and stands out, so it's good to explore on a large scale, what the competition is doing. 

# In[ ]:


flights[flights['serp_searchTerms'].str.contains('vienna')][['serp_searchTerms', 'serp_link', 'scrape_h1']].sort_values('serp_link')


# ### Number of H1 tags per page
# Let's see if there is an overuse of H1 tags or not. 

# In[ ]:


(flights['scrape_h1'].dropna()
 .str.split('@@').str.len()
 .value_counts(normalize=False)
 .to_frame().reset_index()
 .rename(columns={'index': 'h1_tags_per_page',
                  'scrape_h1': 'count'})
 .assign(perc=lambda df: df['count'].div(df['count'].sum()))
 .style.format({'perc': '{:.1%}'})
.hide_index())


# The majority have one, but it might be interesting to see what people are trying to achieve by having multiple H1's. The following is a sample of pages having more than two tags. This is just an overview, and obviously you would have to dig deeper to figure out how to tackle this. 

# In[ ]:


flights[flights['scrape_h1'].str.split('@@').str.len() > 2].sort_values(['serp_displayLink', 'serp_link'])['scrape_h1'].str.split('@@')[:15]


# ## Extract structured entities from text
# The entities referred to are simply patterns that can tell us about the content.  
# `advertools` provides a few functions to extract some entities that might provide some good insights. 

# In[ ]:


[f for f in dir(adv) if f.startswith('extract')]


# ## Extract currency
# Currency symbols signify prices, and these are very important for understanding the market. In this case we can easily compare prices of tickets for example.  
# `extract_currency` returns a dictionary with several values.  
# The best approach is to assign it to a name, and explore the keys available in the dictionary. 

# In[ ]:


serp_title_currency = adv.extract_currency(flights['serp_title'])
serp_title_currency.keys()


# In[ ]:


serp_title_currency['overview']


# 2,425 or around 60% of the titles seem to contain a currency symbol.

# In[ ]:


serp_title_currency['top_currency_symbols']


# In[ ]:


{sym[0] for sym in serp_title_currency['currency_symbol_names'] if sym}


# Surrounding text: when calling the function, you can specify how many characters you want extracted to the right and left of the currency symbol. This is to give you context on where it appears. If you are sure that all prices are displayed properly you can specify `left_chars` to be one, and `right_chars` to be three or four for this case, and you will get clean prices extracted. Watch out for different number formatting and currency usage conventions, as they differ from country to country. 

# In[ ]:


[x for x in serp_title_currency['surrounding_text'] if x][:15]


# ## Emoji

# In[ ]:


snippet_emoji = adv.extract_emoji(flights['serp_snippet'])
snippet_emoji.keys()


# In[ ]:


snippet_emoji['overview']


# In[ ]:


snippet_emoji['top_emoji']


# ## Questions
# It might interesting to see if websites are using questions to encourage people to click. Let's see how many snippets contain questions. 

# In[ ]:


snippet_questions = adv.extract_questions(flights['serp_snippet'])
snippet_questions.keys()


# In[ ]:


snippet_questions['overview']


# 23.75% of landing page snippets contain questions. That's a significant portion.  
# Let's see if there are duplicates across sites. 

# In[ ]:


pd.Series(' '.join(q) for q in snippet_questions['question_text'] if q).value_counts()[:10]


# ## Hashtags

# In[ ]:


adv.extract_hashtags(flights['scrape_body_text'].dropna())['overview']


# Not so many hashtags in the body text. 
# 
# ## Body text
# Let's see how long the body text is distributed, by checking the word counts. 

# In[ ]:


body_text_length = flights['scrape_body_text'].dropna().str.split().str.len()


# In[ ]:


fig = go.Figure()
fig.add_histogram(x=body_text_length.values)
fig.layout.title = 'Distribution of word count of body text'
fig.layout.bargap = 0.1
fig.layout.xaxis.title = 'Number of words per page'
fig.layout.yaxis.title = 'Count'
fig


# In[ ]:


fig = go.Figure()
fig.add_histogram(x=body_text_length[body_text_length < 1700].values)
fig.layout.title = 'Distribution of word count of body text (for pages having less than 1,700 words)'
fig.layout.bargap = 0.1
fig.layout.xaxis.title = 'Number of words per page'
fig.layout.yaxis.title = 'Count'
fig


# There is a lot more that can be analyzed, this was just a quick exploration of the dataset. 
