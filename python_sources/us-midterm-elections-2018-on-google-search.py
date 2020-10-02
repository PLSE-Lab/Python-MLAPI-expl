#!/usr/bin/env python
# coding: utf-8

# # US Midterm Elections 2018 on Google Search
# ### Who are the most influential websites?
# #### Search engine results pages data for 1,230 candidates
# 
# The upcoming election is going to be determined by (sorry to state the obvious) who people vote for. People vote based on what they know about their candidates. People go online to know more about those candidates to make up their minds. Quite often, they use a search engine called Google!
# 
# This is what they see...
# 
# ### The Keywords
# 
# Although the majority of searches are probably for "election", "midterm elections", or "elections 2018", the keywords that will ultimately determine the results, are the ones that are about the candidates that people will vote for. 
# 
# I got the list of updated candidates from [ProPublica](https://www.propublica.org/datastore/dataset/2018-midterm-election-congressional-candidates). They have included and added some great meta data about candidates (state, party, URL, and more).  
# 
# 
# ### Methodology
# 
# I ran a Google query for each of the candidates' names, which returned ten results each. Each result contains the visible information on the search results page (title, domain, snippet, etc.), as well as some additional meta data like the total number of results and rank.  
# For more information on how this works, you can check out my tutorial on how to use [Google Custom Search Engine to get search data with Python.](https://www.kaggle.com/eliasdabbas/search-engine-results-pages-serps-research)
# 
# Keep in mind that this approach treats all candidates equally, which is known not to be the case. Several states are pretty much determined which way they will vote, and some are not. There are differences across states in terms of population and representation. Even across equal states, there might be a different number of candidates for each.   
# This is not a problem, because we are not trying to predict the outcome of the election here. We are trying to find out which websites are the most influential based on their strength in SEO. It will be clear that there is a set of sites that are consistently prominent on search results pages. 
# 
# Let's first load the main libraries, and quickly explore the ProPublica data. 

# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
# import advertools as adv

cx = 'CUSTOM_SEARCH_ENGINE_ID'  # used to get data from Google's custom search engine
key = 'API_KEY'                 


# In[ ]:


candidates = pd.read_csv('../input/midterm_election_candidates_2018_0921.csv')
print(candidates.shape)
candidates.head(2)


# The columns that look interesting are `party`, `state`, and `url`. The first two for segmenting candidates, and the URL for comparing their personal URL with actual results and seeing how these sites rank for the main keyword, the name of the candidate. 

# In[ ]:


candidates.url.isna().agg(['mean','sum'])


# Checking for missing data on URLs it seems that 259 candidates don't have a personal website, which is around 21% of the candidates. I reached out to ProPublica and they confirmed that indeed these candidates do not have personal websites. That's a lot, and could be a hint as to how serious the candidate is.
# 
# Let's quickly take a look at the candidates, and visualize them by state and by party. 

# In[ ]:


cand_summary = (candidates['party']
                .value_counts()
                .to_frame()
                .assign(perc=candidates['party']
                        .value_counts(normalize=True))
                .assign(cum_perc=lambda df: df['perc'].cumsum()))
print('Top 10 parties\ntotal candidates:', candidates.shape[0])
cand_summary.head(10).style.format({'perc': '{:.1%}', 'cum_perc': '{:.1%}'})


# In[ ]:


fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(10, 6)
ax.bar(cand_summary.index[:10], cand_summary['party'][:10], color='olive')
ax.set_frame_on(False)
ax.grid()
ax.set_title('US Midterm Elections Candidates Per Party - Top 10', fontsize=18)
ax.tick_params(labelsize=14)
ax.set_xlabel('Party', fontsize=14)
ax.set_ylabel('Number of Candidates', fontsize=14)
plt.show()


# In[ ]:


cand_summary_state = (candidates['state']
                      .value_counts()
                      .to_frame()
                      .assign(perc=candidates['state']
                              .value_counts(normalize=True))
                      .assign(cum_perc=lambda df: df['perc'].cumsum()))
print('Top 10 states (582 out of 1,230 candidates)')#, cand_summary_state['state'].sum())
cand_summary_state.head(10).style.format({'perc': '{:.1%}', 'cum_perc': '{:.1%}'})


# In[ ]:


fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(10, 6)
ax.bar(cand_summary_state.index[:10], 
       cand_summary_state['state'][:10], color='olive')
ax.set_frame_on(False)
ax.grid()
ax.set_title('US Midterm Elections Candidates Per State - Top 10', fontsize=18)
ax.tick_params(labelsize=14)
ax.set_xlabel('State', fontsize=14)
ax.set_ylabel('Number of Candidates', fontsize=14)
plt.show()


# ### Generating Search Data
# I used the name from the `clean_name` column for the queries. 
# This is the call to import the data from Google:  
# `serp_candidates = adv.serp_goog(q=candidates['clean_name'].tolist(), cx=cx, key=key, gl='us')`  
# This will make 1,230 requests, put each set of results in a DataFrame, and merge all results together. For such a number of requests, it's better to split them into smaller requests, in case you get an issue midway.  
# The search results will be merged with the `candidates`  dataset later for further analysis.
# 
# SERP is the acronym used frequenty by SEO's for "search engine results pages". 

# In[ ]:


serp_candidates = pd.read_csv('../input/serp_candidates_oct_21.csv')
print(serp_candidates.shape)
serp_candidates.head(2)


# In[ ]:


top_domains = (serp_candidates
               .displayLink.str.replace('.*(house.gov)', 'house.gov')
               .value_counts())
top_domains.to_frame()[:25]


# In[ ]:


fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(8,8)
ax.set_frame_on(False)
ax.barh(top_domains.index.str.replace('www.', '')[:15], top_domains.values[:15])
ax.invert_yaxis()
ax.grid()
ax.tick_params(labelsize=14)
ax.set_xticks(range(0, 1000, 100))
ax.set_title('Top Domains Search Ranking - 2018 Midterm Elections', pad=20, fontsize=20)
ax.text(0.5, 1, 'Searching for 1,230 Candidates\' Names', fontsize=16,
        transform=ax.transAxes, ha='center')
ax.set_xlabel('Number of appearances on SERPs', fontsize=14)
plt.show()


# No surprise with the top social media sites and Wikipedia. The election-specific ones that seem to be big are also prominent (Ballotpedia, VoteSmart, and OpenSecrets).  house.gov is mainly for incumbent candidates who already have a profile page on this website in the form of `<name>.house.gov`.
# I would have expected a much stronger presence by the news sites.  
# We can now define a DataFrame that contains the top sites (15 in this case - `top_df`), and then merge it with the `candidates` DataFrame: 

# In[ ]:


top_df = (serp_candidates
          [serp_candidates['displayLink'].str.replace('.*(house.gov)', 'house.gov')
          .isin(top_domains.index[:15])].copy())
top_df['displayLink'] = top_df['displayLink'].str.replace('.*(house.gov)', 'house.gov')

# similar to top_df, but containing all domains:
all_serp_can = (pd.merge(serp_candidates, candidates, how='left', left_on='searchTerms', 
                         right_on='clean_name')
                .sort_values(['searchTerms', 'rank'])
                .reset_index(drop=True))

top_serp_cand = pd.merge(top_df, candidates, how='left', left_on='searchTerms', right_on='clean_name').sort_values(['searchTerms'])
top_serp_cand.head(1)


# ### Appearances on SERPs and average rank per domain

# In[ ]:


(all_serp_can
 .pivot_table('rank', ['displayLink'], aggfunc=['count', 'mean'])
 .reset_index()
 .sort_values([('count', 'rank')], ascending=False)
 .reset_index(drop=True)
 .head(15).style.format({('mean', 'rank'): '{:.2f}'}))


# ### Visualizing the different domains 
# Although we now know the average rank from the above table, we can get a better perspective to visualize how those results are distributed.  It would also be interesting to see if there are any differences across the two major parties.  
# The darker the circle the more frequent that domain appears in that position. 

# In[ ]:


for i, party in enumerate(['DEM', 'REP']):
    fig, ax = plt.subplots(1, 1, facecolor='#ebebeb')
    fig.set_size_inches(12, 8)

    ax.set_frame_on(False)
    ax.scatter((top_serp_cand[top_serp_cand['party']==party]
                   .sort_values('displayLink')['displayLink']
                   .str.replace('www.', '')), 
                  (top_serp_cand[top_serp_cand['party']==party]
                   .sort_values('displayLink')['rank']), 
                  s=850, alpha=0.02, edgecolor='k', lw=2, color='navy' if party == 'DEM' else 'darkred')
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.yaxis.set_ticks(range(1, 11))
    ax.tick_params(labelsize=15, rotation=30, labeltop=True,
                   labelbottom=False, length=8)
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('Search engine results page rank', fontsize=16)
    ax.set_title('Midterm Election Search Ranking for Candidate Names - ' + party, pad=95, fontsize=24)
    fig.tight_layout()
    plt.show()


# It doesn't seem there is a noticeable difference across the two major parties when it comes to SEO presence. The dominant domains are the same, and their distribution also looks similar. 
# 
# We can do the same for the top five states for example (including all parties).

# In[ ]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt


for i, state in enumerate(cand_summary_state.index[:5]):
    fig, ax = plt.subplots(facecolor='#ebebeb')
    fig.set_size_inches(12, 8)
    ax.set_frame_on(False)
    ax.scatter((top_serp_cand[top_serp_cand['state']==state]
                   .sort_values('displayLink')['displayLink']
                   .str.replace('www.', '')), 
                  (top_serp_cand[top_serp_cand['state']==state]
                   .sort_values('displayLink')['rank']), 
                  s=850, alpha=0.1, edgecolor='k', lw=2, color='olive')
    ax.grid(alpha=0.25)
    ax.invert_yaxis()
    ax.yaxis.set_ticks(range(1, 11))
    ax.tick_params(labelsize=15, rotation=30, labeltop=True,
                   labelbottom=False, length=8)
    ax.xaxis.set_ticks_position('top')
    ax.set_ylabel('Search engine results page rank', fontsize=16)
    ax.set_title('Midterm Election Search Ranking for Candidate Names - ' + state, pad=95, fontsize=20)
    fig.tight_layout()
    plt.show()


# Again, no major differences. This confirms that the top sites are quite consistent, and seem to be influential when it comes to what people know about their candidates. 
# 
# Now we can take a look at the candidates' own personal websites, and how they rank on SERPs, split by party. 

# In[ ]:


own_domain = (all_serp_can
 [all_serp_can.url.str.replace('https?://(www.)?|/$', '')
  .eq(all_serp_can.displayLink)][['rank','party']].copy())

own_domain_summary = (own_domain
                      .pivot_table('rank', 'party',
                                   aggfunc=['count', 'mean', 'std'])
                      .sort_values([('count', 'rank')], ascending=False))
own_domain_summary


# In[ ]:


fig, ax = plt.subplots(facecolor='#ebebeb')
fig.set_size_inches(14, 7)
for party in own_domain['party'].unique():
    ax.scatter('party', 'rank', data=own_domain.query('party == @party'), s=800, alpha=0.1)
    ax.set_frame_on(False)
    ax.grid()
    ax.tick_params(labelsize=13)
    ax.set_yticks(range(1, 11))
    ax.invert_yaxis()
    ax.set_title('Search Results Position for Candidates\' Own Domain', fontsize=18, pad=30)
plt.show()


# The majority rank high, but not as strongly and consistently as for example house.gov. 
# 
# 
# So, this was a quick summary of how influential sites can be uncovered, especially when you have a long list of keywords / topics. The same keywords could be run in different ways, for example, by explicitly saying "name last_name congress" or "name last_name state". The visualizations can be filtered to show several possible combinations of party, state, even city, zip code and district. 
# 
# For details on how to produce similar data, you can check the [tutorial](https://www.kaggle.com/eliasdabbas/search-engine-results-pages-serps-research) mentioned above. Feel free to share any [feedback](https://twitter.com/eliasdabbas), or [bugs.](https://github.com/eliasdabbas/advertools) 
