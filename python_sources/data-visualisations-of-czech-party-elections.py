#!/usr/bin/env python
# coding: utf-8

# This work contains simple data scrapping, processing and visualization on two datasets from the Czech political party election.

# In[ ]:


# Import needed packages
import numpy as np
import pandas as pd
import sklearn as skit
import matplotlib
from matplotlib import pyplot as plt
import html5lib
import requests
import itertools
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# # 1. Webscrap data from the official Czech elections website
# We will only use data from the years: 2002, 2006, 2010, 2014 and 2018 for this work, aiming for two datasets: data of political **parties** and data of chosen **candidates**.

# ## 1.1 Scrapping data of parties
# As you cannot scrap data on a kaggle notebook, I ran the code on my personal machine and uploaded the needed csv files. I'll only leave the code I've used for the reader to see how it was done.

# In[ ]:


# ### 1. Save URLs from 2002 to 2018
# urls = []
# for i in range(0, 5):
#     year = str(2002 + i * 4)
#     urls.append('https://www.volby.cz/pls/kv{}/kv1111?xjazyk=CZ&xid=0&xdz=3&xnumnuts=4102&xobec=554961'.format(year))
    
# # save tables from URLs to list of Dataframes
# dfs_list = []
# for url in urls:
#     dfs_list.append(pd.read_html(url, flavor='html5lib'))
    
# # add Year column to each table and assign their years
# for i, j in itertools.product(range(0,5), range(0,2)):
#     dfs_list[i][j][('Year', 'Year')] = 2002 + i * 4
    
# # rename not matching column names in the table from 2002
# for i in range(0, len(dfs_list[0][1].columns)):
#     if dfs_list[0][1].columns[i] != dfs_list[1][1].columns[i]:
#         new_col = dfs_list[1][1].columns[i]
#         old_col = dfs_list[0][1].columns[i]
#         dfs_list[0][1] = dfs_list[0][1].rename(columns={old_col[0] : new_col[0]}, level=0)
#         dfs_list[0][1] = dfs_list[0][1].rename(columns={old_col[1] : new_col[1]}, level=1)
        
# # tables from 2006 to 2018 have wrong percentages in columns 3 and 6
# for i in range(1, 5):
#     dfs_list[i][1].iloc[:,3] = dfs_list[i][1].iloc[:,3] / 100
#     dfs_list[i][1].iloc[:,6] = dfs_list[i][1].iloc[:,6] / 100
    
# # split list of Dataframes by table types
# dfs_table1 = []
# dfs_table2 = []
# for i, j in itertools.product(range(0,2), range(0,5)):
#     if i == 0:
#         dfs_table1.append(dfs_list[j][i])
#     elif i == 1:
#         dfs_table2.append(dfs_list[j][i])
# # concat Dataframes with the same table type
# df1 = pd.concat(dfs_table1, ignore_index=True)
# df2 = pd.concat(dfs_table2, ignore_index=True)

# # export Dataframe to csv
# df2.to_csv('parties.csv', index=False)


# ## 1.2 Scrapping data of candidates
# Done similarly to the previous section. Again, I will leave the code for refference.

# In[ ]:


# ### 1. Save URLs from 2002 to 2018
# urls = []
# for i in range(0, 5):
#     year = str(2002 + i * 4)
#     urls.append('https://www.volby.cz/pls/kv{}/kv21111?xjazyk=CZ&xid=0&xv=23&xdz=3&xnumnuts=4102&xobec=554961&xstrana=0&xodkaz=1'.format(year))
    
# # save tables from URLs to list of Dataframes
# dfs_list = []
# for url in urls:
#     dfs_list.append(pd.read_html(url, flavor='html5lib'))
    
# # add Year column to each table and assign their years
# for i, j in itertools.product(range(0,5), range(0,2)):
#     dfs_list[i][j][('Year', 'Year')] = 2002 + i * 4
    
# # rename not matching column names in the table from 2002
# for i in range(0, len(dfs_list[0][1].columns)):
#     if dfs_list[0][1].columns[i] != dfs_list[1][1].columns[i]:
#         new_col = dfs_list[1][1].columns[i]
#         old_col = dfs_list[0][1].columns[i]
#         dfs_list[0][1] = dfs_list[0][1].rename(columns={old_col[0] : new_col[0]}, level=0)
#         dfs_list[0][1] = dfs_list[0][1].rename(columns={old_col[1] : new_col[1]}, level=1)
        
# # tables from 2006 to 2018 have wrong percentages in columns 3 and 6
# for i in range(1, 5):
#     dfs_list[i][1].iloc[:,3] = dfs_list[i][1].iloc[:,3] / 100
#     dfs_list[i][1].iloc[:,6] = dfs_list[i][1].iloc[:,6] / 100
    
# # split list of Dataframes by table types
# dfs_table1 = []
# dfs_table2 = []
# for i, j in itertools.product(range(0,2), range(0,5)):
#     if i == 0:
#         dfs_table1.append(dfs_list[j][i])
#     elif i == 1:
#         dfs_table2.append(dfs_list[j][i])
# # concat Dataframes with the same table type
# df1 = pd.concat(dfs_table1, ignore_index=True)
# df2 = pd.concat(dfs_table2, ignore_index=True)

# # export Dataframe to csv
# df2.to_csv('parties.csv', index=False)


# # 2. Data visualisation

# In[ ]:


# load the csv files we have prepared in the previous section
df_parties = pd.read_csv('../input/parties.csv', header=[0, 1])
df_candidates = pd.read_csv('../input/candidates.csv', header=[0, 1])


# ## 2.1 Number of parties vs number of chosen candidates

# In[ ]:


# extract yearly data from both DataFrames and count them
n_parties = pd.DataFrame(df_parties.iloc[:,9].value_counts(sort=False))
n_candidates = pd.DataFrame(df_candidates.iloc[:,10].value_counts(sort=False))


# In[ ]:


# assign custom column names
n_parties.columns = ['# of parties']
n_candidates.columns = ['# of chosen candidates']
# join these DataFrames
tmp = pd.concat([n_candidates,n_parties], axis=1)


# In[ ]:


# plot
ax = tmp.plot(kind='bar', title='Number of parties and chosen candidates throughout the years')
ax.set_xlabel('Year')


# ## 2.2 Number of candidates for a party

# In[ ]:


# get candidates by party throughout the years
party_col = df_candidates.iloc[:,1]
@interact
def plot_number_of_candidates_by_party(party= party_col.unique()):
    # get candidates and years column names
    candidates_col_name = df_parties.columns[4]
    years_col_name = df_parties.columns[9]
    # pull out needed data into temporal DataFrame
    tmp = df_parties[df_parties.iloc[:, 1] == party][[candidates_col_name, years_col_name]].set_index(years_col_name)
    # rename column
    tmp.columns = ['# of candidates']
    # plot DataFrame
    ax = tmp.plot(kind='bar', title='Number of \"{}\" candidates throughout the years'.format(party))
    ax.set_xlabel('Year')


# ## 2.3 Age structure of candidates
# ### 2.3.1 Age structure overall

# In[ ]:


# age structure overall
age_col = df_candidates.iloc[:,4]
ax = age_col.plot(kind='hist', title='Age structure overall')
ax.set_xlabel('Age')


# ### 2.3.2 Age structure by party

# In[ ]:


# age structure by party
party_col = df_candidates.iloc[:,1]
@interact
def plot_age_structure_by_party(party=party_col.unique()):
    ax = df_candidates[party_col == party].iloc[:,4].plot(kind='hist', title=party)
    ax.set_xlabel('Age')


# ### 2.3.3 Age structure by year

# In[ ]:


# group age structure data by year and plot them each seperately
year_col = df_candidates.iloc[:,10]
year_groups = df_candidates.iloc[:,[4,10]].groupby(df_candidates.columns[10])
for group in year_groups:
    group[1].columns = ['Candidate ages', '']
    group[1].set_index(group[1].columns[1]).plot(kind='hist', title=group[0])


# ## 2.4 Vote count

# In[ ]:


party_col = df_parties.iloc[:, 1]
@interact
def plot_votes_by_party(party= party_col.unique()):
    tmp = df_parties[party_col == party].iloc[:,[9, 2]]
    # rename column since plotting writes a multilevel index badly
    tmp.columns = ['Year', 'Votes']
    # set year data as index for xaxis 
    tmp = tmp.set_index(tmp.columns[0])
    # delete whitespaces in numbers for conversion
    tmp[tmp.columns[0]] = tmp[tmp.columns[0]].str.replace("\s+", "")
    # convert values to number so we can plot
    tmp[tmp.columns[0]] = pd.to_numeric(tmp[tmp.columns[0]])
    tmp.plot(kind='bar', title=party)


# ## 2.5 Candidates with title vs without title

# In[ ]:


# split data by title
has_title = df_candidates[df_candidates.iloc[:,3].str.contains('\.')]
no_title = df_candidates[~df_candidates.iloc[:,3].str.contains('\.')]


# In[ ]:


# put counted data into a new dataframe for concat
a = pd.DataFrame(has_title.iloc[:,10].value_counts(sort=False))
b = pd.DataFrame(no_title.iloc[:,10].value_counts(sort=False))
# rename columns
a.columns = ['with title']
b.columns = ['no title']


# In[ ]:


# concat both dataframes and plot
pd.concat([a, b], axis=1).plot(kind='bar', title='Candidates with a title vs candidates without one')


# ## 2.6 Categorizing candidates by titles and sex
# ### 2.6.1 Categorize by title

# In[ ]:


# define title categories
bachelor_titles = ['Bc.']
master_titles = ['Ing.', 'Ing. arch.', 'MUDr.', 'MDDr.', 'MVDr.', 'MgA.', 'Mgr.']
doc_titles = ['Ph.D.', 'JUDr.', 'PhDr.', 'RNDr.', 'CSc.']


# In[ ]:


# extract only needed columns
has_title = has_title.iloc[:,[3,10]]


# In[ ]:


# helper function to find atleast one match between two arrays
# return true if found, false otherwise
def match_atleast_one(a, b):
    return len([x for x in a if x in b]) != 0


# In[ ]:


# define variables
output = pd.DataFrame(columns=['year', 'bachelors', 'masters', 'doctors', 'non academic title', 'women', 'men'])
bachelors = masters = doctors = others = women = men = 0
year = 2002
# iterate through each line (candidate)
for i in has_title.itertuples():
    # if year changed, append counted data into output DataFrame
    if i[2] != year:
        output = output.append(dict({'year': year, 'bachelors': bachelors, 'masters': masters, 'doctors': doctors, 'non academic title': others, 'women': women, 'men': men}), ignore_index=True)
        # assign new year and reset output variables
        year = i[2]
        bachelors = masters = doctors = others = women = men = 0
    # tokenize each name of a candidate
    tokens = i[1].split()
    titles = []
    names = []
    # exctarct only title tokens
    for t in tokens:
        if '.' in t:
            titles.append(t)
        else:
            names.append(t)
    # if candidate is a woman, first name ends with 'a'
    if names[-1][-1] == 'a':
        women += 1
    else:
        men += 1
    # clasify candidate by titles
    # need to check every title since some have f.e. bachelors and masters in name and we want to classify them as master only
    if match_atleast_one(titles, doc_titles):
        doctors += 1
        continue
    elif match_atleast_one(titles, master_titles):
        masters += 1
        continue
    elif match_atleast_one(titles, bachelor_titles):
        bachelors += 1
        continue
    else:
        others += 1
# append output variables again at the end for the last year
output = output.append(dict({'year': year, 'bachelors': bachelors, 'masters': masters, 'doctors': doctors, 'non academic title': others, 'women': women, 'men': men}), ignore_index=True)


# In[ ]:


# plot title types
output.set_index('year').iloc[:,0:4].plot(kind='bar')


# ### 2.6.2 Categorize by sex

# In[ ]:


# plot num. of women/men
output.set_index('year').iloc[:,[4,5]].plot(kind='bar')


# ## 2.7 Most active candidates

# In[ ]:


# extract needed data
data = df_candidates.iloc[:,[3, 6, 7]].copy()
# rename columns
data.columns = ['name', 'party', 'votes']


# In[ ]:


# votes column contain numbers with different whitespaces
# thus we delete whitespaces and retype to int
intvotes = data.votes.str.replace('\s', '').astype(int)


# In[ ]:


# put int votes into Dataframe
data.votes = intvotes.values


# In[ ]:


# delete titles from data as some cadidates might have gotten a new title between elections
# there are also different whitespaces mixed in, so we replace every whitespace by a normal one since names might not match
# str.strip() for stripping any leading and trailing whitespaces
filtered_titles = data.name.str.replace(r'[A-z]+?\.', r'').str.replace(r'\s', r' ').str.strip()


# In[ ]:


# put filtered column in main Dataframe
data.name = filtered_titles.values


# In[ ]:


# count occurances and choose top 10 candidates
top10 = data.iloc[:,0].value_counts().head(10)
top10.plot(kind='bar', title='Candidates by participation in elections')


# In[ ]:


# put top 10 names into an array
top10 = top10.index.values


# In[ ]:


# group by party and name and sum their votes
data[data.name.isin(top10)].groupby(['party', 'name']).sum()


# In[ ]:




