#!/usr/bin/env python
# coding: utf-8

# # Interactive Dashboard - Boxoffice Mojo Alltime Revenue Data
# ### [Data for 16,542 Hollywood movies](https://www.boxofficemojo.com/alltime/domestic.htm) as of August 3, 2019
# #### Aquick analysis of the words used in movie titles together with how they related to box-office revenue
# 
# This dashboard allows analysis of the different movies and studios to more easily read the data. 
# It is available on https://www.dashboardom.com/boxofficemojo

# In[ ]:


from IPython.display import IFrame
IFrame(src='https://www.dashboardom.com/boxofficemojo', width='100%', height=600)


# In[ ]:


import advertools as adv
import pandas as pd
pd.options.display.max_columns = None


# ## Top 20 Movies Sorted by Lifetime Gross US$

# In[ ]:


boxoffice = pd.read_csv('../input/boxoffice_march_2019.csv')
boxoffice.head(20).style.format({'lifetime_gross': '{:,}'})


# ## Most used words in movie titles
# ### `abs_freq`: number of occurrences of the word in movie titles
# ### `wtd_freq`: number of occurrences multiplied by the lifetime gross of the movie
# ### `rel_value`: value per occurrence in a movie title

# In[ ]:


word_freq_1 = adv.word_frequency(text_list=boxoffice['title'], 
                                 num_list=boxoffice['lifetime_gross'], 
                                 phrase_len=1, 
                                 rm_words=list(adv.stopwords['english']) + ['-', '', 'the'])
word_freq_1.head(20).style.format({'wtd_freq':"{:,}", 'rel_value': "{:,}"})


# ## Most used 2-word phrases in movie titles

# In[ ]:


word_freq_2 = adv.word_frequency(boxoffice['title'], boxoffice['lifetime_gross'], phrase_len=2,
                                rm_words=['of the', 'and the', 'in the', 'to the'])
word_freq_2.head(20)


# ## Movie titles containing "the last"

# In[ ]:


boxoffice[boxoffice['title'].str.contains('(?i)the last')].head(20)


# ## Movie titles containing "the dark"

# In[ ]:


boxoffice[boxoffice['title'].str.contains('(?i)the dark')].head(20)


# ## Script that generated the data
# The below script goes through the pages where the movies are listed, and create one big DataFrame that contains all movies. 

# In[ ]:


# import re
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup

# final_list = []
# for i in range(1, 280):
#     try:
#         page = 'http://www.boxofficemojo.com/alltime/domestic.htm?page=' + str(i) + '&p=.htm'
#         resp = requests.get(page)
#         soup = BeautifulSoup(resp.text, 'lxml')
#         table_data = [x.text for x in soup.select('tr td')[11:511]]  # trial and error to get the exact positions
#         temp_list = [table_data[i:i+5] for i in range(0, len(table_data[:-4]), 5)] # put every 5 values in a row
#         for temp in temp_list:
#             final_list.append(temp)
#         if not i%10:
#             print('getting page:', i)
#     except Exception as e:
#         break
# print('scraped pages:', i)
        
# na_year_idx = [i for i, x in enumerate(final_list) if x[4] == 'n/a']  # get the indexes of the 'n/a' values
# new_years = [1998, 1999, 1960, 1973]  # got them by checking online

# print(*[(i, x) for i, x in enumerate(final_list) if i in na_year_idx], sep='\n')
# print('new year values:', new_years)

# for na_year, new_year in zip(na_year_idx, new_years):
#     final_list[na_year][4] = new_year
#     print(final_list[na_year], new_year)
    

# regex = '|'.join(['\$', ',', '\^'])

# columns = ['rank', 'title', 'studio', 'lifetime_gross', 'year']

# boxoffice_df = pd.DataFrame({
#     'rank': [int(x[0]) for x in final_list],  # convert ranks to integers
#     'title': [x[1] for x in final_list],  # get titles as is
#     'studio': [x[2] for x in final_list],  # get studio names as is
#     'lifetime_gross': [int(re.sub(regex, '', x[3])) for x in final_list],  # remove special characters and convert to integer
#     'year': [int(re.sub(regex, '', str(x[4]))) for x in final_list],  # remove special characters and convert to integer
# })
# print('rows:', boxoffice_df.shape[0])
# print('columns:', boxoffice_df.shape[1])
# print('\ndata types:')
# print(boxoffice_df.dtypes)
# boxoffice_df.to_csv('path/to/file.csv', index=False)


# ## More information on the functions used: 
# 
# * [Datacamp article explaining how the `word_frequency` function is created and how it works](https://www.datacamp.com/community/tutorials/absolute-weighted-word-frequency)
# * [SEMrush article on text analysis for online marketers](https://www.semrush.com/blog/text-analysis-for-online-marketers/)
# * [`word_frequency` function documentation](https://advertools.readthedocs.io/en/master/advertools.html#module-advertools.word_frequency)
