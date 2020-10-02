#!/usr/bin/env python
# coding: utf-8

# CORD-19: Match SJR Rank journals
# 
# The SCImago Journal Rank (SJR indicator) is a measure of the scientific influence of academic journals. The scientific impact of journal is related to the number of citations that yours articles receveid. The idea of this kernel is exact match between the SJR Rank and all_sources_metadata file.
# 
# I) Contents
# 
# 1) Python libraries <br />
# 2) Reading all_sources_metadata <br />
# 3) Reading scimagojournalcountryrank <br />
# 4) Macht all_sources_metadata and scimagojournalcountryrank <br />
# 5) Show Result Figure <br />
# <br />
# Note: It is possible that the same journal, in both databases, has a variation in its name and makes some relationships unfeasible.<br />
# https://www.scimagojr.com/journalrank.php

# In[ ]:


# 1) Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# 2) Reading all_sources_metadata
#https://www.kaggle.com/sklasfeld/exploring-corona-abstracts

input_dir = "/kaggle/input/CORD-19-research-challenge/2020-03-13/"
data_file = ("%s/all_sources_metadata_2020-03-13.csv" % input_dir)
covid_19_journal = pd.read_csv(data_file)
covid_19_journal = covid_19_journal.reset_index().rename({"index":"paper_id"}, axis='columns')['journal']
covid_19_journal =list(set(covid_19_journal.dropna())) # remove duplicate
display(covid_19_journal[:5])
print(len(covid_19_journal))


# In[ ]:


# 3) Reading scimagojournalcountryrank

input_dir = "../input/scimagojournalcountryrank"
data_file = ("%s/scimagojr 2018.csv" % input_dir)
sjr_journal = pd.read_csv(data_file, sep=';')[['Title','Rank']]
display(sjr_journal.head())
print(len(sjr_journal))


# In[ ]:


# 4) Macht all_sources_metadata and scimagojournalcountryrank

result=[]
for ls1,ls3 in zip(sjr_journal.Title,sjr_journal.Rank):
    if ls1 in covid_19_journal:
        result.append([ls1,ls3])
result=pd.DataFrame(result)
result.columns=["Jornal","SJR"]
result.sort_values(by=['SJR'],inplace=True)
result.head(20)


# In[ ]:


# 5) Show Result Figure
#https://www.science-emergence.com/Articles/How-to-add-text-on-a-bar-with-matplotlib-/

fig, ax = plt.subplots(figsize=(20,5))
bar_x = result.SJR[:10].values
bar_height = result.Jornal[:10].values
bar_tick_label = result.SJR[:10].values
bar_label = result.Jornal[:10].values

bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width(), 1.05*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=80)

autolabel(bar_plot)

plt.ylim(0,20)
plt.xticks(rotation=70)
plt.title('SJR')
plt.xlabel('Rank SJR')


# Conclusion: I hope that the present kernel will be useful as the researchers will be able to focus on the papers deposited in the higher ranked journals.

# 
