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


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('white')


# ## Tour de France Winners
# 
# Dave normally starts the screencasts by looking at some categorical variable with a quick plot. 

# In[ ]:


tdf_winners = pd.read_csv("/kaggle/input/tour-de-france-winners/tdf_winners.csv", parse_dates = ['start_date', 'born', 'died'])
tdf_winners.head()


# In[ ]:


tdf_winners['birth_country'].value_counts().plot(kind = 'barh',
                                                title = 'What countries were the Most Tour de France winners from?')
sns.despine()


# Now, let's find out which cyclyst wont the most Tours.

# In[ ]:


tdf_winners['winner_name'].value_counts()


# In[ ]:


tdf_winners.groupby(['winner_name', 'birth_country'])['edition'].size().sort_values(ascending = False).head(10)


# ## How is the age distrbution changed over time?
# 
# However, let's first create the decade variable.

# In[ ]:


tdf_winners['year'] = tdf_winners['start_date'].dt.to_period('Y').astype(str).astype(int)
tdf_winners['decade'] = 10*(tdf_winners['year'].astype(int) // 10)
tdf_winners['decade'].value_counts().sort_index()


# Now, let's calculate the average winner age by decade! Notice that pandas makes it natural to do what Dave wanted to do with across. 

# In[ ]:


grouping_vars = ['age', 'height', 'weight', 'time_margin', 'time_overall', 'speed']
tdf_winners.eval('speed = distance / time_overall', inplace = True)
winners_by_decade = tdf_winners.groupby('decade')[grouping_vars].agg(np.nanmean).round(2).reset_index()
winners_by_decade


# I love dplyr, however, non having to deal with NSE problems in pandas makes it much easier to reuse code. In this case, let's see if 

# In[ ]:


for grouping_var in grouping_vars:
    winners_by_decade.plot(x = 'decade', y = grouping_var, ylim = 0)
    sns.despine()


# Let's zoom in on the time margin and eliminate the first two observations. 

# In[ ]:


(winners_by_decade.query('decade > 1910').eval('time_margin = time_margin * 60')
 .plot(x = 'decade', y = 'time_margin', ylim = 0, title = 'Tour de France races have been getting closer'))
sns.despine()


# In[ ]:


winners_by_decade.plot('decade', 'speed', ylim = 0, title = "Tour de France winners have been getting faster")
sns.despine()


# ## Life expectancy
# 
# Let's look at the life expectancy of the winners. How many have died? Note that `pd.Series.count()` counts the number of non-missing observations

# In[ ]:


survival_df = tdf_winners.drop_duplicates(subset = 'winner_name').dropna(axis = 0, subset = ['born'])
survival_df['died'].count()


# In[ ]:


survival_df['birth_year'] = pd.to_numeric(survival_df['born'].dt.to_period('Y').astype(str), errors = "coerce")
survival_df['death_year'] = pd.to_numeric(survival_df['died'].dt.to_period('Y').astype(str), errors = "coerce")


# In[ ]:


survival_columns = ['winner_name', 'birth_year', 'death_year', 'dead']
survival_df['dead'] = np.where(survival_df['death_year'].isna(), 0, 1)
survival_df = survival_df[survival_columns]
survival_df.head()


# In[ ]:


survival_df['death_year'] = np.where(survival_df['death_year'].isna(), 2020, survival_df['death_year'])
survival_df.eval('age_at_death = death_year - birth_year', inplace = True)
survival_df.head()


# In[ ]:


get_ipython().system('pip install lifelines')


# In[ ]:


from lifelines import KaplanMeierFitter


# In[ ]:


T = survival_df["age_at_death"]
E = survival_df["dead"]
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)


# In[ ]:


kmf.plot()
plt.title('Survival function of Tour de France Winners');
sns.despine()
plt.savefig('survival.png',
           dpi=300)


# In[ ]:


kmf.median_survival_time_


# Thus, the median life expectancy of a Tour de France winner is 77. 

# ## Stage Data

# In[ ]:


tdf_stages = pd.read_csv("/kaggle/input/tour-de-france-winners/tdf_stages.csv", parse_dates = ['Date'])
tdf_stages['year'] = pd.to_numeric(tdf_stages['Date'].dt.to_period('Y').astype('str'), errors = 'coerce')
stage_data = pd.read_csv("/kaggle/input/tour-de-france-winners/stage_data.csv")
print(tdf_stages.columns)
print(stage_data.columns)


# In[ ]:


tdf_stages.head()


# Let's create the stage data that we are going to use later on to join. 

# In[ ]:


stage_data['Stage'] = stage_data['stage_results_id'].str.split('-').str.get(1)


# In[ ]:


stages_joined = stage_data.merge(tdf_stages, how = 'inner', on = ['Stage', 'year'])
stages_joined.head()


# In[ ]:


stages_joined['rank'].value_counts().sort_values(ascending = False)


# In[ ]:


stages_joined['rank'] = pd.to_numeric(stages_joined['rank'], errors = 'coerce')


# In[ ]:


stages_joined.groupby('Winner_Country').agg({'Stage': 'size',
                                             'rank': np.nanmedian}).sort_values('Stage', ascending = False)


# In[ ]:


stages_joined.groupby(['year', 'Stage'])['rank'].agg('size').plot(kind = 'hist')


# Let's try to replicate the `add_count` function

# In[ ]:


stages_joined = (stages_joined.groupby(['year', 'Stage'])['rank'].agg(pd.Series.count).
    reset_index().rename({'rank': 'competitors'} ,axis = 1).merge(stages_joined, how = 'right', on = ['year', 'Stage']).
    eval('percentile = 1 - rank / competitors'))
stages_joined['Stage'] = pd.to_numeric(stages_joined['Stage'], errors = 'coerce')


# No country does better than average! No country is dominant within the first stage

# In[ ]:


stages_joined.query('Stage == 2').groupby('Winner_Country').agg({'Stage': 'size',
                                             'percentile': np.nanmedian}).sort_values('Stage', ascending = False).round(3)


# In[ ]:


total_points_per_year = stages_joined.groupby(['year', 'rider'])['points'].agg(np.sum).reset_index().rename({'points': 'total_points'},
                                                                                                           axis = 1)
total_points_per_year['point_rank'] = total_points_per_year.groupby(['year'])['total_points'].rank(pct = True)
total_points_per_year.head(10)


# Does the performance on the first stage predict the final performance?

# In[ ]:


first_and_total = stages_joined.query('Stage == 1').merge(total_points_per_year, how = 'right',
                                       on = ['year', 'rider'])
first_and_total['percentile'] = pd.cut(first_and_total['percentile'], [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
first_and_total['percentile'] = first_and_total['percentile'].dropna(axis = 0)
chart = sns.boxplot(x = 'percentile', y = 'point_rank', data = first_and_total)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart.set(xlabel='c', ylabel='common ylabel')
sns.despine()
plt.savefig('boxplot.png',
           dpi=300)


# In[ ]:




