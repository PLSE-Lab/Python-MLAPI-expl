#!/usr/bin/env python
# coding: utf-8

# Doing an analysis on the Academy Awards data requires selection of the right years and categories to maintain continuity.  This notebook visualizes the category shifts over time and determines category names that I want to consider the same for the purposes of analyzing award nominations and wins.
# 
# I'm going to be ignoring categories from before 1940 and those that won't be involved in multiple nominations for a film.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/database.csv")
df.head()


# Let's begin by showing the shift in categories over time. The names of categories haven't remained the same, and any analysis of them needs to understand this. 
# 
# Since there are mixed years as well, we'll first process the `Year` column.

# In[ ]:


df["YearInt"] = df.Year.apply(lambda x: int(x) if '/' not in x else int(x.split("/")[1]))
df = df[df.YearInt >= 1940]


# In[ ]:


def plot_award_years(award_names, min_years, max_years, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(style="ticks")
    award_list = []
    ct = 0
    for award in award_names:
        end_year = max_years[award]
        start_year = min_years[award]
        if end_year != start_year:
            award_list.append(award)
            plt.axhline(y=ct, linestyle='--', lw=0.5, c='k')
            ax.plot([start_year, end_year], [ct, ct], '-', lw=5)
            ct += 1
    ax.set_yticks(range(ct)); ax.set_yticklabels(award_list); ax.set_ylim(-1, ct-0.5);
    ax.set_xticks(range(1940, 2020, 5)); ax.set_xlim(1940, 2015); sns.despine();
    return fig


# In[ ]:


# Get the min and max years of each award
min_years = df.groupby("Award").YearInt.aggregate('min')
max_years = df.groupby("Award").YearInt.aggregate('max')

fig = plot_award_years(df.Award.unique(), min_years, max_years, figsize=(10,20));


# It's easier to consider categories in major groups:

# In[ ]:


groups = ['Sound', 'Music', 'Writing', 'Actor', 'Actress', 'Cinematography',
          'Makeup', ['Visual', 'Special Effects'], 'Costume', 'Best', 'Direct']


# In[ ]:


for group in groups:
    group = group if not isinstance(group, list) else "(" + "|".join(group)  + ")"
    gdf = df[df.Award.str.contains(group)]
    fig = plot_award_years(gdf.Award.unique(), min_years, max_years, figsize=None);


# From that, I'll make the following grouping rules. Some history checking showed that 'Sound Mixing' was 'Sound' beforehand. Everything else goes into 'Sound Editing'. There is probably a regex to make this simpler, but I think that spelling it out is better for anyone who is a historian of the awards who wants to quickly critique the mapping.

# In[ ]:


group_map = {'Sound Mixing': ['Sound', 'Sound Mixing'],
             'Sound Editing': ['Sound Recording', 'Sound Effects',
                               'Special Achievement Award (Sound Effects)',
                               'Special Achievement Award (Sound Effects Editing)',
                               'Special Achievement Award (Sound Editing)','Sound Effects Editing'],
             'Original Song': ['Music (Song)', 'Music (Song, Original for the Picture)',
                               'Music (Original Song)'],
             'Original Score': ['Music (Original Score)', 'Music (Scoring)',
                                'Music (Music Score of a Dramatic Picture)', 
                                'Music (Scoring of a Musical Picture)',
                                'Music (Music Score of a Dramatic or Comedy Picture)',
                                'Music (Music Score, Substantially Original)',
                                'Music (Scoring of Music, Adaptation or Treatment)',
                                'Music (Original Music Score)',
                                'Music (Original Score, for a Motion Picture [Not a Musical])',
                                'Music (Score of a Musical Picture, Original or Adaptation)',
                                'Music (Original Song Score)',
                                'Music (Original Dramatic Score)',
                                'Music (Scoring: Adaptation and Original Song Score)',
                                'Music (Scoring: Original Song Score and Adaptation -Or- Scoring: Adaptation)',
                                'Music (Original Song Score and Its Adaptation or Adaptation Score)',
                                'Music (Adaptation Score)',
                                'Music (Original Song Score and Its Adaptation -Or- Adaptation Score)',
                                'Music (Original Song Score or Adaptation Score)',
                                'Music (Original Musical or Comedy Score)'],
             'Original Screenplay': ['Writing (Original Screenplay)',
                                  'Writing (Original Story)',
                                  'Writing (Screenplay)',
                                  'Writing (Original Motion Picture Story)',
                                  'Writing (Motion Picture Story)',
                                  'Writing (Story and Screenplay)',
                                  'Writing (Screenplay, Original)',
                                  'Writing (Story and Screenplay, Written Directly for the Screen)',
                                  'Writing (Story and Screenplay, Based on Material Not Previously Published or Produced)',
                                  'Writing (Story and Screenplay, Based on Factual Material or Material Not Previously Published or Produced)',
                                  'Writing (Screenplay Written Directly for the Screen, Based on Factual Material or on Story Material Not Previously Published or Produced)',
                                  'Writing (Screenplay Written Directly for the Screen)'],
             'Adapted Screenplay': ['Writing (Screenplay, Adapted)',
                                   'Writing (Screenplay, Based on Material from Another Medium)',
                                   'Writing (Screenplay Adapted from Other Material)',
                                   'Writing (Screenplay Based on Material from Another Medium)',
                                   'Writing (Screenplay Based on Material Previously Produced or Published)',
                                   'Writing (Adapted Screenplay)'],
              'Best Actor': ['Actor', 'Actor in a Leading Role'],
              'Supporting Actor': ['Actor in a Supporting Role', ],
              'Best Actress': ['Actress', 'Actress in a Leading Role'],
              'Supporting Actress': ['Actress in a Supporting Role',],
              'Cinematography': ['Cinematography (Black and White)', 'Cinematography (Color)',
                                 'Cinematography'],
              'Makeup': ['Makeup', 'Makeup and Hairstyling'],
              'Visual Effects': ['Special Effects', 'Special Visual Effects',
                                 'Special Achievement Award (Visual Effects)',
                                 'Visual Effects'],
              'Costume Design': ['Costume Design (Black and White)', 'Costume Design (Color)',
                                 'Costume Design'],
              'Best Picture': ['Best Motion Picture', 'Best Picture'],
              'Best Director': ['Directing']}


# With the category mappings in hand, I can do some easier statistical work in a future notebook!

# In[ ]:




