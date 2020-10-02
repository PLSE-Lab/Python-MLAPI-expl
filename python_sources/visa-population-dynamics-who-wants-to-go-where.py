#!/usr/bin/env python
# coding: utf-8

# **Who wants to go where?**
# ----
# 
# Analysing population flux through VISA applications.
# 
# *NOTE: given the data provided, this focuses on people that NEED a Visa for their Schengen trip*
# 
# The objective is to see if any trends are able to emerge, such as:
# - Certain VISA country sources looking to visit a specific Schengen states much more than the average
# - Certain Schengen states denying more VISAs from a certain origin than the average etc...
# 
# This is also an example of basic data processing in pandas and a discussing about how much information is too much information on a graph

# **Data loading**
# ----
# 
# I am using the two CSV files and we will keep the year information to see if any VISA applications trends might change over these two years (remember this is only two years so it might be very annecdotal)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as pl # an other interface to matplotlib
from matplotlib.gridspec import GridSpec # more advanced subplots layout

# Loading data
df_2017 = pd.read_csv('../input/schengen-visa-stats/2017-data-for-consulates.csv')
df_2018 = pd.read_csv('../input/schengen-visa-stats/2018-data-for-consulates.csv')
df_2017['year'] = 2017
df_2018['year'] = 2018
# adding extra information: the year
df = pd.concat([df_2017, df_2018])
print(df.columns)


# **Basic data cleaning**
# ----
# 
# 3 steps here:
# - removing any data we won't use
# - renaming columns to make the data more easily usable
# - changing data types to ones easier to work with / fixing or invalidating bad values (ex: a summary row containing strings with '%' in a numeric column) 

# In[ ]:




# Data cleaning
# dropping data we won't use
df.drop(
    [
        'Airport transit visas (ATVs) applied for ',
        ' ATVs issued (including multiple)',
        'Multiple ATVs issued',
        'ATVs not issued ',
        'Not issued rate for ATVs',
        'Not issued rate for ATVs and uniform visas ',
        'Not issued rate for uniform visas',
        'Total ATVs and uniform visas applied for',
        'Total ATVs and uniform visas issued  (including multiple ATVs, MEVs and LTVs) ',
        'Total ATVs and uniform visas not issued',
        'Not issued rate for uniform visas',
        'Share of MEVs on total number of uniform visas issued',

    ],
    axis=1,
    inplace=True
)
# renaming columns
df.rename(
    columns={
        'Schengen State': 'schengen_state',
        'Country where consulate is located': 'country',
        'Consulate': 'city',
        'Uniform visas applied for': 'applications',
        'Total  uniform visas issued (including MEV) \n': 'tuv_issued',
        'Multiple entry uniform visas (MEVs) issued': 'mev_issued',
        'Total LTVs issued': 'ltv_issued',
        'Uniform visas not issued': 'tuv_not_issued',
    },
    inplace=True
)

# fixing data format: to string to int/float when possible (for the fields where it makes sense)
def us_format_to_int(value):
    try:
        return int(value.replace(',', '')) if isinstance(value, str) else value
    except ValueError:
        return np.NaN


for col in ['applications', 'mev_issued', 'tuv_issued', 'ltv_issued', 'tuv_not_issued']:
    df[col] = df[col].apply(lambda r: us_format_to_int(r))

print(df.head())


# **Grouping data to look for insights**
# ----
# 
# We will first have a look at which coutries (meaning consulates in the given country) applies to which schengen state

# In[ ]:


# ---- who goes where ? / who wants to go where?
country2schengen_state_stats = df.groupby(['schengen_state', 'country']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['schengen_state', 'country'])
# for the two following, we sort by nb of applications since this will be the axis order for our next graph>
# other possible choice: sort by state/country name
schengen_state_stats = df.groupby(['schengen_state']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['applications'], ascending=False).reset_index()
country_stats = df.groupby(['country']).sum()[['applications', 'mev_issued']].reset_index().sort_values(by=['applications'], ascending=False).reset_index()
print(country2schengen_state_stats)
print(schengen_state_stats.head())
print(country_stats.head())


# Let's prepare the data a bit more to make some graphs. Basically just creating scalar indexes associated with labels (countries, states), easier to work this way for the coming graphs.

# In[ ]:


# we now add an index to our sorted data: this index will be used to ensure everying is displayed in the correct order 
# over the x and y axis through the various graphs (ticks values)
schengen_state_order2index = {schengen_state_stats.schengen_state[i]: i for i in range(len(schengen_state_stats.schengen_state))}
country2index = {country_stats.country[i]: i for i in range(len(country_stats.country))}

country2schengen_state_stats['ss_index'] = country2schengen_state_stats.schengen_state.apply(lambda r: schengen_state_order2index[r])
country2schengen_state_stats['cc_index'] = country2schengen_state_stats.country.apply(lambda r: country2index[r])
schengen_state_stats['ss_index'] = schengen_state_stats.schengen_state.apply(lambda r: schengen_state_order2index[r])
country_stats['cc_index'] = country_stats.country.apply(lambda r: country2index[r])


# **First graph: too much information?**
# ----
# 
# **At the center:** for each Schengen state (x) and applying country (x), sorted by number of applications (see previous block), the two circles at the x/y intersection represents the number of applications (blue) and the number of VISA issued for the given state/country couple
# 
# **top:** VISA applications (blue) and issued VISAs (yellow) per Schengen state
# 
# **left:** VISA applications (blue) and issued VISAs (yellow) per Aplying Country

# In[ ]:


def plot_visa_counts(country2schengen_state_stats, schengen_state_stats, country_stats, keys=None, extra_margin=9, figsize=(16, 16)):
    fig = pl.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    gs = GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])

    ax_joint.set_xticks(range(len(schengen_state_stats.schengen_state)))
    ax_joint.set_xticklabels(schengen_state_stats.schengen_state, rotation=70)
    ax_joint.set_yticks(range(len(country_stats.country)))
    ax_joint.set_yticklabels(country_stats.country)
    ax_joint.set_xlim(-1, len(schengen_state_stats.schengen_state) + 1)
    ax_joint.set_ylim(-1 - extra_margin, len(country_stats.country) + 1)

    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_x.set_xticks(range(len(schengen_state_stats.schengen_state)))
    ax_marg_x.set_xticklabels(schengen_state_stats.schengen_state)
    ax_marg_x.set_xlim(-1, len(schengen_state_stats.schengen_state) + 1)
    ax_marg_y = fig.add_subplot(gs[1:4, 3])

    ax_marg_y.set_yticks(range(len(country_stats.country)))
    ax_marg_y.set_yticklabels(country_stats.country)
    ax_marg_y.set_ylim(-1 - extra_margin, len(country_stats.country) + 1)

    keys = ['applications', 'mev_issued'] if keys is None else keys
    width = 0.7 / len(keys)
    for i, key in enumerate(['applications', 'mev_issued']):
        # for the size of the dots (s), we use square root: from the doc -> s :
        # scalar or array_like, shape (n, ), optional
        # The marker size in points**2. Default is rcParams['lines.markersize'] ** 2.
        ax_joint.scatter(country2schengen_state_stats['ss_index'], country2schengen_state_stats['cc_index'], s=1 * np.sqrt(country2schengen_state_stats[key]), alpha=0.3)
        ax_marg_x.bar(schengen_state_stats['ss_index'] + (i-0.5)*width, schengen_state_stats[key], width=width, alpha=0.5, label=key)
        ax_marg_y.barh(country_stats['cc_index'] + (i-0.5)*width, country_stats[key], height=width, alpha=0.5)

    # Turn off tick labels on marginals
    pl.setp(ax_marg_x.get_xticklabels(), visible=False)
    pl.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel('Schengen States')
    ax_joint.set_ylabel('Applying Countries')

    # Set labels on marginals
    ax_marg_y.set_xlabel('VISA / Schengen state')
    ax_marg_x.set_ylabel('VISA / Applying country')
    
    ax_marg_x.legend(loc='upper right')
    pl.suptitle('Schengen VISA applications and issued (mev)', size=16)
    pl.tight_layout()
    pl.show()


plot_visa_counts(country2schengen_state_stats, schengen_state_stats, country_stats, keys=None, figsize=(18, 26))


# **Let's zoom in**
# ----
# 
# **Can anyone read anything?** Maybe it's not too bad on a huge display but here it's a little too packed to see any interesting information.
# Let's filter things a bit: the 16 most popular Schengen states and the 16 most common VISA sources.
# 
# AGAIN:
# 
# **At the center:** for each Schengen state (x) and applying country (x), sorted by number of applications (see previous block), the two circles at the x/y intersection represents the number of applications (blue) and the number of VISA issued for the given state/country couple
# 
# **top:** VISA applications (blue) and issued VISAs (yellow) per Schengen state
# 
# **left:** VISA applications (blue) and issued VISAs (yellow) per Aplying Country

# In[ ]:


schengen_state_stats_small = schengen_state_stats.head(16)
country_stats_small = country_stats.head(16)
country2schengen_state_stats_small = country2schengen_state_stats[country2schengen_state_stats.apply(lambda r: r.cc_index in set(country_stats_small.cc_index) and r.ss_index in set(schengen_state_stats_small.ss_index), axis=1)]
plot_visa_counts(country2schengen_state_stats_small, schengen_state_stats_small, country_stats_small, keys=None, extra_margin=1)


# **A quick analysis**
# ----
# Just a few informations we can take from this graph:
# - **Some schengen states are stricter than other**: top graph, France VS Germany for example: Frances recieves way more applications than Germany does but issues less VISAs
# - **Certain schengen states recieve applications from everywhere, some heavily from just a few country**: France, Germany, Italy or the Netherlands have an important variety of countries applying (even tho you can notice a few differences), while Finland for example mostly recieves applications from Russia.
# - **The same way, certain countries mostly apply at specific schengen state**s: Algeria With France, Belarus with Poland and Lituania, Morocco with Spain and France etc...
# - **Schengen States have heavily different VISA issual rates for a same country of application**: France or Spain rejects a lot of applications coming from China while Germany doesn't, while these 3 states issual rate for russian applications are roughly the same.
# - **Some applications origins are way more likely to be rejected than others**: Algeria has more applications sent overall than Belarus but less are approved. A note: noce how Belarus applies to states with high acceptation ratios and Algeria low acceptation. Then again, who can first, the chicken or the egg? Does Belarus gets a higher ratio of accepted VISA because it applies to high acceptation states (eg Lithuania) or or does Lithuania has a high acceptation rate because most applications come from Belarus, a country with a high acceptation rate? 
# 
# These are just a few examples and a lot more can be found by digging into this data.
# 
# **IMPORTANT REMINDER**: *the idea here is not to discuss why these relations between certain countires exist (than can be multiple: common history, colonialism, langauge, geographical closeness), simply to point our that a simple graph like that already allows to notice certain patterns of human population mouvement already and such. Same for why certain countries get rejected more / why certain couple don't work as well: this graph only gives what's actually happens, not the reason (quotas, application qualities, temporary political tensions, again common history...)*
# 
