#!/usr/bin/env python
# coding: utf-8

# # Streams and bumps in python
# 
# Throughout my research work, I always think about how best to present data. My tools for the trade are [matplotlib](https://matplotlib.org/), [pandas](http://pandas.pydata.org/) and [seaborn](http://seaborn.pydata.org/). However, sometimes there are graph varieties that are not readily available using `python`. This notebook shows how to create two little-used (in python), but nonetheless interesting graphs that may prove useful to others.

# ## 1. Stream graphs
# 
# What appears to be a cross between a [stacked area graph](http://pandas.pydata.org/pandas-docs/stable/visualization.html#area-plot) and and [violin plot](http://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot), stream graphs can show a visually appealing and story rich method for presenting frequency data in multiple categories across a time-like dimension.
# 
# The inspiration for this section will be to recreate the main figure of a New York Times' article of [A Visual History of Which Countries Have Dominated the Summer Olympics](https://www.nytimes.com/interactive/2016/08/08/sports/olympics/history-olympic-dominance-charts.html).
# 
# The data used was provided by [The Guardian](https://www.kaggle.com/the-guardian) at Kaggle: [Olympic Sports and Medals, 1896-2014](https://www.kaggle.com/the-guardian/olympic-games). The first step will be to see the form of the data and manipulate it into a suitable format: rows as countries, columns as olympic games, values as medal counts.

# In[31]:


import pandas as pd


# In[32]:


summer = pd.read_csv("../input/summer.csv")
summer.head()


# Here we see that each entry is an `Athlete` representing a `Country`, of a given `Gender`, who won a `Medal` in some `Event` in the Olympics in `City` in a particular `Year`. For team-based sports, multiple individuals can receive medals, but we'll want to count these medals only once.

# In[33]:


summer = summer[["Year", "Sport", "Country", "Gender", "Event", "Medal"]].drop_duplicates()


# Then using a groupby on Country and Year, if we count the Medals and unstack the result, we end up with a dataframe in the desired format.

# In[34]:


summer = summer.groupby(["Country", "Year"])["Medal"].count().unstack()
summer.head()


# Now, the NYT only includes eight named countries (the rest are grouped by continent). So we'll want to identify what these countries are in the list, based on their [IOC country codes](https://en.wikipedia.org/wiki/List_of_IOC_country_codes). There's some interesting trivia in which countries/regions/groups are included/excluded/merge/divide with time. At this point we can ignore the rest of the data and just focus on these categories (this is an exercise in making a plot after-all, not a deep dive into analysing the data).

# In[35]:


countries = [
    "USA", # United States of America
    "CHN", # China
    "RU1", "URS", "EUN", "RUS", # Russian Empire, USSR, Unified Team (post-Soviet collapse), Russia
    "GDR", "FRG", "EUA", "GER", # East Germany, West Germany, Unified Team of Germany, Germany
    "GBR", "AUS", "ANZ", # Australia, Australasia (includes New Zealand)
    "FRA", # France
    "ITA" # Italy
]

sm = summer.loc[countries]
sm.loc["Rest of world"] = summer.loc[summer.index.difference(countries)].sum()
sm = sm[::-1]


# Before any plotting, let's define colours similar to those in the NYT graph. For simplicity, I'll be using the named colours in matplotlib.

# In[36]:


country_colors = {
    "USA":"steelblue",
    "CHN":"sandybrown",
    "RU1":"lightcoral", "URS":"indianred", "EUN":"indianred", "RUS":"lightcoral",
    "GDR":"yellowgreen", "FRG":"y",  "EUA":"y", "GER":"y", 
    "GBR":"silver",
    "AUS":"darkorchid", "ANZ":"darkorchid",
    "FRA":"silver",
    "ITA":"silver",
    "Rest of world": "gainsboro"}


# Let's present this data as a stacked bar plot. This will show: i) the total number of medals won (total height) and ii) compare the relative number of medals countries won in different years.

# In[37]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.2)


# In[38]:


colors = [country_colors[c] for c in sm.index]

plt.figure(figsize=(12,8))
sm.T.plot.bar(stacked=True, color=colors, ax=plt.gca())

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

# Set labels and remove superfluous plot elements
plt.ylabel("Number of medals")
plt.title("Stacked barchart of select countries' medals at the Summer Olympics")
sns.despine()


# This plot is quite different to the desired graph. In particular, the bars don't have any continuity (which we'll achieve by using the plot.area method of DataFrames. And secondly, we don't have zero values for when the World Wars occurred.

# In[39]:


sm[1916] = np.nan # WW1
sm[1940] = np.nan # WW2
sm[1944] = np.nan # WW2
sm = sm[sm.columns.sort_values()]


# In[40]:


plt.figure(figsize=(12,8))
sm.T.plot.area(color=colors, ax=plt.gca(), alpha=0.5)

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

# Set labels and remove superfluous plot elements
plt.ylabel("Number of medals")
plt.title("Stacked areachart of select countries' medals at the Summer Olympics")
plt.xticks(sm.columns, rotation=90)
sns.despine()


# This is looking much better. There are two features we are missing: i) this plot has a baseline (i.e. the bottom of the chart) set at zero, whereas we want the baseline to wiggle about ii) the transitions between times are jagged.
# 
# To fix the baseline, instead of using pandas's plot.area method, we use the stackplot function from matplotlib. Here, we show what the different baselines look like.
# 

# In[41]:


for bl in ["zero", "sym", "wiggle", "weighted_wiggle"]:
    plt.figure(figsize=(6, 4))
    f = plt.stackplot(sm.columns, sm.fillna(0), colors=colors, baseline=bl, alpha=0.5, linewidth=1)
    [a.set_edgecolor(sns.dark_palette(colors[i])[-2]) for i,a in enumerate(f)] # Edges to be slighter darker
    plt.title("Baseline: {}".format(bl))
    plt.axis('off')
    plt.show()


# By changing the baseline away from zero, it no longer makes sense to include a yscale. The basic differences between the baselines is:
# * sym - plots area symmetrically about y=0
# * wiggle - minimizes the (average) slopes
# * weighted_wiggle - minimizes the (weighted average) slopes, where it is weighted by the size.
# 
# For more about how these are derived, check out the paper linked from [Lee Byron's website](http://leebyron.com/streamgraph/), the creator of this type of graph.
# 
# `wiggle` seems like the best choice here. Now to move onto making the graph smooth.
# 
# Here we'll use the `interpolate.PchipInterpolator` from scipy - a monotonic piecewise cubic interpolator. For more details on how this works and why it is suitable, see the appendix at the end of this document.

# In[42]:


from scipy import interpolate

def streamgraph(dataframe, **kwargs):
    """ Wrapper around stackplot to make a streamgraph """
    X = dataframe.columns
    Xs = np.linspace(dataframe.columns[0], dataframe.columns[-1], num=1024)
    Ys = [interpolate.PchipInterpolator(X, y)(Xs) for y in dataframe.values]
    return plt.stackplot(Xs, Ys, labels=dataframe.index, **kwargs)


# In[43]:


plt.figure(figsize=(12, 8))

# Add in extra rows that are zero to make the zero region sharper
sm[1914] = np.nan
sm[1918] = np.nan
sm[1938] = np.nan
sm[1946] = np.nan
sm = sm[sm.columns.sort_values()]

f = streamgraph(sm.fillna(0), colors=colors, baseline="wiggle", alpha=0.5, linewidth=1)
[a.set_edgecolor(sns.dark_palette(colors[i])[-2]) for i,a in enumerate(f)] # Edges to be slighter darker

plt.axis('off')

# Reverse the order of labels, so they match the data
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc=(1, 0.35))

# Instead of ticks, left-align the dates and names of the Summer Olympics
# Use annotate to draw a line to the edge. 
cities = pd.read_csv("../input/summer.csv")[["Year", "City"]].drop_duplicates().set_index("Year")["City"].to_dict()
for c in cities:
    plt.annotate(xy=(c, plt.ylim()[1]), 
                 xytext=(c, plt.ylim()[0]-100), s="{} {}".format(c, cities[c]), 
                 rotation=90, verticalalignment="bottom", horizontalalignment="center", alpha=0.5, zorder=1,
                 arrowprops={"arrowstyle":"-", "zorder":0, "color":"k", "alpha":0.5})

# Block out regions when the World Wars occurred
plt.axvspan(xmin=1914, xmax=1918, color='white')
plt.axvspan(xmin=1938, xmax=1946, color='white')

plt.show()


# ## 2. Bumps plots
# 
# A bunch of text about these. Let's link to some pages that describe them.
# 
# Here we'll be looking at rankings in the Winter Olympics. The format here is the same as the Summer Olympics.

# In[44]:


winter = pd.read_csv("../input/winter.csv")
winter = winter[["Year", "Sport", "Country", "Gender", "Event", "Medal"]].drop_duplicates()
winter = winter.groupby(["Country", "Year"])["Medal"].count().unstack()


# In[45]:


from collections import defaultdict

def add_widths(x, y, width=0.1):
    """ Adds flat parts to widths """
    new_x = []
    new_y = []
    for i,j in zip(x,y):
        new_x += [i-width, i, i+width]
        new_y += [j, j, j]
    return new_x, new_y

def bumpsplot(dataframe, color_dict=defaultdict(lambda: "k"), 
                         linewidth_dict=defaultdict(lambda: 1),
                         labels=[]):
    r = dataframe.rank(method="first")
    r = (r - r.max() + r.max().max()).fillna(0) # Sets NAs to 0 in rank
    for i in r.index:
        x = np.arange(r.shape[1])
        y = r.loc[i].values
        color = color_dict[i]
        lw = linewidth_dict[i]
        x, y = add_widths(x, y, width=0.1)
        xs = np.linspace(0, x[-1], num=1024)
        plt.plot(xs, interpolate.PchipInterpolator(x, y)(xs), color=color, linewidth=lw, alpha=0.5)
        if i in labels:
            plt.text(x[0] - 0.1, y[0], s=i, horizontalalignment="right", verticalalignment="center", color=color)
            plt.text(x[-1] + 0.1, y[-1], s=i, horizontalalignment="left", verticalalignment="center", color=color)
    plt.xticks(np.arange(r.shape[1]), dataframe.columns)


# In[46]:


winter_colors = defaultdict(lambda: "grey")
lw = defaultdict(lambda: 1)

top_countries = winter.iloc[:, 0].dropna().sort_values().index
for i,c in enumerate(top_countries):
    winter_colors[c] = sns.color_palette("husl", n_colors=len(top_countries))[i]
    lw[c] = 4

plt.figure(figsize=(18,12))
bumpsplot(winter, color_dict=winter_colors, linewidth_dict=lw, labels=top_countries)
plt.gca().get_yaxis().set_visible(False)

cities = pd.read_csv("../input/winter.csv")[["Year", "City"]].drop_duplicates().set_index("Year")["City"]
plt.xticks(np.arange(winter.shape[1]), ["{} - {}".format(c, cities[c]) for c in cities.index], rotation=90)

# Add in annotation for particular countries
host_countries = {
    1924: "FRA",
    1928: "SUI",
    1932: "USA",
    1948: "SUI",
    1952: "NOR",
    1960: "USA",
    1964: "AUT",
    1968: "FRA",
    1976: "AUT",
    1980: "USA",
    1988: "CAN",
    1992: "FRA",
    1994: "NOR",
    2002: "USA",
    2010: "CAN",
}
for i,d in enumerate(winter.columns):
    if d in host_countries:
        plt.axvspan(i-0.1, i+0.1, color=winter_colors[host_countries[d]], zorder=0, alpha=0.5)

sns.despine(left=True)


# ## A. 'Nice' interpolation
# 
# In order to make the above plots have aesthetically (to my eyes) curves, I use [**P**iecewise **C**ubic **H**ermite **I**nterpolating **P**olynomials](https://epubs.siam.org/doi/10.1137/0717021). Here I'll present why these are nice, why regular cubic splines are not, and how it relates to *smoothstep*.
# 
# First, some examples.

# In[47]:


x = [0, 1, 2, 3, 4]
xs = np.linspace(0, 4)
y = [0, 1, 1, 1, 0]

plt.plot(x, y, 'o-k', label="raw", alpha=0.5, markersize=16)
plt.plot(xs, interpolate.interp1d(x, y, kind="cubic")(xs), label="cubic spline", lw=4, alpha=0.8)
plt.plot(xs, interpolate.pchip(x, y)(xs), label="PCHIP", lw=4, alpha=0.8)

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
sns.despine()


# Above is a very simple step. But this immediately shows the problems of cubic splines - there are extra humps that aren't present in the data. This is a general problem that they do not maintain *monotonicity* of the desired function.
# 
# PCHIP solves this problem, even though piecewise it uses cubic splines as well. Below will be some formalism for how to think about these functions. This is taken from the paper referenced above.
# 
# Let $\pi: a = x_1 < x_2 < \cdots < x_n = b,$ be a partition of the interval $I = [a,b]$.
# 
# Let ${y_i: i=1, 2, \cdots, n}$ be a given set of monotonic data values. That is, each entry is larger or equal (*or* smaller or equal) to the proceeding value.
# 
# We want to find some function $f(x)$ such that:
# * $f(x_i) = y_i$
# * $f(x)$ is differentiable
# * monotonicity is preserved.
# 
# In general, on some subinterval $I_i = [x_i, x_{i+1}]$, $f(x)$ can be written as:
# 
# $$ f(x) = f(x_i)H_1(x) + f(x_{i+1})H_2(x) + d_iH_3(x) + d_{i+1}H_4(x) $$
# 
# $d_j = f'(x_j)$ and $H_k$ is the cubic Hermite basis functions:
# * $H_1(x) = \phi((x_{i+1} - x)/h_i)$
# * $H_2(x) = \phi((x - x_i)/h_i)$
# * $H_3(x) = -h_i\psi((x_{i+1} - x)/h_i)$
# * $H_4(x) = h_i\psi((x - x_i)/h_i)$
# * $h_i = x_{i+1} - x_i$
# * $\phi(t) = 3t^2 - 2t^3$
# * $\psi(t) = t^3 - t^2$
# 
# The choice comes from the $d$ variables, which exist to control the gradient (since, without loss of generalization, both functions are zero at the ends of the intervals. The paper goes into some requirements for this, which are beyond the scope of this notebook.
# 
# An interesting thing to note is that $\phi(x)$ has the same form as the first order *smoothstep* function:
# 
# $$ S_N(x) = x^{N+1} \sum_{n=0}^{N} {N+n \choose n }{2N + 1 \choose N - 1}(-x)^n$$
# $$ \mathrm{smoothstep}(x) = S_1(x) = 3x^2 - 2x^3$$
# 
# Thus, in the bumps charts by adding in the flat part `add_widths`, the joining sections by nature are *smoothsteps*.

# In[ ]:




