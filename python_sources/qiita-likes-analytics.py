#!/usr/bin/env python
# coding: utf-8

# - [1. Items Stats](#1.-Items-Stats)
#     - Load Items
#     - Yearly Items Stats
#     - Monthly Items Stats
#     - Exclude outliers
#     - Distribution of Likes Count Based on Created-date
# - [2. Likes Stats](#2.-Likes-Stats)
#     - Load Likes
#     - Yearly Likes Stats
#     - Monthly Likes Stats
#     - Exclude outliers
# - [3. Merge Likes and Items](#3.-Merge-Likes-and-Items)
# - [4. Cumulative Sum of Likes](#4.-Cumulative-Sum-of-Likes)
# - [5. Likes Count Matrix by CreatedDate and LikedDate](#5.-Likes-Count-Matrix-by-CreatedDate-and-LikedDate)
#     - Likes Count
#     - Likes Count / Yearly Items
#     - Likes Count / Yearly Likes
# - [6. Period from Created-date to Liked-date](#6.-Period-from-Created-date-to-Liked-date)
#     - Basic Statistics
#     - Distribution
#     - Distribution of Likes for Trend Itmes

# In[ ]:


from datetime import datetime
import os

import numpy as np
import pandas as pd
import dask.dataframe as dd

import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("../input"))

sns.set(style="darkgrid")


# ---
# # 1. Items Stats

# ## Load Items

# In[ ]:


items = pd.read_csv(
    "../input/items.tsv", 
    sep="\t", 
    parse_dates=["created_at"],
    index_col="created_at",
).sort_index()
items.head()


# ## Yearly Items Stats

# In[ ]:


def items_stats(items, freq):
    
    groupby = items.groupby(pd.Grouper(level="created_at", freq=freq))

    items_count = groupby.item_id.count().rename("items")
    authors_count = groupby.author.nunique().rename("authors")
    likes_count = groupby.likes_count.sum().rename("likes")
    
    items_cumsum = items_count.cumsum().rename("items cumsum")

    items_per_author = (items_count / authors_count).rename("items / author")
    likes_per_item = (likes_count / items_count).rename("likes / item")

    return pd.concat([
        items_count,
        items_cumsum,
        authors_count,
        items_per_author,
        likes_count,
        likes_per_item,
    ], axis=1)


# In[ ]:


yearly_items_stats = items_stats(items, "YS")
display(yearly_items_stats.reset_index().style.hide_index().format({
    "created_at": "{:%Y}",
    "items / author": "{:.2f}",
    "likes / item": "{:.2f}",
}))


# ## Monthly Items Stats

# In[ ]:


monthly_items_stats = items_stats(items, "MS")
display(monthly_items_stats.head())


# In[ ]:


display(monthly_items_stats["2016-01-01":"2018-01-01"])


# In[ ]:


monthly_items_stats.to_csv("monthly_items.tsv", sep="\t")


# In[ ]:


for y in monthly_items_stats.columns:
    sns.relplot(data=monthly_items_stats.reset_index(),
                kind="line", x="created_at", y=y, aspect=2)


# ## Exclude outliers

# In[ ]:


items_for_plot = monthly_items_stats[monthly_items_stats.index.month != 12]["2013":]
for y in items_for_plot.columns:
    sns.relplot(data=items_for_plot.reset_index(),
                kind="line", x="created_at", y=y, aspect=2)


# ## Distribution of Likes Count Based on Created-date

# In[ ]:


items.likes_count.describe()


# In[ ]:


(items
 .groupby(pd.Grouper(level="created_at", freq="YS"))
 .likes_count.describe()
)


# In[ ]:


ax = sns.distplot(items.likes_count)
# ax.set_xscale("log")
# ax.set_yscale("log")


# In[ ]:


(items
 .groupby(pd.cut(
     items.likes_count, 
     bins=[0, 1, 10, 100, 99999], 
     right=False,
 ))
 .likes_count
 .agg([{
     "count": "count", 
     "sum": "sum", 
     "proportion": lambda x: x.sum() / items.likes_count.sum()
 }])
)


# In[ ]:


likes_count_dist = (
    items
    .groupby([
        pd.Grouper(level="created_at", freq="YS"),
        pd.cut(items.likes_count, bins=[0, 1, 10, 100, 99999], right=False),
    ])
    .likes_count.sum()
    .groupby("created_at")
    .apply(lambda x: x / x.sum())
    .rename("percent")
)
display(likes_count_dist.unstack("likes_count"))


# In[ ]:


sns.relplot(
    data=likes_count_dist.reset_index(),
    kind="line",
    x="created_at", y="percent", hue="likes_count",
)


# In[ ]:


def stack_bar_plot(df, x, hue, xticklabels=None):

    f, ax = plt.subplots()

    a = df.groupby(x).cumsum().unstack(hue)
    index = reversed(a.columns)
    a = a.rename(columns=str).reset_index()
    
    color_codes = ["pastel", "muted", "deep", "dark"]
    
    for category, colors in zip(index, color_codes):
        sns.set_color_codes(colors)
        g = sns.barplot(
            data=a, 
            x=x, y=str(category), 
            color="b",
            label=str(category),
        )
        g.set(ylabel="")
        if xticklabels:
            g.set(xticklabels=xticklabels)    
    
    ax.legend(ncol=1, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)


stack_bar_plot(
    likes_count_dist.rename(columns=str), 
    x="created_at", hue="likes_count",
    xticklabels=list(yearly_items_stats.index.year),
)


# In[ ]:


sns.heatmap(
    data=likes_count_dist.unstack("likes_count"),
    annot=True,
    fmt=".1%",
    yticklabels=yearly_items_stats.index.year,
)


# ---
# # 2. Likes Stats

# ## Load Likes

# In[ ]:


get_ipython().run_cell_magic('time', '', 'likes = pd.read_csv(\n    "../input/likes.tsv", \n    sep="\\t", \n    parse_dates=["liked_at"],\n    index_col="liked_at",\n).sort_index()\ndisplay(likes.head())')


# ## Yearly Likes Stats

# In[ ]:


def likes_stats(likes, items, freq):
    
    grouped_likes = (
        dd.from_pandas(likes, 2)
        .groupby(pd.Grouper(level="liked_at", freq=freq))
    )
    
    grouped_items = (
        dd.from_pandas(items, 2)
        .groupby(pd.Grouper(level="created_at", freq=freq))
    )
    
    likes_count = grouped_likes.item_id.count().rename("likes")
    likers_count = grouped_likes.liker.nunique().rename("likers")

    items_count = grouped_items.item_id.count().rename("items")
    
    likes_per_liker = (likes_count / likers_count).rename("likes / liker")
    likes_per_item = (likes_count / items_count).rename("likes / item")

    return (
        dd.concat([
            likes_count,
            likers_count,
            likes_per_liker,
            likes_per_item,
        ], axis=1).compute().sort_index()
    )


# In[ ]:


get_ipython().run_cell_magic('time', '', 'yearly_likes_stats = likes_stats(likes, items, "YS")\ndisplay(yearly_likes_stats.reset_index().style.hide_index().format({\n    "liked_at": "{:%Y}",\n    "likes / liker": "{:.2f}",\n    "likes / item": "{:.2f}",\n}))')


# ## Monthly Likes Stats

# In[ ]:


get_ipython().run_cell_magic('time', '', 'monthly_likes_stats = likes_stats(likes, items, "MS")\ndisplay(monthly_likes_stats.head())')


# In[ ]:


display(monthly_likes_stats["2016-01-01":"2018-01-01"])


# In[ ]:


monthly_likes_stats.to_csv("monthly_likes.tsv", sep="\t")


# In[ ]:


for y in monthly_likes_stats.columns:
    sns.relplot(data=monthly_likes_stats.reset_index(),
                kind="line", x="liked_at", y=y, aspect=2)


# ## Exclude outliers

# In[ ]:


likes_for_plot = monthly_likes_stats[monthly_likes_stats.index.month != 12].reset_index()
for y in monthly_likes_stats.columns:
    sns.relplot(data=likes_for_plot,
                kind="line", x="liked_at", y=y, aspect=2)


# ---
# # 3. Merge Likes and Items

# In[ ]:


likes_items = pd.merge(likes.reset_index(), items.reset_index(), on="item_id", how='left')
display(likes_items.head())


# In[ ]:


get_ipython().run_cell_magic('time', '', 'likes_items.to_csv("likes_items.tsv", sep="\\t")')


# ---
# # 4. Cumulative Sum of Likes

# In[ ]:


grouped_likes = (
    likes_items
    .groupby([
        pd.Grouper(key="liked_at", freq="YS"),
        pd.Grouper(key="created_at", freq="YS"),
    ])
    .liker.count()
    .rename("likes")
)

likes_cumsum = (
    grouped_likes
    .groupby("created_at")
    .apply(lambda x: x.cumsum() / yearly_items_stats["items"])
    .rename("likes / item")
)


# In[ ]:


g = sns.relplot(
    data=likes_cumsum.reset_index(),
    kind="line", 
    x="liked_at", y="likes / item", hue="created_at", 
)
for t, i in zip(g._legend.texts[1:], yearly_items_stats.index):
    t.set_text(i.year)


# ---
# # 5. Likes Count Matrix by CreatedDate and LikedDate

# ## Likes Count

# In[ ]:


get_ipython().run_cell_magic('time', '', '# likes_df = likes_items.groupby([\n#     pd.Grouper(key="liked_at", freq="YS"),\n#     pd.Grouper(key="created_at", freq="YS"),\n# ]).liker.count().rename("likes").reset_index()\n\nlikes_matrix = grouped_likes.reset_index().pivot_table(\n    index="liked_at",\n    columns="created_at",\n    values="likes",\n    aggfunc="sum",\n#     margins=True,\n)\ndisplay(likes_matrix)')


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    likes_matrix,
    annot=True, fmt=".0f",
    xticklabels=yearly_likes_stats.index.year,
    yticklabels=yearly_likes_stats.index.year,
)


# ## Likes Count / Yearly Items

# In[ ]:


yearly_items_stats["likes / item"].reset_index().style.hide_index().format({
    "created_at": "{:%Y}",
    "likes / item": "{:.1f}",
})


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    likes_matrix / yearly_items_stats["items"],
    annot=True, fmt=".1f",
    xticklabels=yearly_likes_stats.index.year,
    yticklabels=yearly_likes_stats.index.year,
)


# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    likes_matrix / yearly_items_stats["likes"],
    annot=True, fmt=".1%",
    xticklabels=yearly_likes_stats.index.year,
    yticklabels=yearly_likes_stats.index.year,
)


# ## Likes Count / Yearly Likes

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    (likes_matrix.T / yearly_likes_stats.likes).T,
    annot=True, fmt=".1%",
    xticklabels=yearly_likes_stats.index.year,
    yticklabels=yearly_likes_stats.index.year,
)


# ---
# # 6. Period from Created-date to Liked-date

# In[ ]:


likes_items["time_delta"] = likes_items.liked_at - likes_items.created_at
likes_items[likes_items.time_delta <= "1minute"].head()


# ## Basic Statistics

# In[ ]:


likes_items.time_delta.describe()


# ## Distribution

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sns.distplot(likes_items[likes_items.time_delta.notnull()].time_delta.dt.days)')


# In[ ]:


def cut_time_delta(df):
    return pd.cut(
        df.time_delta.dt.days,
        bins=[0, 1, 7, 30, 99999], 
        labels=["1day", "1week", "1month", "1month over"],
#         df.time_delta,
#         bins=[pd.Timedelta("0minute"), 
#               pd.Timedelta("12hours"), 
#               pd.Timedelta("4days"), 
#               pd.Timedelta("30days"), 
#               pd.Timedelta("99999days"), 
#              ], 
#         labels=["12hours", "daily trend", "1month", "1month over"],
        right=False,
    ).value_counts().sort_index()

cut_time_delta(likes_items) / len(likes_items)


# In[ ]:


period_stats = (
    likes_items
    .groupby(pd.Grouper(key="liked_at", freq="YS"))
    .apply(lambda x: cut_time_delta(x) / len(x))
    .stack()
    .rename("value")
)
display(period_stats.unstack("time_delta"))


# In[ ]:


sns.relplot(
    data=period_stats.reset_index(),
    kind="line",
    x="liked_at",
    y="value",
    hue="time_delta",
)


# In[ ]:


stack_bar_plot(
    period_stats,
    x="liked_at",
    hue="time_delta",
    xticklabels=list(yearly_items_stats.index.year),
)


# In[ ]:


sns.heatmap(
    period_stats.unstack("time_delta"),
    annot=True, fmt=".1%",
    yticklabels=yearly_likes_stats.index.year,
)


# ## Distribution of Likes for Trend Itmes

# In[ ]:


trend_likes = likes_items[
    ("1days" <= likes_items.time_delta) & 
    (likes_items.time_delta < "7days")
]


# In[ ]:


trend_items = trend_likes.groupby([
    pd.Grouper(key="created_at", freq="YS"),
    "item_id"
]).liker.count().rename("likes")

trend_likes_dist = (
    trend_items.groupby([
        "created_at", 
        pd.cut(trend_items, bins=[1, 10, 100, 99999], right=False)
    ])
    .sum()
    .groupby("created_at")
    .apply(lambda x: x / x.sum())
    .rename("value")
)
# trend_likes_dist.rename(columns=str)
trend_likes_dist.unstack("likes")


# In[ ]:


sns.relplot(
    data=trend_likes_dist.reset_index(),
    kind="line",
    x="created_at",
    y="value",
    hue="likes",
)


# In[ ]:


stack_bar_plot(
    trend_likes_dist,
    x="created_at",
    hue="likes",
    xticklabels=list(yearly_items_stats.index.year),
)


# In[ ]:


sns.heatmap(
    trend_likes_dist.unstack("likes"),
    annot=True, fmt=".1%",
    yticklabels=yearly_likes_stats.index.year,
)

