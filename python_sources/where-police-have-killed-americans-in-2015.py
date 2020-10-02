#!/usr/bin/env python
# coding: utf-8

# # Where Police Have Killed Americans In 2015
# _By Nick Brooks, June 2020_
# 
# **Resources:** <br>
# - https://www.kaggle.com/fivethirtyeight/fivethirtyeight-police-killings-dataset
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import time
import math
import itertools
from wordcloud import WordCloud
from itertools import combinations

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SIA = SentimentIntensityAnalyzer()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

sns.set_style("whitegrid")
notebookstart = time.time()
pd.options.display.max_colwidth = 500


# In[ ]:


def meta_text_features(df, col):
    df[col] = df[col].astype(str)
    df[col + '_num_words'] = df[col].apply(lambda comment: len(comment.split())) # Count number of Words
    df[col + '_num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df[col + '_words_vs_unique'] = df[col+'_num_unique_words'] / df[col+'_num_words'] * 100 # Count Unique Words
    if col == "text":
        df[col+"_vader_Compound"]= df[col].apply(lambda x:SIA.polarity_scores(x)['compound'])

    return df

def big_count_plotter(plot_df, plt_set, columns, figsize, hue = None,
                      custom_palette = sns.color_palette("Paired", 15), top_n = 15):
    """
    Iteratively Plot all categorical columns
    Has category pre-processing - remove whitespace, lower, title, and takes first 30 characters.
    """
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            c_col = plt_set[i]
            plt_tmp = plot_df.loc[plot_df[c_col].notnull(),c_col]                .astype(str).str.lower().str.strip()                .str.title().apply(lambda x: x[:30])
            plot_order = plt_tmp.value_counts().index[:top_n]
            if hue:
                sns.countplot(y = plt_tmp, ax = ax, hue = hue, order = plot_order, palette = custom_palette)
            else:
                sns.countplot(y = plt_tmp, ax = ax, order = plot_order, palette = custom_palette)
            ax.set_title("{} - {} Missing".format(c_col.title(), plot_df[c_col].isnull().sum()))
            ax.set_ylabel("{} Categories".format(c_col.title()))
            ax.set_xlabel("Count")
        else:
            ax.axis('off')

    plt.tight_layout(pad=1)
    
    
def big_boxplotter(plot_df, plt_set, columns, figsize, hue = None, plottype='kde',
                   custom_palette = sns.color_palette("Dark2", 15), quantile = .99):
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    palette = itertools.cycle(custom_palette)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            cont_col = plt_set[i]
            if hue:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) & 
                                          (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      [cont_col, hue]]
                if plottype == 'box':
                    sns.boxplot(data=plt_tmp, x=cont_col, y=hue, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    for h in plt_tmp.dropna()[hue].value_counts()[:5].index:
                        c = next(palette)
                        sns.distplot(plt_tmp.loc[plt_tmp[hue] == h,cont_col], bins=10, kde=True, ax=ax,
                                     kde_kws={"color": c, "lw": 2, "label":h}, color=c)
                    ax.set_ylabel("Density Occurence")
            else:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) &
                                          (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      cont_col].astype(float)
                if plottype == 'box':
                    sns.boxplot(plt_tmp, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    sns.distplot(plt_tmp, bins=10, kde=True, ax=ax,
                        kde_kws={"color": "k", "lw": 2}, color=next(palette))
                    ax.set_ylabel("Density Occurence")
            ax.set_title("{} - {:.0f} Missing - {:.2f} Max".format(cont_col.title(),
                plot_df[cont_col].isnull().sum(), plot_df[cont_col].max()))
            ax.set_xlabel("Value")
            
        else:
            ax.axis('off')

    plt.tight_layout(pad=1)
    
def big_word_cloud(plot_df, plt_set, columns, figsize, cmap = "plasma"):
    """
    Iteratively Plot WordClouds
    """
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            str_col = plt_set[i]
            string = " ".join(plot_df.loc[plot_df[str_col].notnull(),str_col]                              .astype(str).str.lower().str.replace("none", "").str.title())
            string += 'EMPTY'
            ax = plt.subplot(rows, 2, i+1)
            plot_cloud(string, ax, title = "{} - {:.0f} Missing".format(
                str_col.title(), plot_df[str_col].isnull().sum()), cmap = cmap)
        else:
            ax.axis('off')
    plt.tight_layout(pad=0)
    
def plot_cloud(string, ax, title = "WordCloud", cmap = "plasma"):
    wordcloud = WordCloud(width=800, height=500,
                          collocations=True,
                          background_color="black",
                          max_words = 100,
                          colormap=cmap
                ).generate(string)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title,  fontsize=18)
    ax.axis('off')
    
    
def rank_correlations(df, figsize=(12,20), n_charts = 18, polyorder = 2, custom_palette = sns.color_palette("Paired", 5)):
    # Rank Correlations
    palette = itertools.cycle(custom_palette)
    continuous_rankedcorr = (df
                             .corr()
                             .unstack()
                             .drop_duplicates().reset_index())
    continuous_rankedcorr.columns = ["f1","f2","Correlation Coefficient"]
    continuous_rankedcorr['abs_cor'] = abs(continuous_rankedcorr["Correlation Coefficient"])
    continuous_rankedcorr.sort_values(by='abs_cor', ascending=False, inplace=True)

    # Plot Top Correlations
    top_corr = [(x,y,cor) for x,y,cor in list(continuous_rankedcorr.iloc[:, :3].values) if x != y]
    f, axes = plt.subplots(int(n_charts/3),3, figsize=figsize, sharex=False, sharey=False)
    row = 0
    col = 0
    for (x,y, cor) in top_corr[:n_charts]:
        if col == 3:
            col = 0
            row += 1
        g = sns.regplot(x=x, y=y, data=df, order=polyorder, ax = axes[row,col], color=next(palette))
        axes[row,col].set_title('{} and {}'.format(x, y))
        axes[row,col].text(0.18, 0.93,"Cor Coef: {:.2f}".format(cor),
                           ha='center', va='center', transform=axes[row,col].transAxes)
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
    
    
# Data Exploration
def custom_describe(df, value_count_n = 5):
    """
    Custom Describe Function - More Tailored to categorical type variables..
    """
    unique_count = []
    for x in df.columns:
        unique_values_count = df[x].nunique()
        value_count = df[x].value_counts().iloc[:5]

        value_count_list = []
        value_count_string = []
        
        for vc_i in range(0,value_count_n):
            value_count_string += ["ValCount {}".format(vc_i+1),
                                   "Occ"]
            if vc_i <= unique_values_count - 1:
                value_count_list.append(value_count.index[vc_i])
                value_count_list.append(value_count.iloc[vc_i])
            else:
                value_count_list.append(np.nan)
                value_count_list.append(np.nan)
        
        unique_count.append([x,
                             unique_values_count,
                             df[x].isnull().sum(),
                             df[x].dtypes] + value_count_list)
        
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count,
            columns=["Column","Unique","Missing","dtype"
                    ] + value_count_string
                       ).set_index("Column")

print("Helper Functions Ready")


# In[ ]:


time_cols = [
    'month',
    'day',
    'year',
]

categorical_cols = [
 'gender',
 'raceethnicity',
 'city',
 'state',
 'cause',
 'armed',
#  'county_bucket',
#  'nat_bucket',
#  'state_fp',
#  'county_fp',
#  'tract_ce',
#  'geo_id',
#  'county_id',
]


continuous_cols = [
    'age',
    'pop',
    'share_white',
    'share_black',
    'share_hispanic',
    'p_income',
    'h_income',
    'county_income',
    'comp_income',
    'pov',
    'urate',
    'college'
]

text_cols = [
    'name',
    'namelsad',
    'lawenforcementagency',
    'streetaddress',
]

geo_cols =  [
    'latitude',
    'longitude'

]

df = pd.read_csv("/kaggle/input/fivethirtyeight-police-killings-dataset/police_killings.csv", encoding = "ISO-8859-1")
print("DF Shape: {} Rows, {} Columns".format(*df.shape))

# Data Cleaning
df = df.assign(
    age = df.age.astype(str).replace('Unknown', np.nan),
    share_white = df.share_white.replace('-', np.nan),
    share_black = df.share_black.replace('-', np.nan),
    share_hispanic = df.share_hispanic.replace('-', np.nan),
    p_income = df.p_income.replace('-', np.nan),
    pov = df.pov.replace('-', np.nan)
)
for col in continuous_cols:
    df[col] = pd.to_numeric(df[col])


# In[ ]:


display(df.sample(5))


# In[ ]:


print("Categorical Variables")
display(custom_describe(df[categorical_cols+text_cols]))
print("Continuous Variables")
display(df[continuous_cols].describe().T)


# In[ ]:


big_word_cloud(df,
               plt_set = ['name'],
               columns = 1,
               cmap='Spectral',
               figsize = [15,15])
plt.show()


# In[ ]:


big_count_plotter(plot_df = df,
                  plt_set = categorical_cols,
                  columns = 2,
                  figsize = [14,16],
                  custom_palette = sns.color_palette("Paired", 15))


# In[ ]:


big_boxplotter(plot_df = df,
               plt_set = continuous_cols,
               hue = None,
               columns = 2,
               figsize = [12,18],
               quantile = .98)


# In[ ]:


# Plot Correlation Matrix
f, ax = plt.subplots(figsize=[12,8])
ax = sns.heatmap(df[continuous_cols].corr(), 
                 annot=True, fmt=".2f",
                 vmin=-1, vmax=1,
                 cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("Continuous Variable Correlation Matrix")
plt.show()


# In[ ]:


rank_correlations(df = df.loc[:,continuous_cols])


# In[ ]:


big_boxplotter(plot_df = df,
               plt_set = continuous_cols,
               hue = 'gender',
               plottype= 'box',
               columns = 2,
               figsize = [10,14],
               quantile = .98)


# In[ ]:


big_boxplotter(plot_df = df,
               plt_set = continuous_cols,
               hue = 'raceethnicity',
               plottype= 'box',
               columns = 2,
               figsize = [16,20],
               quantile = .98)


# In[ ]:


def big_count_pivot_tables(df, heatmap_vars, index_variable, top_n_factors):
    aggfunc='count'
    f, axes = plt.subplots(len(heatmap_vars), 3, figsize = [16,4*len(heatmap_vars)], sharex=False, sharey=False)
    for row,column_var in enumerate(heatmap_vars):
        tmp_plot_pd = df.loc[(df[column_var].isin(df[column_var].value_counts()[:top_n_factors].index)) &
                              (df[index_variable].isin(df[index_variable].value_counts()[:top_n_factors].index)) ,:].copy()
        tmp_plot_pd['placeholder'] = 1

        pivot_pd = pd.pivot_table(tmp_plot_pd, index=index_variable, columns=column_var,
                                  values='placeholder', aggfunc=aggfunc).fillna(0)
        pivot_pd_row_norm = pivot_pd.div(pivot_pd.sum(axis=1), axis=0).mul(100).round(0)
        pivot_pd_col_norm = pivot_pd.div(pivot_pd.sum(axis=0), axis=1).mul(100).round(0)

        # Plot
        sns.heatmap(pivot_pd, annot=True, fmt="g", linewidths=.5,
                    linecolor='black', cbar=False, cmap="Blues", ax = axes[row,0])

        sns.heatmap(pivot_pd_row_norm, annot=True, fmt="g", linewidths=.5,
                    linecolor='black', cbar=False, cmap="Reds", ax = axes[row,1])

        sns.heatmap(pivot_pd_col_norm, annot=True, fmt="g", linewidths=.5,
                    linecolor='black', cbar=False, cmap="Purples", ax = axes[row,2])  

        axes[row,0].set_title("{}: Total Police Killins Cases".format(column_var))
        axes[row,1].set_title("Row Normalised Pivot Table")
        axes[row,2].set_title("Columns Normalised Pivot Table")

    plt.tight_layout(pad=1)
    plt.show()
    
def big_agg_pivot_tables(df, heatmap_vars, top_n_factors, value_var, aggfunc, figsize, color, columns=2):
    all_combinations = [x for x in combinations(heatmap_vars, 2)]
    rows = math.ceil(len(all_combinations)/columns)
    
    f, ax = plt.subplots(rows, columns, figsize = figsize, sharex=False, sharey=False)
    for i,(index_var, column_var) in enumerate(all_combinations):
        ax = plt.subplot(rows, columns, i+1)            
        tmp_plot_pd = df.loc[(df[column_var].isin(df[column_var].value_counts()[:top_n_factors].index)) &
                              (df[index_var].isin(df[index_var].value_counts()[:top_n_factors].index)),
                             [index_var,column_var,value_var]].copy()

        pivot_pd = pd.pivot_table(tmp_plot_pd, index=index_var, columns=column_var,
                                  values=value_var, aggfunc=aggfunc)
        # Plot
        sns.heatmap(pivot_pd, annot=True, fmt="g", linewidths=.5,
                    linecolor='black', cbar=False, cmap=color, ax = ax)

        ax.set_title("{} {} Pivot Table on\nX: {}, Y: {}".format(
            value_var.title(), aggfunc,column_var.title(), index_var.title()))

    plt.tight_layout(pad=1)
    plt.show()


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed']
index_variable = 'raceethnicity'
top_n_factors = 5

big_count_pivot_tables(df=df, heatmap_vars=heatmap_vars, index_variable=index_variable, top_n_factors=top_n_factors)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='OrRd', value_var='age',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='Blues', value_var='college',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='Purples', value_var='county_income',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='Greens', value_var='comp_income',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='mako', value_var='county_bucket',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='Purples', value_var='nat_bucket',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='GnBu', value_var='urate',
                     figsize=[14,20], columns=2)


# In[ ]:


heatmap_vars = ['gender','city','state','cause','armed','raceethnicity']
top_n_factors = 5
big_agg_pivot_tables(df=df, heatmap_vars=heatmap_vars,
                     top_n_factors=top_n_factors, aggfunc='mean',
                     color='Reds', value_var='pov',
                     figsize=[14,20], columns=2)


# In[ ]:


print("Script Complete - Runtime: {:.2f} Minutes".format((time.time() - notebookstart) / 60))


# In[ ]:




