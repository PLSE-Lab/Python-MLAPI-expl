#!/usr/bin/env python
# coding: utf-8

# This will be an attempt to find some structure in feature importance, primarily by looking for a relationship between `whiskey-copper-turtle-magic` and which features are useful for identifying `target`

# In[ ]:


import os, re, gc, warnings
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as wd
from tqdm import tqdm_notebook
warnings.filterwarnings("ignore")
gc.collect()


# In[ ]:


DATA_DIR = '../input/'
FILES={}
for fn in os.listdir(DATA_DIR):
    FILES[ re.search( r'[^_\.]+', fn).group() ] = DATA_DIR + fn

CAT_COL='wheezy-copper-turtle-magic'    

train = pd.read_csv(FILES['train'],index_col='id')
# test = pd.read_csv(FILES['test'],index_col='id')
CATS = sorted(train[CAT_COL].unique())


# In[ ]:


feats = train.columns.drop([CAT_COL,'target'])
wd_matrix = pd.DataFrame(index=range(512), columns=feats)


# [Wasserstein Distance on wikipedia](https://en.wikipedia.org/wiki/Wasserstein_metric)
# I could have used a different, faster, metric but I did not want to miss any useful signal until I knew what I would be throwing away.

# In[ ]:


for wctm in tqdm_notebook(CATS):
    for f in feats:
        wd_matrix.loc[wctm][f] = wd(train[(train[CAT_COL]==wctm) & (train['target']==0) ][f], train[(train[CAT_COL]==wctm) & (train['target']==1) ][f])
        


# In[ ]:


wd_matrix = wd_matrix[ wd_matrix.columns ].astype('float')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set(style="whitegrid")
cmap = sns.color_palette('YlGn',10)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(wd_matrix, xticklabels=8, yticklabels=16, cmap=cmap ,cbar_kws={"shrink": .5} )


# I had hoped to see some clear pattern that would allow unscrambling which columns were significant if given only the value for `whiskey-copper-turtle-magic`. No feature seems to share the same sequence of high WD scores with any other feature (the vertical columns are not duplicated, and there isn't a reptition within them that I can see).

# In[ ]:


# viewing by ranked feature importance within each of 512 models doesn't reveal much
# wd_matrix.rank(axis=1,ascending=True).astype('int')


# In[ ]:


sns.set(style="whitegrid")
cmap = sns.color_palette('colorblind',10)
f, ax = plt.subplots(figsize=(20, 40))
f.add_subplot(2,1,1)
sns.heatmap(wd_matrix.iloc[:,0:16], vmin=.6, vmax=2, cmap=cmap ,cbar_kws={"shrink": .5} )
f.add_subplot(2,1,2)
sns.heatmap(wd_matrix.filter(like='important'), vmin=.6, vmax=2, xticklabels=1, yticklabels=16, cmap=cmap ,cbar_kws={"shrink": .5} )


# Above are heatmaps for 2 different subsections of the features. The colormap is non-sequential to help identify any pattern visually.
# The 2 subsections are:
# * 16 sequential columns of the data as given in `train.csv`
# * 16 columns that have `importance` in their name
# 
# It's easier to pay attention when there is less information shown, but I just see noise.

# In[ ]:


wd_matrix.describe()


# Every feature varies from useless (WD=.05) to excellent (WD=2)
# They all take turns being significant to the same degree, but any predictable rotation eludes me.

# In[ ]:


wd_matrix['WCTM'] = wd_matrix.index


# In[ ]:


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
df = pd.melt(wd_matrix, 
             id_vars=['WCTM'], 
             value_vars=feats, 
             var_name='feature', value_name='wDistance')

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(256, rot=-.1, light=.6)
g = sns.FacetGrid(df, row="feature", hue="feature", aspect=3, height=3, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "wDistance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.05)
g.map(sns.kdeplot, "wDistance", clip_on=False, color="w", lw=2, bw=.05)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color='black', size='large',
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "wDistance")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


# This is a ridgeline plot for each of the 255 features, showing the distribution of wasserstein distances of that feature's value distributions for `target==0` or `target==1`, among the 512 values of `whiskey-copper-turtle-magic`.

# In[ ]:


df.sort_values('wDistance',ascending=False)[40*512:50*512:250]


# Convention in this competition has been to work on modelling using just the best 40-50 features for any given value of `whiskey-copper-turtle-magic`. That would be a dividing line of about 0.2 in terms of Wasserstein distance

# In[ ]:


df.sort_values('wDistance',ascending=False).hist()
df.sort_values('wDistance',ascending=False)[:45*512].hist()
df.sort_values('wDistance',ascending=False)[:10*512].hist()


# Here we're getting hints at how the data was generated. Most of the data is more noise than signal, but once you drop 90% of it there is a scale-free distribution of wasserstein distances. The amount of these better features is fairly even across instances of `whiskey-copper-turtle-magic`.

# In[ ]:


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


df2=df[df['wDistance']>1]

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(256, rot=-.1, light=.6)
g = sns.FacetGrid(df2, row="feature", hue="feature", aspect=5, height=2, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "wDistance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.05)
g.map(sns.kdeplot, "wDistance", clip_on=False, color="w", lw=2, bw=.05)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color='black', size='large',
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "wDistance")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)


# One more pretty ridgeline plot, this time just focusing on the wasserstein distances when that feature is in the top 10 strongest features for a value of `whiskey-copper-turtle-magic`.

# In[ ]:




