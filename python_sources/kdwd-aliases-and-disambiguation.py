#!/usr/bin/env python
# coding: utf-8

# # Kensho Derived Wikimedia Dataset - Entity Aliases and Disambiguation Candidates with Anchor Link Statistics
# 
# Let's see how we can use the link annotated text of the KDWD to calculate aliases for entities in Wikipedia as well as candidate disambiguation targets from text input.  The first problem (entity aliases) asks, "What text spans are typically used to refer to a Wikipedia page?".  The second (disambiguation candidates) asks, "What entities are typically refered to by a text span?"    

# In[ ]:


from collections import Counter
import json
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# In[ ]:


sns.set()
sns.set_context('talk')


# Lets check the input directory to see what files we have access to.

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# All of the KDWD files have one "thing" per line. We'll hard code the number of lines in the files we're going to use so we can have nice progress bars when streaming through them.  We will also define a simple text normalizer function. 

# In[ ]:


MIN_VIEWS = 5
MIN_ANCHOR_TARGET_COUNT = 2
NUM_KLAT_LINES = 5_343_564
NUM_PAGE_LINES = 5_362_174
kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")

def text_normalizer(text):                              
    """Return text after stripping external whitespace and lower casing."""   
    return text.strip().lower()


# # Collect (anchor text, target page) statistics from link annotated text
# Here we create a class for iterating over lines in the link annotated text file and count occurrences of (anchor text, target page) tuples.  The [anchor text](https://en.wikipedia.org/wiki/Anchor_text) of a link is the visible, clickable text that is usually highlighted blue.  The target page is the page you land on after clicking the link.  

# In[ ]:


class KdwdLinkAnnotatedText:
    def __init__(self, file_path):
        self.file_path = file_path
    def __iter__(self):
        with open(self.file_path) as fp:
            for line in fp:
                yield json.loads(line)


# In[ ]:


file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")
klat = KdwdLinkAnnotatedText(file_path)


# In[ ]:


anchor_target_counts = Counter()
for page in tqdm(
    klat, 
    total=NUM_KLAT_LINES, 
    desc='calculating anchor-target counts'
):
    for section in page['sections']:
        spans = [
            (offset, offset + length) for offset, length in 
            zip(section['link_offsets'], section['link_lengths'])]
        anchor_texts = [section['text'][ii:ff] for ii,ff in spans]
        keys = [
            (anchor_text, target_page_id) for anchor_text, target_page_id in 
            zip(anchor_texts, section['target_page_ids'])]
        anchor_target_counts.update(keys)


# In[ ]:


at_count_df = pd.DataFrame([
    (row[0][0], row[0][1], row[1]) for row in anchor_target_counts.most_common()],
    columns=['anchor_text', 'target_page_id', 'anchor_target_count'])


# In[ ]:


at_count_df


# # Normalize and re-group anchor text
# Here we apply a simple text normalization function to the anchor texts and merge those that normalize to the same string.  For example "Chicago " and "chicago" will be the same after normalization. 

# In[ ]:


at_count_df["normalized_anchor_text"] = at_count_df["anchor_text"].apply(text_normalizer)
at_count_df = at_count_df.loc[at_count_df['normalized_anchor_text'].str.len() > 0, :]


# In[ ]:


at_count_df = (                                               
    at_count_df.                                              
    groupby(["normalized_anchor_text", "target_page_id"])["anchor_target_count"].   
    sum().                                                               
    to_frame("anchor_target_count").
    sort_values('anchor_target_count', ascending=False).
    reset_index()                                                        
)


# In[ ]:


at_count_df


# Note that normalization and re-grouping took us from 12,128,173 to 11,405,840 rows

# # Anchor target count distribution
# Lets examine the distribution of (anchor text, target page) tuples.

# ## Zipf style plot 
# 
# Below we plot tuple frequency (the number of times a tuple occurs) vs rank (rank order when sorted by count).  This is a [Zipfian](https://en.wikipedia.org/wiki/Zipf%27s_law) plot but we use the (anchor text, target page) tuples as opppsed to words.  Note that in this plot the most frequently occuring tuples are on the left.  

# In[ ]:


num_rows = at_count_df.shape[0]
ii_rows_logs = np.linspace(0, np.log10(num_rows-1), 30)
ii_rows = [int(el) for el in 10**ii_rows_logs]
rows = at_count_df.iloc[ii_rows, :]
indexs = np.log10(rows.index.values)
counts = np.log10(rows['anchor_target_count'].values + 1)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(indexs, counts)
ax.set_xlabel('log10 (anchor text, target page) rank')
ax.set_ylabel('log10 count')
ax.set_title('Zipf style plot for (anchor text, target page) tuples');


# ## Distribution of Counts
# Below we plot a straightforward histogram of the (anchor text, target page) counts.  Note that in this plot the most frequently occuring tuples are on the right. 

# In[ ]:


atc = at_count_df['anchor_target_count'].values
logatc = np.log10(atc)

fig, ax = plt.subplots(figsize=(12,8))
patches = ax.hist(logatc, log=True, bins=30)
ax.set_xlabel('log10 (anchor text, target page) count')
ax.set_ylabel('log10 count')
ax.set_title('Distribution of (anchor text, target page) Counts');


# # Load page data
# Now that we have (anchor text, target page) tuple counts, lets load in page metadata about the target pages. 

# In[ ]:


file_path = os.path.join(kdwd_path, "page.csv")
page_df = pd.read_csv(
    file_path, 
    keep_default_na=False) # dont read the page title "NA" as a null


# In[ ]:


page_df


# # Merge anchor-target counts with page metadata

# In[ ]:


at_count_df = pd.merge(
    at_count_df,
    page_df,
    how="inner",
    left_on="target_page_id",
    right_on="page_id")


# In[ ]:


at_count_df = at_count_df.rename(columns={
    'title': 'target_page_title',
    'item_id': 'target_item_id',
    'views': 'target_page_views'})


# In[ ]:


at_count_df = at_count_df[[
    "normalized_anchor_text",
    "target_page_id",
    "target_item_id",
    "target_page_title",
    "target_page_views",
    "anchor_target_count"]]


# In[ ]:


at_count_df


# # Filter on anchor-target count and page views

# In[ ]:


bool_mask_1 = at_count_df["anchor_target_count"] >= MIN_ANCHOR_TARGET_COUNT
bool_mask_2 = at_count_df["target_page_views"] >= MIN_VIEWS
bool_mask = bool_mask_1 & bool_mask_2
at_count_df = at_count_df.loc[bool_mask, :].copy()


# # Calculate conditional probabilities 
#  * P(anchor text | target page)
#  * P(target page | anchor text)

# In[ ]:


norm = at_count_df.groupby("target_page_id")["anchor_target_count"].transform("sum")
at_count_df["p_anchor_given_target"] = at_count_df["anchor_target_count"] / norm
norm = at_count_df.groupby("normalized_anchor_text")["anchor_target_count"].transform("sum")
at_count_df["p_target_given_anchor"] = at_count_df["anchor_target_count"] / norm


# In[ ]:


at_count_df


# In[ ]:


pagt = at_count_df['p_anchor_given_target'].values
ptga = at_count_df['p_target_given_anchor'].values

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,6))
axes[0].hist(pagt, log=True, bins=41)
axes[0].set_xlabel('P(anchor|target)')
axes[0].set_ylabel('log10 count')
axes[0].set_ylim(1e3, 1e7)


axes[1].hist(ptga, log=True, bins=41)
axes[1].set_xlabel('P(target|anchor)')
axes[1].set_ylim(1e3, 1e7)

fig.suptitle('Distribution of conditional probabilities');


# # Create anchor-target class to provide entity alias and text disambiguation functionality 

# In[ ]:


class AnchorTargetStats:
    
    def __init__(
        self,
        at_count_df,
        text_normalizer,
    ):
        """Anchor-target statistics 
        
        Args:
            at_count_df: (normalized_anchor_text, target_page) counts and metadata
            text_normalizer: text cleaning function for anchor texts
        """
        self._at_count_df = at_count_df
        self.text_normalizer = text_normalizer

    def get_aliases_from_page_id(self, page_id):
        """Return anchor strings used to refer to entity"""
        bool_mask = self._at_count_df['target_page_id'] == page_id
        return (
            self._at_count_df.
            loc[bool_mask].copy().
            sort_values('p_anchor_given_target', ascending=False)
        )
    
    def get_disambiguation_candidates_from_text(self, text):
        """Return candidate entities for input text"""
        normalized_text = self.text_normalizer(text)
        bool_mask = self._at_count_df['normalized_anchor_text'] == normalized_text
        return (
            self._at_count_df.
            loc[bool_mask].copy().
            sort_values('p_target_given_anchor', ascending=False)
        )


# In[ ]:


anchor_target_stats = AnchorTargetStats(at_count_df, text_normalizer)


# # Entity alias examples

# In[ ]:


def plot_pagt(aliases, page_title, width=0.7):
    """Plot P(anchor|target) for a specific page"""
    labels = aliases['normalized_anchor_text'].values
    probas = aliases['p_anchor_given_target'].values
    yy = np.arange(len(labels)) 

    figsize = (16, len(probas) * 16/25)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    rects = ax.barh(yy, probas, width)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlabel('P(anchor text|target page)')
    ax.set_ylabel('normalized anchor text')
    
    ax = axes[1]
    log_probas = np.log10(1 + 100 * probas)
    rects = ax.barh(yy, log_probas, width)
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_xlabel('log10[1 + 100 * P(anchor text|target page)]')

    fig.suptitle(f'anchor texts for target_page="{page_title}"');


# In[ ]:


page_id = 18717338   # https://en.wikipedia.org/wiki/United_States_dollar
aliases = anchor_target_stats.get_aliases_from_page_id(page_id)
aliases.head(25)


# In[ ]:


plot_pagt(aliases.head(25), "United States dollar")


# In[ ]:


page_id = 651269   # https://en.wikipedia.org/wiki/S&P_Global
aliases = anchor_target_stats.get_aliases_from_page_id(page_id)


# In[ ]:


plot_pagt(aliases, "S&P Global")


# In[ ]:


page_id = 32544339   # https://en.wikipedia.org/wiki/Hydraulic_fracturing
aliases = anchor_target_stats.get_aliases_from_page_id(page_id)


# In[ ]:


plot_pagt(aliases, "Hydraulic fracturing")


# In[ ]:


page_id = 58900   # https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle
aliases = anchor_target_stats.get_aliases_from_page_id(page_id)


# In[ ]:


plot_pagt(aliases.head(25), "Unmanned aerial vehicle")


# In[ ]:


page_id = 25226624   # https://en.wikipedia.org/wiki/Patient_Protection_and_Affordable_Care_Act
aliases = anchor_target_stats.get_aliases_from_page_id(page_id)


# In[ ]:


plot_pagt(aliases.head(25), "Patient Protection and Affordable Care Act")


# # Disambiguation candidate examples

# In[ ]:


def plot_ptga(disambigs, anchor_text, width=0.7):
    """Plot P(target|anchor) for a specific page"""
    labels = disambigs['target_page_title'].values
    probas = disambigs['p_target_given_anchor'].values
    yy = np.arange(len(labels)) 

    figsize = (16, len(probas) * 16/25)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    rects = ax.barh(yy, probas, width)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels)
    ax.set_xlabel('P(target page|anchor text)')
    ax.set_ylabel('page title')
   
    ax = axes[1]
    log_probas = np.log10(1 + 100 * probas)
    rects = ax.barh(yy, log_probas, width)
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_xlabel('log10[1 + 100 * P(target page|anchor text)]')
    
    fig.suptitle(f'page titles for anchor_text="{anchor_text}"');


# In[ ]:


text = "chicago"
disambigs = anchor_target_stats.get_disambiguation_candidates_from_text(text)
disambigs.head(25)


# In[ ]:


plot_ptga(disambigs.head(25), text)


# In[ ]:


text = "point"
disambigs = anchor_target_stats.get_disambiguation_candidates_from_text(text)


# In[ ]:


plot_ptga(disambigs.head(25), text)


# In[ ]:


text = 'pound'
disambigs = anchor_target_stats.get_disambiguation_candidates_from_text(text)


# In[ ]:


plot_ptga(disambigs.head(25), text)


# In[ ]:


text = 'abc'
disambigs = anchor_target_stats.get_disambiguation_candidates_from_text(text)


# In[ ]:


plot_ptga(disambigs.head(25), text)


# In[ ]:


text = 'aca'
disambigs = anchor_target_stats.get_disambiguation_candidates_from_text(text)


# In[ ]:


plot_ptga(disambigs.head(25), text)

