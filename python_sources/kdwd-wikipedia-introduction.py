#!/usr/bin/env python
# coding: utf-8

# # Kensho Derived Wikimedia Dataset - Wikipedia Introduction
# 
# This notebook will introduce you to the Wikipedia Sample of the Kensho Derived Wikimedia Dataset (KDWD).  We'll explore the files and make some "getting to know you" plots.  Lets start off by importing some packages.

# In[ ]:


from collections import Counter
import csv
import json
import os
import string

import numpy as np
import pandas as pd
from tqdm import tqdm


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()
sns.set_context('talk')


# Lets check the input directory to see what files we have access to. 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# All of the KDWD files have one "thing" per line.  We'll hard code the number of lines in the files we're going to use so we can have nice progress bars when streaming through them.  **If you want to run small experiments, set `MAX_PAGES` to a small number like 100,000**.

# In[ ]:


NUM_KLAT_LINES = 5_343_564  # link_annotated_text.jsonl
NUM_PAGE_LINES = 5_362_174  # page.csv
MAX_PAGES = max(NUM_KLAT_LINES, NUM_PAGE_LINES) # change this to a smaller integer for faster runs
kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")


# # Page Metadata: Ingest
# 
# Lets examine the Wikipedia sample starting with page metadata in `page.csv`.  After this, we'll move on to the `link_annotated_text.jsonl` file. 

# In[ ]:


page_df = pd.read_csv(
    os.path.join(kdwd_path, "page.csv"),
    keep_default_na=False, # dont read the page title "NaN" as a null value
    nrows=MAX_PAGES, # if we want to experient with subsets
) 
page_df


# We store pages in ascending `page_id` order.  The `page_id` is the primary Wikipedia identifier for a page and the `item_id` is the primary Wikidata identifier for the associated Wikidata page.

# # Page Metadata: URL Construction
# We can construct Wikipedia and Wikidata urls from the metadata if we like.

# In[ ]:


def wikipedia_url_from_title(title):
    return 'https://en.wikipedia.org/wiki/{}'.format(title.replace(' ', '_'))

def wikipedia_url_from_page_id(page_id):
    return 'https://en.wikipedia.org/?curid={}'.format(page_id)

def wikidata_url_from_item_id(item_id):
    return 'https://www.wikidata.org/entity/Q{}'.format(item_id)


# In[ ]:


iloc = 0
title, page_id, item_id = page_df.iloc[iloc][['title', 'page_id', 'item_id']]
print('title={}'.format(title))
print('page_id={}'.format(page_id))
print('item_id={}'.format(item_id))


# In[ ]:


print(wikipedia_url_from_title(title))
print(wikipedia_url_from_page_id(page_id))
print(wikidata_url_from_item_id(item_id))


# # Page Metadata: Views
# The `views` column represents page views for the month of November 2019.  Lets see what the most viewed pages were.

# In[ ]:


page_df.sort_values("views", ascending=False).head(25)


# If you read in the full `page.csv` you will see that the most viewed pages are mostly about pop culture with some current events and sports mixed in. The main Wikipedia page (title=`Wikipedia`) is always near the top of the list, but `Simple Mail Transfer Protocol` being in the number one spot appears to be an anomaly for this particular month.  Wikimedia provides a pageviews analysis tool that is very useful for these sorts of investigations ([Simple Mail Transfer Protocol - pageview analysis](https://tools.wmflabs.org/pageviews/?project=en.wikipedia.org&platform=all-access&agent=user&start=2018-01&end=2020-01&pages=Simple_Mail_Transfer_Protocol)).  Lets see what the full distribution looks like.
# 
# Many things in Wikimedia (and in the world at large) are more naturally viewed on a log scale.  Below we'll create a new column in the DataFrame to store log views and we'll do this for other variables as they come up.  We add 1 to any column that we do this for so that we don't run into infinite values for non-negative numbers (i.e., we handle values of 0).

# In[ ]:


page_df["log_views"] = np.log10(page_df["views"] + 1)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)
ax = axes[0]
counts, bins, patches = ax.hist(page_df['log_views'], bins=40, density=True)
ii = np.argmax(counts)
xx = (bins[ii] + bins[ii+1]) / 2
ax.axvline(xx, color='red', ls='--', alpha=0.7)
ax.axhline(0.5, color='red', ls='--', alpha=0.7)
ax.set_xlim(-0.3, 5)
ax.set_xlabel('log10 views')
ax.set_ylabel('fraction')
ax.set_title('probability distribution')

ax = axes[1]
counts, bins, patches = ax.hist(page_df['log_views'], bins=40, density=True, cumulative=True)
ax.axvline(xx, color='red', ls='--', alpha=0.7)
ax.axhline(0.5, color='red', ls='--', alpha=0.7)
ax.set_xlabel('log10 views')
ax.set_title('cumulative distribution')

fig.suptitle('Distribution of page views for {} pages'.format(page_df.shape[0]));


# Above we show probability and cumulative distributions for page views. The vertical dashed line indicates the peak of the probability distribution and the horizontal dashed line indicates a fraction of 0.5. If you read in the full `page.csv` file you will notice a cutoff at $log_{10}(5) = 0.70$ due to the [raw source of pageviews](https://dumps.wikimedia.org/other/pagecounts-ez/merged/) not including counts below 5. 

# # Link Annotated Text: Iterator
# 
# Lets start exploring the link annotated text. The `link_annotated_text.jsonl` file is in the [JSON Lines](http://jsonlines.org/) format.  Each line represents one Wikipedia page and is a string that can be loaded as a JSON object. First we'll write a class to iterate over page lines and load them into dictionaries.

# In[ ]:


class KdwdLinkAnnotatedText:
    
    def __init__(self, file_path, max_pages=MAX_PAGES):
        self.file_path = file_path
        self.num_lines = NUM_KLAT_LINES
        self.max_pages = max_pages
        self.pages_to_parse = min(self.num_lines, self.max_pages)
        
    def __iter__(self):
        with open(self.file_path) as fp:
            for ii_line, line in enumerate(fp):
                if ii_line == self.pages_to_parse:
                    break
                yield json.loads(line)


# In[ ]:


file_path = os.path.join(kdwd_path, "link_annotated_text.jsonl")
klat = KdwdLinkAnnotatedText(file_path)


# Note that we use the variable `klat` to refer to instances of the `KdwdLinkAnnotatedText` class. Because of the way we defined the `__iter__` method, iterating over instances of the class will yield Wikipedia page dictionaries.  

# # Link Annotated Text: Data Schema
# Next we will grab a single page from the iterator and examine its structure.

# In[ ]:


first_page = next(iter(klat))


# In[ ]:


print('page_id: ', first_page['page_id'])
section = first_page['sections'][0]
print('section name: ', section['name'])
print('section text: ', section['text'])
print('section link_offsets: ', section['link_offsets'])
print('section link_lengths: ', section['link_lengths'])
print('section target_page_ids: ', section['target_page_ids'])


# The link data can be used to examine link anchor texts and their target pages.

# In[ ]:


print('anchor text -> target page id')
print('-----------------------------')
for offset, length, target_page_id in zip(
    section['link_offsets'], 
    section['link_lengths'], 
    section['target_page_ids']
):
    anchor_text = section['text'][offset: offset + length]
    print('{} -> {}'.format(anchor_text, target_page_id))


# # Link Annotated Text: In-links and Out-links
# Let's count the incoming and outgoing links to/from each page. This is a slightly more robust measure of "page importance" than page views.

# In[ ]:


in_links = Counter()
out_links = Counter()
for page in tqdm(klat, total=klat.pages_to_parse, desc='calculating in/out links'):
    for section in page['sections']:
        in_links.update(section['target_page_ids'])
        out_links[page['page_id']] += len(section['target_page_ids'])


# In[ ]:


in_links_df = pd.DataFrame(
    in_links.most_common(),
    columns=['page_id', 'in_links'],
)


# In[ ]:


page_df = pd.merge(
    page_df, 
    in_links_df, 
    how='left').fillna(0.0)


# In[ ]:


out_links_df = pd.DataFrame(
    out_links.most_common(),
    columns=['page_id', 'out_links'],
)


# In[ ]:


page_df = pd.merge(
    page_df, 
    out_links_df, 
    how='left').fillna(0.0)


# In[ ]:


page_df['log_in_links'] = np.log10(page_df['in_links'] + 1)
page_df['log_out_links'] = np.log10(page_df['out_links'] + 1)


# In[ ]:


page_df


# # Link Annotated Text: Plots(views | in-links | out-links)

# In[ ]:


print(page_df['in_links'].max())
print(page_df['out_links'].max())
print(page_df['views'].max())


# In[ ]:


LIN_BINS = np.array([
    -0.5, 0.5, 10.5, 100.5, 1_000.5, 10_000.5,
    100_000.5, 1_000_000.5, 1_000_000_000.5])

BIN_NAMES = [
    '0', '1-10', '11-100', '101-1k', '1k-10k',
    '10k-100k', '100k-1M', '>1M']

lin_bins = {
    'in_links': LIN_BINS[:-1],
    'out_links': LIN_BINS[:-1],
    'views': LIN_BINS[1:],
}

bin_names = {
    'in_links': BIN_NAMES[:-1],
    'out_links': BIN_NAMES[:-1],
    'views': BIN_NAMES[1:],  
}

log_bins = {k: np.log10(v + 1) for k,v in lin_bins.items()}


# In[ ]:


for key, bins in log_bins.items():
    page_df['{}_digi'.format(key)] = np.digitize(page_df['log_{}'.format(key)], bins=bins)
    page_df['{}_bin_name'.format(key)] = page_df['{}_digi'.format(key)].apply(lambda x: bin_names[key][x-1])


# In[ ]:


page_df


# In[ ]:


def facetgrid_view_links(x_key, y_key, page_df, hist_or_box):
    if hist_or_box not in ('hist', 'box'):
        raise ValueError()
    
    x_col = '{}_bin_name'.format(x_key)
    y_col = 'log_{}'.format(y_key)    

    grpagg_df = page_df.groupby([x_col])[y_col].agg(['mean', 'median', 'size'])
    grpagg_df = grpagg_df.loc[bin_names[x_key]]
    means = grpagg_df['mean'].values
    medians = grpagg_df['median'].values
    sizes = grpagg_df['size'].values

    g = sns.FacetGrid(
        page_df, 
        col=x_col, 
        height=5, 
        aspect=0.4,
        col_order=bin_names[x_key]
    )

    bins = np.linspace(0, 7, 31)
    if hist_or_box == 'hist':
        g.map(
            sns.distplot, y_col, vertical=True, bins=bins, 
            kde=False, hist_kws={'log': False, 'density': True, 'alpha': 0.9})
    elif hist_or_box == 'box':
        g.map(sns.boxplot, y_col, orient='v')
        
    g.set_titles("")
    g.fig.subplots_adjust(wspace=0.1)
    for iax, (ax, level) in enumerate(zip(g.axes.flat, bin_names[x_key])):
        ax.axhline(y=medians[iax], color='red', ls='-', lw=1.0)
        ax.axhline(y=means[iax], color='red', ls='--', lw=1.0)
        
        if hist_or_box == 'hist':
            ax.text(1e-1, 6.2, 'n={}'.format(sizes[iax]), fontsize=14, weight='bold', color='red')
            ax.set_xlabel(None)
        elif hist_or_box == 'box':
            ax.set_xlabel(bin_names[x_key][iax])

        if iax==0:
            ax.set_ylabel('log {}'.format(y_key.replace('_', '-')))
    
    g.set(ylim=(-0.2, 7))
    g.set(xticklabels=[])
    
    if hist_or_box == 'hist':
        g.set(xlim=(0, 1.5))
        plt.suptitle('PDFs and boxplots: log {} vs {} bins'.format(y_key.replace('_', '-'), x_key.replace('_', '-')))
    elif hist_or_box == 'box':   
        g.fig.text(0.5, 0.0, s=x_key.replace('_', '-'))
        
    g.fig.subplots_adjust(bottom=0.2)


# In[ ]:


x_key = 'views'
y_key = 'in_links'
facetgrid_view_links(x_key, y_key, page_df, 'hist')
facetgrid_view_links(x_key, y_key, page_df, 'box')

x_key = 'views'
y_key = 'out_links'
facetgrid_view_links(x_key, y_key, page_df, 'hist')
facetgrid_view_links(x_key, y_key, page_df, 'box')

x_key = 'in_links'
y_key = 'views'
facetgrid_view_links(x_key, y_key, page_df, 'hist')
facetgrid_view_links(x_key, y_key, page_df, 'box')

x_key = 'out_links'
y_key = 'views'
facetgrid_view_links(x_key, y_key, page_df, 'hist')
facetgrid_view_links(x_key, y_key, page_df, 'box')


# # Link Annotated Text: Outliers(views | in-links | out-links)

# Which pages have a large number of views, but zero in-links or out-links?

# In[ ]:


page_df[page_df['out_links']==0].sort_values('views', ascending=False).head(25)


# In[ ]:


page_df[page_df['in_links']==0].sort_values('views', ascending=False).head(25)


# Which pages have a large number of in-links or out-links but a low number of views?

# In[ ]:


page_df[page_df['views']<=10].sort_values('in_links', ascending=False).head(25)


# In[ ]:


page_df[page_df['views']<=10].sort_values('out_links', ascending=False).head(25)


# There are many ways to segment wikipedia pages using these three metrics (views, in-links, out-links).  Hopefully this will get you thinking about the possibilities!  

# # Link Annotated Text: Vocabulary
# 
# Now lets iterate through part of the corpus and examine the vocabulary used in `Introduction` sections of the first 1M pages.  We'll create a quick tokenizer function that will split on whitespace, lowercase, and remove punctuation. 

# In[ ]:


table = str.maketrans('', '', string.punctuation)
def tokenize(text):
    tokens = [tok.lower().strip() for tok in text.split()]
    tokens = [tok.translate(table) for tok in tokens]
    return tokens


# In[ ]:


unigrams = Counter()
words_per_section = []
for page in tqdm(
    klat, total=min(klat.num_lines, klat.max_pages), 
    desc='iterating over page text'
):
    for section in page['sections']:
        tokens = tokenize(section['text'])
        unigrams.update(tokens)
        words_per_section.append(len(tokens))
        # stop after intro section
        break
print('num tokens= {}'.format(sum(unigrams.values())))
print('unique tokens= {}'.format(len(unigrams)))


# In[ ]:


def filter_unigrams(unigrams, min_count):
    """remove tokens that dont occur at least `min_count` times"""
    tokens_to_filter = [tok for tok, count in unigrams.items() if count < min_count]
    for tok in tokens_to_filter:
        del unigrams[tok]
    return unigrams


# In[ ]:


min_count = 5
unigrams = filter_unigrams(unigrams, min_count)
print('num tokens= {}'.format(sum(unigrams.values())))
print('unique tokens= {}'.format(len(unigrams)))


# In[ ]:


unigrams_df = pd.DataFrame(unigrams.most_common(), columns=['token', 'count'])


# In[ ]:


unigrams_df


# Lets create the classic Zipf style count vs rank plot for our unigrams.  

# In[ ]:


num_rows = unigrams_df.shape[0]
ii_rows_logs = np.linspace(1, np.log10(num_rows-1), 34)
ii_rows = [0, 1, 3, 7] + [int(el) for el in 10**ii_rows_logs]
rows = unigrams_df.iloc[ii_rows, :]
indexs = np.log10(rows.index.values + 1)
counts = np.log10(rows['count'].values + 1)
tokens = rows['token']

fig, ax = plt.subplots(figsize=(14,12))
ax.scatter(indexs, counts)
for token, index, count in zip(tokens, indexs, counts):
    ax.text(index + 0.05, count + 0.05, token, fontsize=12)
ax.set_xlim(-0.2, 6.5)
ax.set_xlabel('log10 rank')
ax.set_ylabel('log10 count')
ax.set_title('Zipf style plot for unigrams');


# And finally lets examine the distribution of section lengths measured in words. 

# In[ ]:


xx = np.log10(np.array(words_per_section) + 1)

fig, axes = plt.subplots(1, 2, figsize=(18,8), sharex=True, sharey=True)
ax = axes[0]
counts, bins, patches = ax.hist(xx, bins=40, density=True)
ii = np.argmax(counts)
xx_max = (bins[ii] + bins[ii+1]) / 2
ax.axvline(xx_max, color='red', ls='--', alpha=0.7)
ax.axhline(0.5, color='red', ls='--', alpha=0.7)
ax.set_xlabel('log10 tokens/section')
ax.set_ylabel('fraction')
ax.set_title('probability distribution')
ax.set_xlim(0.3, 3.8)

ax = axes[1]
counts, bins, patches = ax.hist(xx, bins=40, density=True, cumulative=True)
ax.axvline(xx_max, color='red', ls='--', alpha=0.7)
ax.axhline(0.5, color='red', ls='--', alpha=0.7)
ax.set_xlabel('log10 tokens/section')
ax.set_title('cumulative distribution')

fig.suptitle('Distribution of tokens/section for {} pages'.format(len(words_per_section)));


# Above we show the probability and cumulative distributions of tokens/section for *Introduction* sections. The vertical dashed line indicates the peak of the probability distribution and the horizontal dashed line indicates a fraction of 0.5.

# In[ ]:




