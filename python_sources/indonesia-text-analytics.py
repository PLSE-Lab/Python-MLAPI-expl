#!/usr/bin/env python
# coding: utf-8

# # Identifying Key Issues and Trends Related to a Indonesia from News Reports

# ### Import dependencies

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import math
import csv
from tqdm import tqdm

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### EDA

# In[ ]:


df = pd.read_csv("/kaggle/input/govtech-text-indon/indonesia_news_jan_may_2020.csv", parse_dates=['NormalisedDate'])
print('Shape of dataset',df.shape)
df.head()


# Checking for missing values

# In[ ]:


df.isna().sum()


# This does not corroborate with the output from df.head(), where we see some Author values registered as an empty list.

# In[ ]:


print('In fact, there are {} empty author entries.'.format(df.Author[df['Author'] == '[]'].count()))
df.Author = df.Author.replace({'[]': np.NaN})


# To account for this shortfall of information from author, we can use the domain name from SOURCEURL.

# In[ ]:


try: 
    import tldextract
except ImportError:
    get_ipython().system('pip install tldextract')
    import tldextract
df['domain'] = df['SOURCEURL'].apply(lambda x: tldextract.extract(x).domain)
df['domain'].value_counts(ascending=True)[-20:].plot(kind='barh',figsize=[6, 5],title='Top 20 Domain Names')
plt.show()
num_articles = sum(df['domain'].value_counts()[:20])
print('The top {} domains published {} out of {} articles ({:.2f}%).'.format(20,num_articles,len(df),num_articles/len(df)*100))
print('There are {} unique domains.'.format(df['domain'].nunique()))


# Example text

# In[ ]:


print(df['Text'].iloc[0])


# Plotting the number of articles per date, we see a cyclic pattern of roughly 4 peaks per month.

# In[ ]:


df.groupby('NormalisedDate').size().plot.line(figsize=[10, 3],title='Number of articles across time')
plt.ylabel('Number of articles')
plt.xlabel('Date')
plt.show()


# Get statistics of number of articles published per day.

# In[ ]:


print('News articles dates range from {} to {}.'.format(df.NormalisedDate.min().strftime('%A %d %B %Y'),df.NormalisedDate.max().strftime('%A %d %B %Y')))
print('Max number of articles: {} published on {}.'.format(df.groupby('NormalisedDate').size().max(),df.groupby('NormalisedDate').size().idxmax().strftime('%A %d %B %Y')))
print('Statistics of articles published per day:')
print(df.groupby('NormalisedDate').size().describe())


# The Monday to Friday work week explains the cyclic nature of publications, with dips representing weekends.

# In[ ]:


df['NormalisedDate'].dt.day_name().value_counts().plot(kind='barh',figsize=[5, 3],title='Histogram of day of week published')
plt.xlabel('Number of articles')
plt.ylabel('Day of week')
plt.show()


# Aggregating by week, we see the number of articles being rather constant

# In[ ]:


df['week'] = df['NormalisedDate'].apply(lambda x : x.isocalendar()[1])
df.groupby('week').size().plot.line(figsize=[10, 3],title='Number of articles across time')

plt.xticks(list(i for i in range(df['week'].max()+1) if i%4 == 0))
plt.xlabel('Week number')
plt.ylabel('Number of articles')
plt.show()


# In[ ]:


df['NormalisedDate'].dt.month_name().value_counts().plot(kind='barh',figsize=[4, 3],title='Histogram of month published')
plt.xlabel('Number of articles')
plt.ylabel('Month')
plt.show()


# ## Text preprocessing
# Cleaning text: Removing punctuation and mapping text to lowercase

# In[ ]:


df['text'] = df['Text'].str.strip().str.lower().str.replace('[^\w\s]','') # regex matching on non letters, numbers, underscores or spaces


# In[ ]:


nlp = spacy.load("en_core_web_lg",disable=["tagger", "parser"])
tqdm.pandas()


# In[ ]:


stopwords = spacy.lang.en.stop_words.STOP_WORDS
custom_stopwords = {'say'}
for word in custom_stopwords:
    stopwords.add(word)


# In[ ]:


def spacy_tokenizer(sentence):
    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1 or word.lemma_ in custom_stopwords)]


# In[ ]:


print('Example cleaned text')
print(' '.join(spacy_tokenizer(df.iloc[0].text)))


# ## LDA Topic Modeling

# In[ ]:


texts = df.text
count_vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, min_df=3)
count_vector = count_vectorizer.fit_transform(tqdm(texts))


# In[ ]:


word_count = pd.DataFrame({'word': count_vectorizer.get_feature_names(), 'count': np.asarray(count_vector.sum(axis=0))[0]})
word_count.sort_values('count', ascending=False).set_index('word')[:20].sort_values('count', ascending=True).plot(kind='barh')
plt.title('Word Count in Combined Corpus')
plt.xlabel('Count')
plt.show()


# In[ ]:


num_topics = 30


# In[ ]:


lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
lda.fit(count_vector)


# In[ ]:


def print_top_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        
        print(message)
print_top_words(lda, count_vectorizer, n_top_words=25)


# In[ ]:


def get_top_words(model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topics


# In[ ]:


topics = pd.Series(get_top_words(lda, count_vectorizer, n_top_words=25))
topics


# In[ ]:


df['topic'].value_counts(ascending=True)[-20:].plot(kind='barh',figsize=[10, 5],title='Top 20 Topics')
plt.title('Topic Count in Combined Corpus')
plt.xlabel('Count')
plt.show()


# In[ ]:


tfidf = TfidfTransformer()
tfidf.fit_transform(count_vector)

idf_weights = pd.DataFrame(tfidf.idf_, index=count_vectorizer.get_feature_names(),columns=["idf_weights"]).sort_values(by=['idf_weights'])
idf_weights


# In[ ]:


doc_topic_dist = pd.DataFrame(lda.transform(count_vector))
df['topic'] = doc_topic_dist.idxmax(axis=1)
df['probability'] = doc_topic_dist.max(axis=1)


# In[ ]:


prob = 0.1
plt.bar(range(20),[doc_topic_dist[doc_topic_dist>prob][i].count() for i in range(20)])
plt.title('Number of articles by topic with >{}% in content'.format(prob*100))
plt.xlabel('Topic')
plt.ylabel('Count')
plt.show()


# ## Visualising topics

# Plotting wordclouds of topics to visualize

# In[ ]:


def get_topic_wordclouds(n_topics,idf=False):
    """
    Generates top n topics by number of articles in dataset.
    Wordclouds of each topic is printed as output.
    Size of words are scaled by count if idf is false. Else, it is scaled by tf-idf.
    """
    for topic in topics[df['topic'].value_counts()[:n_topics].index]:
        if idf:
            weights = dict(((word, len(word_count[word_count['word']==word]['count'].values[0] * idf_weights.loc[word])) for word in topic))
        else:
            weights = dict(((word, word_count[word_count['word']==word]['count'].values[0]) for word in topic))

        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',width=2400, height=1500)
        wordcloud.generate_from_frequencies(weights)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        
get_topic_wordclouds(4,idf=False)


# In[ ]:


get_topic_wordclouds(4,idf=True)


# ## Topics over time

# Plot top n topics against time

# In[ ]:


n = 8

plt.figure(figsize=(20,8))
for i in range(n):
    plt.plot(df[df['topic'] == df['topic'].value_counts().index[i]].groupby('week').size(),marker='x', label='Topic '+str(df['topic'].value_counts().index[i])+': '+' '.join(topics.iloc[df['topic'].value_counts().index[i]][:5]))
plt.legend(loc="upper right")
plt.title('Top '+str(n)+' topics over time')
plt.xticks(list(i for i in range(df['week'].max()+1) if i%4 == 0))
plt.xlabel('Week number')
plt.ylabel('Number of articles')
plt.show()


# ## Comparing topic similarity

# Compare the similarity between topics.

# In[ ]:


def plot_difference_plotly(mdiff, title="", annotation=None):
    """Plot the difference between models.

    Uses plotly as the backend."""
    import plotly.graph_objs as go
    import plotly.offline as py

    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "similarity {}".format(", ".join(diff_tokens))
                for diff_tokens in annotation
            ]
            for row in annotation
        ]

    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    py.iplot(dict(data=[data], layout=layout))


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)


try:
    get_ipython()
    import plotly.offline as py
except Exception:
    #
    # Fall back to matplotlib if we're not in a notebook, or if plotly is
    # unavailable for whatever reason.
    #
    plot_difference = plot_difference_matplotlib
else:
    py.init_notebook_mode()
    plot_difference = plot_difference_plotly


# In[ ]:


top_n = num_topics
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union, list(set(list1).intersection(list2))

print('Example similarity:')
print(jaccard_similarity(topics.iloc[1][:top_n],topics.iloc[2][:top_n]))


# In[ ]:


closest = []
annotation = []
mdiff = np.ones((num_topics, num_topics))
for i in range(num_topics):
    to_add = []
    for j in range(num_topics):
        score, words = jaccard_similarity(topics.iloc[i][:top_n],topics.iloc[j][:top_n])
        to_add.append(words)
        mdiff[i,j] = score
        if 0.2 <score<1: closest.append((i+1,j+1,score,words))
    annotation.append(to_add)

plot_difference(mdiff, title="Topic difference by Jaccard similarity")


# An attempt to compare correlation in publication time with topics failed, due to statistical insignificance.

# In[ ]:


from scipy.stats.stats import pearsonr

topic_time_series = []
for j in range(num_topics):
    week = df[df['topic'] == j].groupby('week').size()
    topic_time_series.append([week[i] if i in week.index else 0 for i in range(1, df['week'].max()+1)])
    
mdiff = np.ones((num_topics, num_topics))
insignificant = []

for i in range(num_topics):
    for j in range(num_topics):
        rho, p = pearsonr(topic_time_series[i],topic_time_series[j])
        mdiff[i,j] = rho
        if not p < 0.05: insignificant.append((i,j,rho,p))

print('There are {} out of {} values with p >= 0.05, i.e. not statistically significant correlation.'.format(len(insignificant),30**2))
plot_difference(mdiff, title="Topic time series by Pearson Correlation")


# Visualizing similarity of topics by Multi-dimensional scaling using T-SNE.

# In[ ]:


import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, count_vector, count_vectorizer, mds='tsne', sort_topics=False)
pyLDAvis.display(panel)


# ## Mapping topics to sources

# ### Initial idea
# Topics exclusive to certain news outlets:
# Calculate # of articles / # topics for each domain and topic. If rarity > threshold, topic is exclusive to these domains (i.e. x% of articles published by domain).

# In[ ]:


domain_topic_count = df.groupby(['topic','domain']).size().reset_index(name='count')
domain_topic_count


# Looking at the range of rarity values calculated (max = 0.7%), we need to try another measure.

# In[ ]:


rarity_range = (domain_topic_count['count'] / df['topic'].value_counts()[domain_topic_count['topic']]).value_counts()
rarity_range


# We see that many domains appear only once or twice (55.9% and 15.4% respectively).

# In[ ]:


(df['domain'].value_counts().value_counts()/df['domain'].nunique())[:10]


# Looking at top topics published by top n domains, for n = 20

# In[ ]:


n = 20
top_domains = {domain for domain in df['domain'].value_counts().index[:n]}
print(top_domains)


# Get top topics per domain within top 20 domains.

# In[ ]:


domain_topic_max = domain_topic_count.iloc[domain_topic_count.groupby(['domain']).idxmax()['count']]
domain_topic_max


# In[ ]:


domain_published_sum = domain_topic_count.groupby('domain').sum()['count']
domain_published_sum = domain_published_sum[domain_published_sum > 2] # sources with at least 3 articles overall
domain_published_sum


# In[ ]:


domain_topic_max = domain_topic_max[domain_topic_max['domain'].isin(domain_published_sum.index)]
domain_topic_max = domain_topic_max[domain_topic_max['count'] > 2] # sources with at least 3 articles in top topic
domain_topic_max


# Getting the most popular topic per publisher with at least 3 publications in the topic.

# In[ ]:


top_topics = domain_topic_max['topic'].value_counts()
top_topics.plot(kind='bar',figsize=[8, 5])

plt.title('Histogram of top topics per publisher with >2 articles on the topic')
plt.ylabel('Number of publishers')
plt.xlabel('Topic number')
plt.show()

for i in top_topics[:3].index:
    print('Topic {}: {}.'.format(i,' '.join(topics[i])))


# Finding domains that publish more than 50% of its publications on its top topic.

# In[ ]:


domain_topic_max['proportion'] = [(domain_topic_max[domain_topic_max['domain']==domain]['count'].values[0] / domain_published_sum[domain]) for domain in domain_topic_max['domain']]
domain_topic_max


# In[ ]:


domain_topic_max_filtered = domain_topic_max[domain_topic_max['proportion']>.5]
domain_topic_max_filtered['topic'].value_counts().plot(kind='bar',figsize=[8, 5])

for i in domain_topic_max_filtered['topic'].value_counts()[:3].index:
    print('Topic {}: {}.'.format(i,' '.join(topics[i])))

plt.title('Histogram of top topics per publisher with >50% of and >3 articles on topic')
plt.ylabel('Number of publishers')
plt.xlabel('Topic number')
plt.show()


# ## NER on topics

# Using spaCy, we classify named entity from each topic by types, and generate wordclouds to visualize the main entities by types.

# In[ ]:


topics_docs = topics.apply(lambda x : nlp(' '.join(x)))

from collections import Counter
ner_count = {}
for doc in topics_docs:
    for ent in doc.ents:
        if ent.label_ not in ner_count: ner_count[ent.label_] = Counter()
        ner_count[ent.label_][ent.text]+=1
        
for entity, counter in ner_count.items():
#     print(entity)
    text = []
    for word, count in counter.most_common(50):
        [text.append(word) for _ in range(count)]
#         print(word,count)
        
    if len(text) > 10:
        wordcloud = WordCloud(background_color="white",contour_width=3, contour_color='steelblue') # , max_words=5000, ,width=2400, height=1500
        wordcloud.generate(' '.join(text))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(entity+': '+spacy.explain(entity))
        plt.show()


# In[ ]:


from spacy import displacy
displacy.render(topics_docs, style="ent")

