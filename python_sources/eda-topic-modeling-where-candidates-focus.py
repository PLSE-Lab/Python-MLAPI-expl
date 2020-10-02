#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Load the data - Use "cp1252" encoding. It is the most complete encoding for special characters / fancy apostrophes.

# In[ ]:


df = pd.read_csv('/kaggle/input/democratic-debate-transcripts-2020/debate_transcripts_v3_2020-02-26.csv', encoding='cp1252')
df['date'] = pd.to_datetime(df.date)
df.head()


# ## Speakers who are candidates still in the race:
# - Joe Biden
# - Elizabeth Warren
# - Bernie Sanders
# - Pete Buttigieg
# - Amy Klobuchar
# - Michael Bloomberg
# - Tom Steyer
# - Tulsi Gabbard

# ## Filter to only the above mentioned candidates

# In[ ]:


df = df.loc[df.speaker.isin({'Joe Biden', 'Elizabeth Warren', 'Bernie Sanders', 'Pete Buttigieg', 'Amy Klobuchar', 'Michael Bloomberg', 'Tom Steyer', 'Tulsi Gabbard'})]
df.speaker.value_counts()


# ## Speaking Time EDA

# ### Candidate Speaking Time

# In[ ]:


df.groupby(by='speaker').speaking_time_seconds.sum().plot.bar()
plt.show()


# ### Candidate Speaking Time over time

# In[ ]:


# Multi-Index on debate, candidate
debate_candidate_time = df.groupby(by=['date', 'speaker']).speaking_time_seconds.sum()
# Most recent debate
debate_candidate_time['2020-02-25']


# In[ ]:


# Multi-Index on candidate, debate
candidate_debate_time = df.groupby(by=['speaker', 'date']).speaking_time_seconds.sum()
candidate_debate_time


# In[ ]:


# Print Median Speaking Times
for candidate in df.speaker.unique():
    med = round(candidate_debate_time[candidate].median()/60)
    print(f'{candidate}: {med} minutes (median)')


# In[ ]:


# Plot Speaking Times Line Graph
plt.figure(figsize=(20,10))
for candidate in df.speaker.unique():
    candidate_debate_time[candidate].plot(label=candidate)
plt.legend()
plt.xlabel('date')
plt.ylabel('num seconds')
plt.title('Candidate Speaking Time over time')
plt.show()


# We can see that Elizabeth Warren is given the most time to speak, followed by Bernie Sanders and Joe Biden, then Buttigieg, Klobuchar, and Bloomberg.
# 
# We can see that Sanders has had the majority of speaking time recently, and despite his just recent entry into the race, Michael Bloomberg spoke for the second longest time in the recent debate.
# 
# Surprisingly, Amy Klobuchar has quite a bit of speaking time despite her low performance in the polls.

# ## Define Methods for Parsing the texts
# - I will be using `spaCy` as I have found their parsing to be quick and accurate.
# 
# Our goal will be to turn a document into a list of terms. I will not be using a DNN like *Bert*, and therefore will be doing a good deal of preprocessing, including stop word removal, lemmatization, and noun chunk/entity merging.

# In[ ]:


import spacy

nlp = spacy.load('en_core_web_sm')


# In[ ]:


# I add additional stops that lead to lower quality topics
additional_stops = {'things', 'way', 'sure', 'thing', 'question', 'able', 'point', 'lot', 'time'}


# In[ ]:


from spacy.util import filter_spans


def _remove_stops(span):
    while span and span[0].pos_ not in {'ADJ', 'NOUN', 'PROPN'}:
        span = span.doc[span.start+1:span.end]
    return span


# Resource: 
# https://github.com/explosion/spacy/blob/master/examples/information_extraction/entity_relations.py
def merge_ents_and_nc(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = list(map(_remove_stops, spans))
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            if span:
                # Added this in from the code - need to lemmatize better and keep ent types
                root = span.root
                attrs = {'LEMMA': span.text.lower(), 'POS': root.pos_, 'ENT_TYPE': root.ent_type_}
                retokenizer.merge(span, attrs=attrs)


def to_terms_list(doc):
    merge_ents_and_nc(doc)
    return [term.lemma_ for term in doc if len(term.lemma_) > 2 and             not term.is_stop and             term.pos_ in {'NOUN', 'PROPN', 'ADJ'} and             term.lemma_ not in additional_stops and             "crosstalk" not in term.lower_]


# ## Simple is Better than Complex. My Simple Idea with Topic Modeling
# 1. Create a Corpus of Documents from `df.speech.values`
# 2. Most Frequent Terms by Speaker
# 3. Vectorize using `sklearn.feature_extraction.text.TfidfVectorizer`, using parsed documents as output from `to_terms_list` as input
# 4. Use NMF Topic Model, `sklearn.decomposition.NMF`, to break our corpus into `N` topics.

# In[ ]:


from spacy.tokens.doc import Doc

corpus = list(nlp.pipe(df.speech.values, n_threads=4))


# In[ ]:


terms = [to_terms_list(doc) for doc in corpus]
df['terms'] = terms


# In[ ]:


import itertools
from collections import Counter

def most_frequent(terms, k):
    flat_terms = list(itertools.chain(*terms))
    return Counter(flat_terms).most_common(k)


# In[ ]:


mf = df.groupby(by='speaker').terms.apply(lambda terms: most_frequent(terms, 10))
speakers = mf.index
top_terms = mf.values
for s, tt in zip(speakers, top_terms):
    print(f"{s}: {tt}\n")


# Notice how the most successful candidate, Bernie Sanders, speaks less about Donald Trump and focuses much more on healthcare, medicare, and the american people.
# 
# Compare this to second tier candidates like Buttigieg and Klobuchar who have "president" or "donald trump" surface higher in their most frequent terms.

# ## Topic Modeling
# I take a heuristic approach to topic modeling, and have executed the below cells a few times to find a suitable value for `N`. I look for a mix of granularity and qualitative meaning.
# 
# After iterative attempts, I chose **8** topics, and I do a human pass through to summarize them. One particularly great approach for helping decide the number of topics is *Stability*, as seen in the paper: *[How Many Topics? Stability Analysis for Topic Models*](https://arxiv.org/pdf/1404.4606.pdf). I need to clean up my code for that, and will implement it here in the coming days, stay tuned...

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Useful hack: this dummy allows us to use our extracted terms from the methods above, overriding sklearn's tokenizer.
dummy = lambda x: x

vectorizer = TfidfVectorizer(max_df=0.5, min_df=10, preprocessor=dummy, tokenizer=dummy, ngram_range=(1,2))
features = vectorizer.fit_transform(terms)


# ## About NMF:
# **NMF** stands for **Non-negative Matrix Factorization** and it is a form of dimensionality reduction. 
# 
# The idea is:
# 
# Given some non-negative matrix $X$, find two matrices, $W$ and $H$, such that $W\cdot H \approx X$   
# 
# Without getting too deep into the math, here is a high level leyman's overview of what happens:
# 
# 1. Initialize matrices $W$ and $H$ with random values.
# 2. Perform $W \cdot H$ and compare the values to the original matrix X
# 3. Now, adjust these values incrementally through a number of iterations to optimize.
# 
# 
# The output of this is the matrices $W$ and $H$. In the case of text processing where $X$ is some matrix containing text feature weights, these represent the 'document-topic matrix' and 'topic-term matrix' respectively.
# 
# The 'document-topic matrix', or `doc_topic_matrix` in our code, has the shape `len(corpus), n_topics` and represents the topic distribution for each document.
# 
# The 'topic-term matrix', `topic_term_matrix` in our code, has the shape `n_topics, len(terms)`, and represents the term distribtuion for each topic, i.e. topic 0 consists dominantly of terms A, B, C, ...

# In[ ]:


N=8
tm = NMF(n_components=N)
doc_topic_matrix = tm.fit_transform(features)
topic_term_matrix = tm.components_
terms = vectorizer.get_feature_names()

# Array where rows are docs, columns are topics
print(doc_topic_matrix.shape)


# In[ ]:


def top_topic_terms(topic_term_matrix, i, n_terms=10):
    topic = topic_term_matrix[i]
    return [(terms[idx], topic[idx]) for idx in np.argsort(topic)[::-1][:n_terms]]


# In[ ]:


for i in range(N):
    print(i)
    print(top_topic_terms(topic_term_matrix, i))


# In[ ]:


names = {0: 'The American People',
         1: 'The President of the United States',
         2: 'Biden Telling us what he is saying is a matter of fact',
         3: 'The American Country',
         4: 'Global Diplomatic Issues',
         5: 'Social/Constitutional Rights',
         6: 'Healthcare and Medicare',
         7: 'Donald Trump'}


# In[ ]:


for i in range(N):
    col = f"topic_{i}"
    df[col] = doc_topic_matrix[:,i]
df.head()


# In[ ]:


def percent_comp(x):
    return x/x.sum()


# ## Speaker-Topic DataFrame
# I create a `pandas.DataFrame` to display: for each speaker, their total weight in each topic.
# This can help answer two things:
# 1. If we apply `percent_comp`, the percent composition on each row, we get the percentage each candidate talks about each topic - which topics they focus on.
# 2. If we apply `percent_comp` on each column, we'll see who dominates a topic.

# In[ ]:


df_speaker_topics = df.groupby(by=['speaker'])[df.columns[-N:]].sum()
df_speaker_topics


# ## Percent of Topic, by Candidate
# Show, for each topic, which candidates talk about it the most, by the above `df_speaker_topics`

# In[ ]:


import matplotlib.pyplot as plt

topic_speaker_dist = df_speaker_topics.apply(percent_comp, axis=1)

for i, topic in enumerate(topic_term_matrix):
    print(names[i])
    print(' | '.join(terms[idx] for idx in np.argsort(topic)[::-1][:10]))
    topic_speaker_dist[f'topic_{i}'].plot.bar()
    plt.title(f'Topic: {names[i]}')
    plt.xlabel('Candidate')
    plt.ylabel('Percent of Topic')
    plt.show()

