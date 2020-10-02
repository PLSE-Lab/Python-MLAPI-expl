#!/usr/bin/env python
# coding: utf-8

# # Interactive Search Engine for Abstracts and Authors
# 
# When a user enters a question (or we use the Kaggle-provided questions), the top-matching abstracts and authors are returned. I hope this is a useful tool for rapid information retrieval using natural language queries.
# 
# **DIRECTIONS: Click the "Copy and Edit" button. With "load_preprocessed_file=True", run all cells, and scroll to the bottom for the interactive widget. Takes about 90 seconds to fully load.**
# 
# You will see an interactive widget (heavily inspired by this notebook: https://www.kaggle.com/dgunning/browsing-research-papers-with-a-bm25-search-engine):
# 
# ![](https://i.imgur.com/7HHQqqx.jpg)
# 
# Code cells have been collapsed to streamline the presentation, but please expand them to dig into the details. My full methodology is described below. I welcome feedback in the comments!

# In[ ]:


# if set to False, the notebook takes about 10 minutes to run
load_preprocessed_file = True


# # 0. Methodology
#  
#  1. **Load data**
#    * Remove duplicate publications
#    * For each row of data, combine the Title and Abstract into the "document" that is used for all further processing and analysis.
#  2. **Clean text** - the goal is to simplify the text as much as possible. This means removing things that seem useful to us humans, like numbers, verb conjugations, and special characters, but that don't provide much signal to the machine
#    * Replace newlines with spaces
#      * except if a newline is preceded by a hyphen, which indicates that a long word was broken by a line break
#    * Remove non-ASCII characters
#    * Remove numbers
#    * Remove known character patterns that provide no value. Ex: "[Image: see text]"
#    * Lemmatize words - reduces all conjugations and pluralizations down to a single word. This reduces the total number of unique words the machine has to process while preserving the essence of the word's meaning 
#      * https://en.wikipedia.org/wiki/Lemmatisation
#    * Remove stopwords
#      * Remove common English stopwords like "and", "it", "the" and many others
#      * Remove any custom stopwords that don't contribute useful signal
#    * Create ngrams
#      * When 2 tokens frequently appear side-by-side, treat them as the same token.
#      * Example "public health" should be treated as a single token. Replace both words with the single token of "public_health". The words "public" and "health" still appear as single tokens, but not when they occur directly next to each other.
#      * When 2 token are joined, that's called a "bigram". When 3 tokens are joined, that's called a "trigram". In this notebook, I only do bigrams and trigrams, but you could continue joining n number of tokens, which is generally called an "ngram".
#    * Dedupe sentences within each document
#      * Just in case the Title is the same as the opening line of the Abstract, dedupe and drop any later sentences that have already appeared.
#      * After all the cleanup above, some sentences may now appear as identical sets of tokens, which implies they don't contribute any additional valuable signal.
#    * Remove extremely rare words and extremely common words
#      * The machine needs many example of how words are used before it can "understand" what a word means. For this reason, we must remove words that do not appear very frequently across all documents.
#      * Conversely, words like "virus" or "infection" appear in a third of all documents, so the machine would treat those words as unhelpful noise. Remove these common words, which leaves us only with "interesting" words that help differentiate the documents from one another.
#    * Example of the final processing:
#      * Original text: *"Cruise ships carry a large number of people in confined spaces with relative homogeneous mixing. On 3 February, 2020, an outbreak of COVID-19 on cruise ship Diamond Princess was reported with 10 initial cases, following an index case on board around 21-25 January. By 4 February, public health measures such as removal and isolation of ill passengers and quarantine of non-ill passengers were implemented."*
#      * Result seen by the machine: *['cruise', 'ship', 'carry', 'large_number', 'people', 'confine', 'space', 'relative', 'homogeneous', 'mix', 'february', 'outbreak', 'covid', 'cruise', 'ship', 'report', 'initial', 'case', 'follow', 'index', 'case', 'board', 'around', 'january', 'february', 'public_health', 'measure', 'removal', 'isolation', 'ill', 'passenger', 'quarantine', 'passenger', 'implement']*
#  3. **Use CorEx to assign topic-liklihood scores to each document**
#    * This allows us to filter search results that somehow have a strong cosine distance score, but that are not relevant to the topic of the question.
#  4. **Train TF-IDF model**
#    * https://en.wikipedia.org/wiki/Tf%E2%80%93idf
#    * I first attempted to train a Doc2Vec model, but the results were weak. Perhaps there wasn't enough training data. I am curious if anyone knows of a finetuned BERT (or other model) based on medical literature?
#    * Once the model is trained, we can convert a document into a numerical vector. This allows us to use cosine distance to measure the similarity between documents.
#    * Any other form of vectorizing text could be used here. If you had success training a Doc2Vec model, or some other model, I'd love to hear about it in the comments!
#  5. **Interactive widget to explore the results**
#  
#  
#  A big thank you to the following resources:
# 
#  * https://www.kaggle.com/morrisb/ipython-magic-functions
#  * https://www.kaggle.com/gkaraman/topic-modeling-lda-on-cord-19-paper-abstracts
#  * https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
#  * https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
#  * https://www.kaggle.com/dgunning/browsing-research-papers-with-a-bm25-search-engine

# # 1. Load Packages and Raw Data

# In[ ]:


final_df_filename = 'df_final_covid_clean_topics.pkl'

import numpy as np
import pandas as pd
import os
import glob
import json

import pickle as pkl
import string
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize#, word_tokenize
from nltk.corpus import stopwords
import time
from multiprocessing import Pool
import numpy as np
import multiprocessing
from collections import Counter
from itertools import chain
import operator
from gensim.models.phrases import Phrases, Phraser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import itertools
import collections
import random
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse as ss

# https://github.com/gregversteeg/corex_topic
get_ipython().system("pip install 'corextopic'")

from corextopic import corextopic as ct
from corextopic import vis_topic as vt # jupyter notebooks will complain matplotlib is being loaded twice
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from ipywidgets import interact
import ipywidgets as widgets


# Load raw data, dedupe publications, and drop any publications that don't have an Abstract. Then create a new column "document" that is the "title" combined with "abstract".

# In[ ]:


if load_preprocessed_file is False:

    # credit: https://www.kaggle.com/gkaraman/topic-modeling-lda-on-cord-19-paper-abstracts
    df = pd.read_csv(
        '/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv'
        , dtype={
            'Microsoft Academic Paper ID': str
            ,'pubmed_id': str
        })

    # Some papers are duplicated since they were collected from separate sources. Thanks Joerg Rings
    duplicate_paper = ~(df.title.isnull() | df.abstract.isnull()) & (df.duplicated(subset=['title', 'abstract']))
    df = df[~duplicate_paper].reset_index(drop=True)

    df = df.dropna(subset=['abstract'])

    # create a column that appends title+abstract. This column will be the "document" that all searching/clustering/vectorization will use
    df['document'] = df['title'] + '. ' + df['abstract']

    print(df.shape)


# # 2. Clean Data
# 
# Each clean function can be tested independently.

# In[ ]:


# "supercalifragili-\nsticexpialidocious\nthis is a new line" -> "supercalifragilisticexpialidocious this is a new line"
def clean_newlines(text):
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ').replace('\r',' ')
    
    return text

test = 'supercalifragili-\nsticexpialidocious\nthis is a new line '
clean_newlines(test)


# In[ ]:


def clean_chars(text):
    text = "".join(i for i in text if ord(i) < 128) # remove all non-ascii characters
    text = text.replace('\t', ' ') # convert a tab to space
    # fastest way to remove all punctuation (except ' . and !) and digits
    text = text.replace('[Image: see text]', '')
    text = text.translate(str.maketrans('', '', '"#$%&()*+,-/:;<=>@[\\]^_`{|}~' + string.digits))
    
    return text.strip()

clean_chars('[Image: see text] Numbers 123 are greater than 456?!\t"I\'m of the op1ni0n it isn\'t..."')


# Lemmatize the sentence to normalize conjucations and pluralization.

# In[ ]:


# credit: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# helper correctly accounts for the same word having a different
# part-of-speech depending on the context of its usage
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

sentence = "The ten foot striped bats are hanging on their good better best feet. The bat's wings were ten feet wide."

print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])


# Dedupe sentences within a document, just in case the article title is also the first sentence of the abstract, or opening sentence of text_body.

# In[ ]:


def dedupe_sentences(sentences):
    deduped = []
    for s in sentences:
        if s not in deduped:
            deduped.append(s)
    
    return deduped

test_sentences = [
    ['see', 'figure', 'for', 'data'],
    ['not', 'cleared', 'for', 'release'],
    ['new', 'sentence', 'here'],
    ['see', 'figure', 'for', 'data'],
    ['not', 'cleared', 'for', 'release'],
    ['finally']
]

dedupe_sentences(test_sentences)


# Define stopwords that will be removed from all text. For example: "and", "it", "the", "thus".

# In[ ]:


stpwrds_list = stopwords.words('english')

# add custom stopwords here, discovered from most common words and ngrams (further below)
stpwrds_list += ['...', 'also', 'could', 'thus', 'therefore']

stpwrds_lower = [wrd.lower() for wrd in stpwrds_list]

stpwrds_list += stpwrds_lower

stpwrds = set(stpwrds_list) # dedupes stopwords


# Wrapper function that calls all the previously-defined functions.

# In[ ]:


# given a string representing an entire document, returns the following format where all the words are non-stopwords and lemmatized:
#example = [
#    ['Sentence', 'one', 'words'],
#    ['Sentence', 'two', 'words']
#]

def clean(text, min_word_len=3, lower=True):
    
    if (lower is True):
        text = text.lower()
    
    text = clean_newlines(text)
    text = clean_chars(text)
    
    sentences = sent_tokenize(text)
    
    clean_sentences = []
    
    for s in sentences:
        clean_sent_words = [
            lemmatizer.lemmatize(w, get_wordnet_pos(w))
            for w in nltk.word_tokenize(s)
            # skip short words, contraction parts of speech, and storwords
            if len(w) >= min_word_len and w[0] != '\'' and w not in stpwrds
        ]
        
        clean_sentences.append(clean_sent_words)
    
    # one and only one identical sentence is allowed per document
    # this helps avoid common phrases like "Table of data below:" appearing
    # many times, which will skew the word associations
    clean_sentences = dedupe_sentences(clean_sentences)
    
    return clean_sentences

test = " NOT CLEARED FOR PUBLIC RELEASE. See table for references. The ss goes here. "
test += "US interests. U.S. Enterprise. The ten foot striped bats are hanging on their good better best feet. The "
test += "bat's Wings were ten feet wide. Is the U.S. Enterprise 123 better than 456?!\t\"I\'m of the op1ni0n it isn\'t...\""
test += " New sentence. NOT CLEARED FOR PUBLIC RELEASE. See table for references."
clean(test)


# Loop through all rows of the dataframe and apply the clean function.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1\ndef parallelize_dataframe(df, func, n_cores=multiprocessing.cpu_count()):\n    df_split = np.array_split(df, n_cores)\n    pool = Pool(n_cores)\n    df = pd.concat(pool.map(func, df_split))\n    pool.close()\n    pool.join()\n\n    return df\n\ndef clean_dataframe(df):\n    df[\'clean\'] = df.apply(lambda x: clean(x[\'document\']), axis=1)\n\n    return df\n\nif load_preprocessed_file is False:\n    # this parallelize_dataframe way takes about 7 minutes\n    df = parallelize_dataframe(df, clean_dataframe)\n\n    # the "swifter" keyword/library aims to make dataframe processing faster, but it didn\'t help in this case\n    # !pip install \'swifter\'\n    # import swifter\n    # https://towardsdatascience.com/add-this-single-word-to-make-your-pandas-apply-faster-90ee2fffe9e8\n    # this method was slower at 17 minutes in total, but it provided a nice progress bar and countdown timer\n    # df[\'clean\'] = df.swifter.apply(lambda x: clean(x[\'document\']), axis=1)\n\n    df[[\'clean\']].head(3)\nelse:\n    df = pkl.load(open(\'/kaggle/input/cached-data-interactive-abstract-and-expert-finder/\' + final_df_filename, "rb" ))')


# Identify the most common words. If you identify some words as low-value, add them to the stopwords list above and rerun the cells until this point. Repeat until you are happy with the words that are left.

# In[ ]:


clean_words = df['clean'].tolist()
clean_words = [item for sublist in clean_words for item in sublist]

print(str(len(clean_words)) + ' total words in corpus')

counter_obj = Counter(chain.from_iterable(clean_words))
word_counts = counter_obj.most_common()
word_counts.sort(key=operator.itemgetter(1), reverse=True)

word_counts[0:10]


# Identify bigrams and trigrams. Example: frequently, the words "public" and "health" appear next to each other, therefore it should be considered a bigram of "public_health".

# In[ ]:


get_ipython().run_cell_magic('time', '', "# higher threshold means fewer ngrams - open question, how to optimize these hyperparams?\n\nbigram = Phrases(clean_words, min_count=384, threshold=64, delimiter=b'_')\ntrigram = Phrases(bigram[clean_words], min_count=64, threshold=32, delimiter=b'_')")


# Display all the bigrams and trigrams

# In[ ]:


sorted(
    {k:v for k,v in bigram.vocab.items() if b'_' in k if v>=bigram.min_count and str(k).count('_') == 1}.items(),
    key=operator.itemgetter(1),
    reverse=True
)


# In[ ]:


sorted(
    {k:v for k,v in trigram.vocab.items() if b'_' in k if v>=trigram.min_count and str(k).count('_') == 2 }.items(),
    key=operator.itemgetter(1),
    reverse=True
)


# Helper function to replace ngrams.

# In[ ]:


def get_ngram_words(sent_arr):
    result = []

    for s in sent_arr:
        sent_result = []
        for w in trigram[bigram[s]]:
            if (w not in stpwrds): # we need to check again, because we may have added ngrams to the stopword list
                sent_result.append(w)

        result.append(sent_result)
    return result

test_sentences = [
    ['polymerase', 'chain', 'reaction'],
    ['infectious', 'disease', 'cause', 'unknown', 'public', 'health'],
    ['significant', 'acute', 'respiratory', 'disease', 'report', 'world', 'health']
]

get_ngram_words(test_sentences)


# Check all documents for ngrams and replace them.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef convert_ngram_dataframe(df):\n    df['clean'] = df.apply(lambda x: get_ngram_words(x['clean']), axis=1)\n\n    return df\n\nif load_preprocessed_file is False:\n    df = parallelize_dataframe(df, convert_ngram_dataframe)\n\n    df[['clean']].head(3)")


# Build dictionary of all words so we can check for both low frequency and high frequency words across entire corpus.

# In[ ]:


all_words = []
docs = []

for index, row in df.iterrows():
    sent_arr = row['clean']
    doc_words = []
    
    for s in sent_arr:
        for w in s:
            doc_words.append(w)
            all_words.append(w)
    
    docs.append(doc_words)

print('TOTAL WORDS: ' + str(len(all_words)))
print('UNIQUE WORDS: ' + str(len(set(all_words))))


# Filter out words that occur in fewer than "no_below" documents, or more than "no_above%"" of the documents. Without filtering, we have about 86k unique tokens. With filtering we have about 5800 unique tokens which we consider "interesting".

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# credit: https://www.kaggle.com/gkaraman/topic-modeling-lda-on-cord-19-paper-abstracts\n\n# Create a dictionary representation of the documents\ndictionary = Dictionary(docs)\ndictionary.filter_extremes(no_below=32, no_above=0.2)\n\n# Create Bag-of-words representation of the documents\n#corpus = [dictionary.doc2bow(doc) for doc in docs]\n\nprint('Number of unique tokens: %d' % len(dictionary))\n#print('Number of documents: %d' % len(corpus))\n\n\ndef remove_non_dict_words(sent_arr):\n    result = []\n\n    for s in sent_arr:\n        for w in s:\n            if w in dictionary.token2id:\n                result.append(w)\n                \n    return result\n\ndef remove_non_dict_words_df(df):\n    df['clean_tfidf'] = df.apply(lambda x: remove_non_dict_words(x['clean']), axis=1)\n\n    return df\n\ndf = parallelize_dataframe(df, remove_non_dict_words_df)\n\ndf = df.reset_index() # after all the processing, there are some gaps in the indices, so we reset them to make index counting easier later")


# # 3. CorEx Topic Modeling
# 
# Now train a CorEx model to generate topics for the entire corpus. Then assign topic-liklihood scores to every document. We will use these later to filter the search results by topic.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif load_preprocessed_file is False:\n\n    def dummy(doc):\n        return doc\n\n    vectorizer = CountVectorizer(\n        tokenizer=dummy,\n        preprocessor=dummy,\n    )  \n\n    corex_docs = df[\'clean_tfidf\'].tolist()\n    doc_word = vectorizer.fit_transform(corex_docs)\n\n    doc_word = ss.csr_matrix(doc_word)\n\n    # Get words that label the columns (needed to extract readable topics and make anchoring easier)\n    words = list(np.asarray(vectorizer.get_feature_names()))\n\n    #doc_word.shape # n_docs x m_words\n\n\n    # https://github.com/gregversteeg/corex_topic\n    # Train the CorEx topic model with x topics (n_hidden)\n    topic_model = ct.Corex(n_hidden=12, words=words, max_iter=500, verbose=False, seed=2020)\n    #topic_model.fit(doc_word, words=words)\n\n    topic_model.fit(doc_word, words=words)\n\n\n    plt.figure(figsize=(10,5))\n    plt.bar(range(topic_model.tcs.shape[0]), topic_model.tcs, color=\'#4e79a7\', width=0.5)\n    plt.xlabel(\'Topic\', fontsize=16)\n    plt.ylabel(\'Total Correlation (nats)\', fontsize=16);\n    # no single topic should contribute too much. If one does, that indicates more investigation for boilerplate text, more preprocessing required\n    # To find optimal num of topics, we should keep adding topics until additional topics do not significantly contribute to the overall TC\n    \n    pkl.dump(topic_model, open(\'corex_topic_model.pkl\', "wb"))\nelse:\n    topic_model = pkl.load(open(\'/kaggle/input/cached-data-interactive-abstract-and-expert-finder/corex_topic_model.pkl\', "rb" ))\n\n# Print all topics from the CorEx topic model\ntopics = topic_model.get_topics()\ntopic_list = []\n\nfor n,topic in enumerate(topics):\n    topic_words,_ = zip(*topic)\n    print(\'{}: \'.format(n) + \',\'.join(topic_words))\n    topic_list.append(\'topic_\' + str(n) + \': \' + \', \'.join(topic_words))')


# Iterate through all documents and assign CorEx topic liklihood scores.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif load_preprocessed_file is False:\n\n    # remove any existing topic columns. This allows us to iterate on number of topics\n    for c in [col for col in df.columns if col.startswith(\'topic_\')]:\n        del df[c]\n\n    # TODO: inefficient code. Ideas to improve this: for each topic, first create a np array of length of rows, then iterate\n    # over those indices setting the scores with the rest default to 0, then set the whole df col\n    for topic_num in range(0, len(topic_model.get_topics())):\n        df[\'topic_\' + str(topic_num)] = 999999.9\n\n    for topic_num in range(0, len(topic_model.get_topics())):\n        for ind, score in topic_model.get_top_docs(topic=topic_num, n_docs=9999999, sort_by=\'log_prob\'):\n            df[\'topic_\' + str(topic_num)].iloc[ind] = score\n\n    # finally save the dataframe so we can load it quicker in situations where we just want to interact with the results.\n\n    pkl.dump(df, open(final_df_filename, "wb"))')


# # 4. TF-IDF Vectorize Documents
# 
# Now train the TF-IDF model, which uses the "clean_tfidf" column as the content to vectorize.

# In[ ]:


# because we are doing our own tokenization, we use a dummy function to bypass
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

tfidf_docs = df['clean_tfidf'].tolist()
tfidf_matrix = tfidf.fit_transform(tfidf_docs)


# # 5. Interactive Search Widget
# 
# DIRECTIONS:
# 
#  1. Type in your own question in the "query" textbox. The default is "cruise ship spread rate".
#  2. Select a topic area to filter the result.
#  3. Drag the topic_threshold slider to filter out results that don't strongly align to the topic.
#    * Interestingly, if you lower the threshold slider all the way to -20, with the default query of "cruise ship spread rate", the top result is not what you'd expect. Apparently, there's an all-caps acronym SHIP that stands for something not boat related. This is why the CorEx topic thresholds can be useful to filter out these unexpectedly good TF-IDF matches.
# 
# Here are a few of the Kaggle-provided questions:
# 
# * What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?
# * What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?
# * What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)

@interact
def search_articles(
    query='cruise ship spread rate',
    topic=topic_list,
    topic_threshold=(-20, 0, 0.01)
):
    clean_query_words = remove_non_dict_words(get_ngram_words(clean(query)))
    query_vector = tfidf.transform([clean_query_words])
    
    scores = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    df['cosine_dist'] = scores

    # these are the ordered search results according to TF-IDF

    # smaller corex_topic scores means more likely to be of that topic
    corex_cols = [col for col in df if col.startswith('topic_')]
    select_cols = ['title', 'abstract', 'authors', 'cosine_dist'] + corex_cols
    
    results = df[select_cols].loc[df[topic.split(':')[0]] > topic_threshold].sort_values(by=['cosine_dist'], ascending=False).head(10)
    
    top_row = results.iloc[0]
    
    print('TOP RESULT:\n')
    print(top_row['title'] + '\n')
    print(top_row['abstract'])
    
    print('\nAUTHORS:\n')
    print(top_row['authors'])
    
    return results

