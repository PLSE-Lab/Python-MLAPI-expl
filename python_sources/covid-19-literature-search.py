#!/usr/bin/env python
# coding: utf-8

# Load in data from a repository of over 30000 documents of peer-reviewed and pre-print articles regarding topics associated with COVID-19

# In[ ]:


import os
import urllib.request
import tarfile
import json
import nltk
import csv
import re
import pickle
from os import path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import Word2Vec
from gensim.models import CoherenceModel
# Gensim logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
word2vec_model = Word2Vec.load("word2vec_1000ITR.model")
ps = PorterStemmer()
stem_mode = 1 #Use stemming?
# Download and unpack the collection
def getData():
    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz',
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz',
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz',
            'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']

    # Create data directory
    try:
        os.mkdir('./data')
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

    # Download all files
    for i in range(len(urls)):
        urllib.request.urlretrieve(urls[i], './data/file' + str(i) + '.tar.gz')
        print('Downloaded file ' + str(i + 1) + '/' + str(len(urls)))
        tar = tarfile.open('./data/file' + str(i) + '.tar.gz')
        tar.extractall('./data')
        tar.close()
        print('Extracted file ' + str(i + 1) + '/' + str(len(urls)))
        os.remove('./data/file' + str(i) + '.tar.gz')


# Parsing the title, abstract and body text of each document, along with lemmatization and word stemming

# In[ ]:


def extract():
    parsed_data = dict()
    papers_read = 0
    limit = 99999999
    dictionary = []
    diction = []
    whole_word_count = dict()
    # Iterate through all files in the data directory
    for root_subdir, root_dirs, root_files in os.walk('./data'):
        for root_dir in root_dirs:
            for subdir, dirs, files in os.walk('./data/' + root_dir):
                for file in files:
                        if papers_read>limit:
                            break
                        with open(os.path.join(subdir, file)) as f:
                            word_count = dict()
                            word_stem_count = dict()
                            file_data = json.loads(f.read())
                            paper_id = file_data['paper_id']
                            metadata = file_data['metadata']
                            print(papers_read)
                            if metadata:
                                title_clean = []
                                title = metadata['title']
                                title_token = word_tokenize(title)
                                for word in title_token:
                                    title_lem = lemmatizer.lemmatize(word.lower())
                                    title_sub = re.sub("[^a-zA-Z0-9]", '', title_lem)
                                    if title_sub != '' and title_sub not in stop_words:
                                        title_clean.append(title_sub)
                                authors = list()
                                for author in metadata['authors']:
                                    authors.append("%s %s" % (author['first'], author['last']))
                                authors_val = ' '.join(authors)
                                author_token = word_tokenize(authors_val)
                                author_clean=[]
                                for word in author_token:
                                    author_lem = lemmatizer.lemmatize(word.lower())
                                    author_sub = re.sub("[^a-zA-Z0-9]", '', author_lem)
                                    if author_sub != '' and author_sub not in stop_words:
                                        author_clean.append(author_sub)
                            abstract = readText(file_data['abstract'])
                            if abstract == '':
                                continue
                            body = readText(file_data['body_text'])

                            # tokenize/stemming/lemmatizer/lower case/remove punctuation
                            words = []
                            words_stem = []
                            ngram_words = []
                            ngram_stem = []
                            for word in word_tokenize(body):
                                # PS_word=PS.stem(lem_word)
                                lem_word = lemmatizer.lemmatize(word.lower())
                                #  if lem_word not in ["?","!",".","/","[","]"]
                                word_short_only = re.sub("[^a-zA-Z0-9]", '', lem_word)
                                if len(word_short_only) < 3:
                                    word_only = ''
                                elif len(word_short_only) >= 3:
                                    word_only = word_short_only
                                if word_only != '' and word_only not in stop_words:
                                    words_stem.append(ps.stem(word_only))
                                    words.append(word_only)
                                    if word_only in word_count:
                                        word_count[word_only] += 1
                                    else:
                                        word_count[word_only] = 1
                                    if ps.stem(word_only) in word_stem_count:
                                        word_stem_count[ps.stem(word_only)] += 1
                                    else:
                                        word_stem_count[ps.stem(word_only)] = 1
                                    if ps.stem(word_only) in whole_word_count:
                                        whole_word_count[ps.stem(word_only)] += 1
                                    else:
                                        whole_word_count[ps.stem(word_only)] = 1
                            papers_read += 1
                            if stem_mode == 0:
                                dictionary.append(words)
                            if stem_mode == 1:
                                dictionary.append(words_stem)
                            #diction = LDA(words_stem)
                            #ngram_words = Ngram(words)
                            #ngram_stem = Ngram(words_stem)
                            parsed_data[paper_id] = Paper(title, title_clean, author_clean, abstract, body, words, words_stem, word_count, word_stem_count)
        diction = corpora.Dictionary(dictionary)
    return parsed_data, dictionary, diction, whole_word_count

a, dictionary, diction, whole_word_count = extract()


# Constructing LDA modeling topics based on pre-defined number of topics (6). Chunk size of 300 and over 5 passes.

# In[ ]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=saved_corpus, id2word=diction, num_topics=6,random_state=100, update_every=1, chunksize=300, passes=5,alpha='auto', per_word_topics=True)


# View topic keywords as well as associate each document to it's predominant topic

# In[ ]:


def associate_topics(lda_model, saved_corpus):

    appended_list = []
    for i in range(len(saved_corpus)):
        articles_topics = []
        article_number = i
        key = lda_model[saved_corpus[i]]
        conf_scores = []
        for entry in key[0]:
            conf_scores.append(entry[1])
            confidence = max(conf_scores)
            index = conf_scores.index(confidence)
            top_topic = key[0][index][0]
        article_index = list(a.keys())[article_number]
        articles_topics.extend([article_index,top_topic,confidence])
        appended_list.append(articles_topics)
    return appended_list #list containing article number, top matched topic, confidence

articles_list = associate_topics(lda_model, saved_corpus)
view_topics = lda_model.print_topics()


# Parse query text, expand query based on a pre-trained word2vec model as well as keyword identification based on inverse document frequency

# In[ ]:


def query_expansion(query):
    clean_query = []
    original_clean_query = []
    rare_term = []
    freq = []
    rare_append = []
    clean_query_stem = []
    for word in word_tokenize(query):
        clean_word = re.sub("[^a-zA-Z0-9]", '', word.lower())
        if clean_word != '' and clean_word not in stop_words:
            clean_query.append(clean_word)
    original_query = clean_query.copy()
    for word in original_query:
        word_lem = lemmatizer.lemmatize(word.lower())
        word_stem = ps.stem(word_lem)
        original_clean_query.append(word_stem)
    if len(clean_query) > 1:
        for term in original_clean_query:
            rare_term.append(term)
            try:
                freq.append(whole_word_count[term])
            except (KeyError):
                print("Vocabulary error, was there a typo?")
        freq_copy = freq.copy()
        mini = min(freq)
        rare = rare_term[freq_copy.index(mini)]
        print("Keyword 1: " + str(rare))
        rare_append.append(rare)
        rare_append.append(rare)
        rare_append.append(rare)
        #rare_append.append(rare)
        #rare_append.append(rare)
        freq.remove(min(freq))
        mini2 = min(freq)
        rare2 = rare_term[freq_copy.index(mini2)]
        print("Keyword 2: " + str(rare2))
        rare_append.append(rare2)
        rare_append.append(rare2)
        #rare_append.append(rare2)
    try:
        expanded_query = (word2vec_model.most_similar(positive=clean_query))
    except (KeyError):
        print("Vocabulary error, was there a typo?")
        expanded_query = []
    for i in range(len(expanded_query)):
       clean_query.append(expanded_query[i][0])
    for word in clean_query:
        word_lem = lemmatizer.lemmatize(word.lower())
        word_stem = ps.stem(word_lem)
        clean_query_stem.append(word_stem)
    clean_query_stem = clean_query_stem + rare_append
    return(clean_query_stem)

def query_lda(lda_model, query_stem):
    conf_scores = []
    query_corpus = diction.doc2bow(query_stem)
    query_doc = lda_model[query_corpus]
    for entry in query_doc[0]:
        conf_scores.append(entry[1])
        confidence = max(conf_scores)
        index = conf_scores.index(confidence)
        top_topic = query_doc[0][index][0]
    print('Query is assigned topic: '+str(top_topic))
    print('Topic keywords: '+str(view_topics[top_topic]))
    print('Topic assignment confidence: '+str(confidence))
    return(top_topic,confidence,query_stem)

query = 'development of coronavirus vaccines in mice models'
expanded_query = query_expansion(query)
topic,confidence,query_stem = query_lda(lda_model,expanded_query)


# Filtering out documents associated with query topic

# Using USE and cosine similarity to determine relevance
