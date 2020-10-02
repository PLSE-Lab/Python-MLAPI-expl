#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This is the final portion of an effor to optimize some parameters for a Latent Dirichlet Allocation (LDA) topic model. We break up this effort into multiple steps in order to speed up testing and making changes. This also uses less memory and is less likely to time out.  There are many parameters we could attempt to optimize.  However, due to processing constraints we limit ourselves to the number of topics and the number of features (word or bigrams).
# 
# A topic model learns a topic-feature matrix of abstract topics and features (word or ngrams) and a document-topic matrix of documents and topics, from a document-feature matrix of documents and features. From this factorization we achieve statistical feature vectors for each topic and topic vectors for each document in the training corpus. We can then find topic vectors for the questions we would like to ask the corpus of documents. We will use the closest matching documents in the CORD-19 dataset in an attempt to answer the task questions. LDA assumes a Dirichlet prior on topic-feature and document-topic distributions. In other words it assumes each topic is defined by and small collection of words or ngrams and that each documnent consists of a small number of topics.
# 
# The advantage of trying to find some configurations of LDA training parameters with better coherence scores is that the resulting topic model might perform better under this tasks queries than they would from random parameter selection.  However, coherence is not a human measure (involving humans would be expensive and negate the benifit of first step automation) and we only varied a select few parameters (there are more parameter that could impact human perception of the results).
# 
# We will be relying on paper titles and PIDs as answers to the task's queries.  The papers have the advantage of being written by humans (rather than a generated summary) and therefore is assumed to be understandable.  However, it is up to the person asking the question to then go to the referenced papers and determine if their question is answered.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Additional Imports 
# Gensim imported for natural language processing tasks; matplotlib imported for plotting; pickle used for opening results saved from previous steps;

# In[ ]:


import gensim
import matplotlib
import pickle
import gc
import pyLDAvis
import pyLDAvis.gensim


# # Definitions from preprocessing kernel
# 
# Including definitions from the preprocessing step here.  They are not used in this kernel.  The output from the preprocessing step is used as input to this notebook.
# 
# **Definition for reading the json files:**
# The purpose of these definitions is to read all json files in a path and return a dictionary containing entries for the paper's id, title, abstract, and body text. These json files are the papers that we have full body text for.  Now that we are up to 50k+ documents we keep running out of memory.  We will limit ourselves to abstracts under the assumption that abstracts should summarize the entire paper including results.
# 
# **Bigram and trigram definitions:**
# Below are definitions for making bigrams (sequences of 2 words) and trigrams (collections of 3 words). We will be skipping trigrams, but the definition is included in case it is desired later.
# 
# **Definitions for tokenizing documents:**
# Reading in the documents was done inside the same function that tokenizes the documents so that the raw documents can go out of scope. This is done to reduce memory usage. We will only be looking at papers that have a json file. We will be saving the resulting dictionary for use later.

# In[ ]:


def readarticle(filepath):
    paperdata = {"paper_id" : None, "title" : None, "abstract" : None}
    with open(filepath) as file:
        filedata = json.load(file)
        paperdata["paper_id"] = filedata["paper_id"]
        paperdata["title"] = filedata["metadata"]["title"]
                
        if "abstract" in filedata:
            abstract = []
            for paragraph in filedata["abstract"]:
                abstract.append(paragraph["text"])
            abstract = '\n'.join(abstract)
            paperdata["abstract"] = abstract
        else:
            paperdata["abstract"] = []
    return paperdata

def read_multiple(jsonfiles_pathnames):
    papers = {"paper_id" : [], "title" : [], "abstract" : []}
    for filepath in jsonfiles_pathnames:
        paperdata = readarticle(filepath)
        if len(paperdata["abstract"]) > 0: 
            papers["paper_id"].append(paperdata["paper_id"])
            papers["title"].append(paperdata["title"])
            papers["abstract"].append(paperdata["abstract"])
            #papers["body_text"].append(paperdata["body_text"])
            #print("not none")
        #else:
            #print("none")
    print(len(papers["paper_id"]))
    print(len(papers["title"]))
    print(len(papers["abstract"]))
    return papers

def make_bigram(tokenized_data, min_count = 5, threshold = 100):
    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
    #after Phrases a Phraser is faster to access
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    gc.collect()
    return bigram

def make_trigram(tokenized_data, min_count = 5, threshold = 100):
    bigram_phrases = gensim.models.Phrases(tokenized_data, min_count = min_count, threshold = threshold)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[tokenized_data], threshold = 100)
    #after Phrases a Phraser is faster to access
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    gc.collect()
    return trigram

def readjson_retbodytext(jsonfiles_pathnames):
    print("reading json files")
    documents = read_multiple(jsonfiles_pathnames)
    print("writing documents dictionary to output for use in another kernel")
    with open("documents_dict.pkl", 'wb') as f:
        pickle.dump(documents, f)
    print("done writing documents dict.  Format is paper_id, title, body_text")
    gc.collect()
    return documents["abstract"]
    
def open_tokenize(jsonfiles_pathnames):
    
    body_text = readjson_retbodytext(jsonfiles_pathnames)
    
    print("removing stopwords, steming, and tokenizing.  This is expensive")
    tokenized_documents = gensim.parsing.preprocessing.preprocess_documents(body_text)
    print("done preprocessing documents. now writing to output to be used in another documents")
    with open("tokenized_documents.pkl", 'wb') as f:
        pickle.dump(tokenized_documents, f)
    print("done writing file")
    
    gc.collect()
    return tokenized_documents


# # Preprocessing steps
# Code available in the public kernel that produced the preprocess input.
# 
# 1. **Look at metadata and print information**
# 2. **Collect json file paths**
# 3. **Remove stopwords, stem, tokenize documents, and save.**  Stopwords are words common to all text, such as "and" and "the," and therefore we remove them. Stemming reduces words to their root form. For example, "infected" and "infecting" will both be reduced to "infect." Tokenization converts the body text of each paper into a vector of whitespace-separated text (words). These tokens will be treated as semantic features, particularly after stemming. When we create the document-feature matrix (term frequency corpus) the exact text of the word (or bigram) will no longer be observed, as these features define the row vectors of that matrix. Here, we save the tokenized documents.
# 4. **Create bigram model and save.  In this section we create and save the bigram model.** This just contains the word pair. We will have to run the tokenized documents through the model in order to create a bigram tokenized version of the documents. We will do that in another step therefore we save the model here.

# # Definitions for "Try LDA Parameters" kernel
# In this portion of our effort to tune some of the LDA parameters, we will try configurations of several hyperparameters.  The parameters we will vary are: ngram (1 or 2), number of features (words or bigrams) in the feature distribution vectors of the topics, and number of topics to factor for when creating the LDA model (topic-feature matrix).
# 
# **Create Dictionary:**
# This is the definition for creating a dictionary for our corpus. Number of features (word stems of ngrams) will be the variable optimized for using this function.
# 
# **Create term frequency corpus:**
# Definition for creating the corpus from the dictionary and documents.
# 
# **Try model configuration:**
# Definition to train a LDA Model and the compute the coherence score.
# 
# **Loop through number of topics:**  Given a set ngram (1 or 2) and feature length, loop through number of topics given the start, stop, and step bounds.  For each n_topics build an LDA topic model, calculate coherence, and return coherence list.
# 
# **Objective function for differential evolution optimization:**
# Originally we wanted to use differential evolution to optimize LDA parameters.  However, we were unable to use this technique because each mumber of the population requires compute and memory resources to evaluate the objective function. 
# 
# Differential evolution uses a population to try many different parameters, removes the poor performing population members, makes combinations of the more successfull population members, performs some mutations, and then repeats until some stopping condintion (number of evolutions, or no improvement after a given number of steps).  In this way it hopes to achieve good non-linear optimization.  Differential evolution is similar to genetic algorithms, but skips the chromosome string encoding and acts directly on the variables.   We would need to reserve extra compute resources for this technique.  Note that we need to define a vector of tuples for the bounds, and that some parameters need to be converted to to int (uint) because this scipy function works with floats only.

# In[ ]:


#no_n_below should be uint, ex: no_n_below = 3 or no_n_below = 5
#no_freq_above should be float [0,1], ex: no_freq_above = 0.5
#n_feats should be uint, ex: n_feats = 1024 or n_feats = 2048
def create_dictionary(tokenized_documents, n_feats, no_n_below = 3, no_freq_above = 0.5):
    print("creating dictionary")
    id2word_dict = gensim.corpora.Dictionary(tokenized_documents)
    print("done creating dictionary")
    
    print("prior dictionary len %i" % len(id2word_dict))
    id2word_dict.filter_extremes(no_below = no_n_below, no_above = no_freq_above, keep_n = n_feats, keep_tokens = None)
    print("current dictionary len %i" % len(id2word_dict))
    
    return id2word_dict

def corpus_tf(id2word_dict, tokenized_documents):
    return [id2word_dict.doc2bow(document) for document in tokenized_documents]

def try_parameters(tokenized_documents, n_feats, n_topics):
    id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)
    tfcorpus = corpus_tf(id2word_dict, tokenized_documents)
    print("training lda model with %i features and %i topics" % (n_feats, n_topics))
    lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)
    coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")
    coherence_score = coherence_model.get_coherence()
    print("coherence for unknown ngram with %i features and %i topics: %f" % (n_feats, n_topics, coherence_score))
    gc.collect()
    return coherence_score

def loop_lda(tokenized_documents, 
                     tfcorpus, 
                     id2word_dict,
                     start, #suggest 2 or something
                     stop, # suggest 20 or similar
                     step,
                     per_word_topics = False): #compute list of topics for each word
    topic_counts = []
    coherence_scores = []
    for n_topics in range (start, stop, step):
        lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = per_word_topics)
        coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        topic_counts.append(n_topics)
        print("coherence of %f with %i topics" % (coherence_score, n_topics))
              
    return topic_counts, coherence_scores;
        
def loop_ntopics_lda(tokenized_documents, n_feats, start, stop, step):
    id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)
    tfcorpus = corpus_tf(id2word_dict, tokenized_documents)
    topic_counts, coherence_scores = loop_lda(tokenized_documents, tfcorpus, id2word_dict, start, stop, step)
    gc.collect()
    return topic_counts, coherence_scores

ngram_bounds = (1,2)
n_feats_bounds = (512,2048)
n_topics_bounds = (1,20)

bounds = [ngram_bounds, n_feats_bounds, n_topics_bounds]

def lda_objective(X, tokenized_documents, tokenized_bigram_documents):
    ngram = int(round(X[0])) #bound should be [1,2]
    n_feats = int(round(X[1])) #bounds should be [512, 2048]
    n_topics = int(round(X[2])) #bouns should be [1,20]
    
    if ngram == 2:
        documents = tokenized_bigram_documents
        type_string = "tokenized_bigram_documents"
    else:
        documents = tokenized_documents
        type_string = "tokenized_documents"

    print("creating dictionary with %s for: %i %i %i" % (type_string, ngram, n_feats, n_topics))
    id2word_dict = create_dictionary(documents, n_feats = n_feats)

    print("done creating dictionary.  creating corpus for: %i %i %i" % (ngram, n_feats, n_topics))
    tfcorpus = corpus_tf(id2word_dict, documents)

    print("done creating corpus.  Building model for: %i %i %i" % (ngram, n_feats, n_topics))
    lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)

    print("calculating coherence for: %i %i %i" % (ngram, n_feats, n_topics))
    coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = documents, dictionary = id2word_dict, coherence = "c_v")
    coherence = coherence_model.get_coherence()
    #we want to MAX coherence.  but we will be using a 
    value2minimize = 1 - coherence
    return value2minimize


# # Try LDA Parameters steps
# Code available in the public kernel that produced the "try-lda-parameters" input.
# Try ngrams (1 and 2), number of features, and number of topics
# 1.  Open tokenized documents from preprocesing steps.  This will serve as our 1gram.
# 2.  Set features to 512 and loop through [2,20] LDA topics and collect coherence scores. Set features to 1024 and loop through [2,20] LDA topics and collect coherence scores.
# 3.  Open bigram model.  Run tokenized document through the bigram model in order to obtain tokenized bigram documents.
# 4.  Set features to 256 and loop through [2,20] LDA topics and collect coherence scores. Set features to 512 and loop through [2,20] LDA topics and collect coherence scores.
# 5.  Save results for use in this notebook.
# 

# # Definitions for query topic vectors and ranking 
# **Topic distribution:**
# This definition takes a provided string, tokenizes it with the provided dictionary (either word or bigram based), and then returns a topic distribution vector using a provided LDA topic model.
# **Print top 3 matches:**
# this takes a query in the form of a topic distribution and a gensim similarity matrix designed to efficiently calculate cosine similarities and then creates a ranked list of indices for the matches.  The indices are used along with a provided dictionary of the raw data in order to print the title and abstract of the top three matching papers.

# In[ ]:


def topic_distribution(query_string, id2word_dict, lda_model):
    tokenized_query = gensim.parsing.preprocessing.preprocess_string(query_string)

    print("tokens in query: %i" % (len(tokenized_query)))
    print(tokenized_query)
    
    vectorized_query = id2word_dict.doc2bow(tokenized_query)
    
    return lda_model[vectorized_query]    #query topic vector (distribution)
    
def corpus_similarities_print3(query_topicvec, index, documents_dict):
    similarities = index[query_topicvec]
    ranked_indices = sorted(enumerate(similarities), key = lambda item: -item[1])
    #papers = {"paper_id" : [], "title" : [], "abstract" : []}
    
    document_pids = documents_dict["paper_id"]
    document_titles = documents_dict["title"]
    #document_abstracts = documents_dict["abstract"]
    
    print(ranked_indices[0][0])
    topdex = ranked_indices[0][0]
    second = ranked_indices[1][0]
    third = ranked_indices[2][0]
    
    print("\nTOP RESULT TITLE: %s" % (document_titles[topdex]))
    print("TOP RESULT PID: %s" % (document_pids[topdex]))
    print(second)
    print("\nSecond RESULT TITLE: %s" % (document_titles[second]))
    print("Second RESULT PID: %s" % (document_pids[second]))
    print(third)
    print("\nThird RESULT TITLE: %s" % (document_titles[third]))
    print("Third RESULT PID: %s" % (document_pids[third]))
    gc.collect()


# # Open and view results from parameter search
# Here we will view the results from our various combinations of ngram (1 and 2), number of features, and number of topics

# In[ ]:


coherence_results_path = "/kaggle/input/try-lda-parameters/coherence_dict.pkl"
with open(coherence_results_path, "rb") as f:
    coherence_results = pickle.load(f)

#including the defintion for this dictionary in the comments for our reference

#coherence_dict = {"topic_counts" : topic_counts,
#                  "coherence_1gram_512features" : coherence_1gram_512features,
#                  "coherence_1gram_1024features" : coherence_1gram_1024features,
#                  "coherence_2gram_256features" : coherence_2gram_256features, 
#                  "coherence_2gram_512features" : coherence_2gram_512features,
#                  "coherence_2gram_1024features" : coherence_2gram_1024features}

start = coherence_results["topic_counts"][0]
length = len(coherence_results["topic_counts"])
stop = length + start

print(start)
print(length)
print(stop)

x = range(start, stop, 1)
matplotlib.pyplot.plot(x, coherence_results["coherence_1gram_512features"], label = "1gram_512feats")
matplotlib.pyplot.plot(x, coherence_results["coherence_1gram_1024features"], label = "1gram_1024feats")
matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_256features"], label = "2gram_256feats")
matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_512features"], label = "2gram_512feats")
matplotlib.pyplot.plot(x, coherence_results["coherence_2gram_1024features"], label = "2gram_1024feats")

matplotlib.pyplot.xlabel("Number of topics")
matplotlib.pyplot.ylabel("Coherence score")

matplotlib.pyplot.title("Coherence values for for gram 1 and 2, and features 256, 512, and 1024")
matplotlib.pyplot.legend()
matplotlib.pyplot.show()


# # Choose ngram, number of features, and number of topics
# From the above results we choose our ngram (words or bigram), number of featurs, and number of topics.  It would be preferable to save the best performing model from the previous testing and load it.
# * We will select the tokenized bigram documents
# * We will select 512 features
# * We will use 18 topics

# In[ ]:


tokenized_path = "/kaggle/input/preprocess-cord19/tokenized_documents.pkl"
print("opening %s" % str(tokenized_path)) 
with open(tokenized_path, "rb") as f:
    tokenized_documents = pickle.load(f)
print("done opening tokenized documents.  Optimizing")

bigram_path = "/kaggle/input/preprocess-cord19/bigram_model.pkl"
print("opening %s" % str(bigram_path))
with open(bigram_path, "rb") as f:
    bigram_model = pickle.load(f)
print("creating bigram documents")
tokenized_document = [bigram_model[document] for document in tokenized_documents]
print("done retrieving documents. lets optimize")

n_feats = 512
n_topics = 18
id2word_dict = create_dictionary(tokenized_documents, n_feats = n_feats)
tfcorpus = corpus_tf(id2word_dict, tokenized_documents)
print("training lda model with %i topics" % (n_topics))
lda_model = gensim.models.ldamodel.LdaModel(corpus = tfcorpus, num_topics = n_topics, id2word = id2word_dict, per_word_topics = False)
coherence_model = gensim.models.CoherenceModel(model = lda_model, texts = tokenized_documents, dictionary = id2word_dict, coherence = "c_v")
coherence_score = coherence_model.get_coherence()
print("Achieved coherence of: %f" % (coherence_score))


# # Visualize topic separation
# We will use pyLDAvis to project the topic-feature vectors onto two principle components and visualize the separation and overlap of the topics on these two axes.

# In[ ]:


pyLDAvis.enable_notebook()
visualization = pyLDAvis.gensim.prepare(lda_model, tfcorpus, id2word_dict)
visualization


# # Create similarity index.
# This gensim matrix is designed to easily calculate cosine similary with-respect-to entry indices.  It can be sorted to 

# In[ ]:


#build index for document similarity.  the similarity method used will be cosine similarity
print("creating index")
index = gensim.similarities.MatrixSimilarity(lda_model[tfcorpus])
print("done creating index")

documents_path = "/kaggle/input/preprocess-cord19/documents_dict.pkl"
with open(documents_path, "rb") as f:
    documents_dict = pickle.load(f)
print("done opening raw documents")


# # Broad questions
# We will start will the broad scope questions for this task.

# # "What is known about transmission, incubation, and environmental stability?"

# In[ ]:


query = "What is known about transmission, incubation, and environmental stability?"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "What do we know about natural history, transmission, and diagnostics for the virus? "

# In[ ]:


query = "What do we know about natural history, transmission, and diagnostics for the virus? "
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "What have we learned about infection prevention and control?"

# In[ ]:


query = "What have we learned about infection prevention and control?"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # Specific Questions
# These are the specific bullet points to be addressed in this task

# # "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery"

# In[ ]:


query = "Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Prevalence of asymptomatic shedding and transmission (e.g., particularly children)."

# In[ ]:


query = "Prevalence of asymptomatic shedding and transmission (e.g., particularly children)."
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Seasonality of transmission."

# In[ ]:


query = "Seasonality of transmission."
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)."

# In[ ]:


query = "Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding)."
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)."

# In[ ]:


query = "Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood)."
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)."

# In[ ]:


query = "Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic)."
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Natural history of the virus and shedding of it from an infected person"

# In[ ]:


query = "Natural history of the virus and shedding of it from an infected person"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Implementation of diagnostics and products to improve clinical processes"

# In[ ]:


query = "Implementation of diagnostics and products to improve clinical processes"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Disease models, including animal models for infection, disease and transmission"

# In[ ]:


query = "Disease models, including animal models for infection, disease and transmission"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Tools and studies to monitor phenotypic change and potential adaptation of the virus"

# In[ ]:


query = "Tools and studies to monitor phenotypic change and potential adaptation of the virus"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Immune response and immunity"

# In[ ]:


query = "Immune response and immunity"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"

# In[ ]:


query = "Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings"

# In[ ]:


query = "Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)


# # "Role of the environment in transmission"

# In[ ]:


query = "Role of the environment in transmission"
query_topicvec = topic_distribution(query, id2word_dict, lda_model)
corpus_similarities_print3(query_topicvec, index, documents_dict)

