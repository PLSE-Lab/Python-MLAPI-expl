#!/usr/bin/env python
# coding: utf-8

# ## Step 0: Latent Dirichlet Allocation ##
# 
# LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. 
# 
# * Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words.
# * LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial. 
# * It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. 

# ## Step 1: Load the dataset ##

# In[ ]:


get_ipython().system('ls -a')


# In[ ]:


'''
Load the dataset from the CSV and save it to 'data_text'
'''
import pandas as pd
data = pd.read_csv(r'../input/abcnews-date-text.csv', error_bad_lines=False);
# We only need the Headlines text column from the data
data_text = data[['headline_text']];


# In[ ]:


data.head(2)


# In[ ]:


'''
Add an index column to the dataset and save the dataset as 'documents'
'''
data_text['index'] = data_text.index
documents = data_text
documents


# In[ ]:


'''
Get the total number of documents
'''
print(len(documents))


# In[ ]:


'''
Preview a document and assign the index value to 'document_num'
'''
document_num = 4310
print("\n**Printing out a sample document:**")
print(documents[documents['index'] == document_num])


# In[ ]:


'''
Seperate the value of the headline from the document selected with 'document_num'
'''
print(documents[documents['index'] == document_num].values[0][0])


# ** Looks like this document will come under `Politics`. **

# ## Step 2: Data Preprocessing ##
# 
# We will perform the following steps:
# 
# * **Tokenization**: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
# * Words that have fewer than 3 characters are removed.
# * All **stopwords** are removed.
# * Words are **lemmatized** - words in third person are changed to first person and verbs in past and future tenses are changed into present.
# * Words are **stemmed** - words are reduced to their root form.
# 

# In[ ]:


'''
Loading Gensim and nltk libraries
'''
# pip install gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)


# In[ ]:


import nltk
#nltk.download()


# In[ ]:


'''
Lemmatizing example for a verb, noun.
'''
print(WordNetLemmatizer().lemmatize('went', pos = 'v')) # past tense to present tense


# In[ ]:


'''
Stemming example
'''
stemmer = SnowballStemmer("english")
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in plurals]
print(' '.join(singles))


# In[ ]:


'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) 
              # Remove stop words and words less than 3 characters long
              if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3]
    return result


# In[ ]:


'''
Preview a document after preprocessing
'''

doc_sample = documents[documents['index'] == document_num].values[0][0]

print("Original document: ")
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print("\n\nTokenized and lemmatized document: ")
print(preprocess(doc_sample))


# In[ ]:


'''
Save the values of the headlines into a variable 'training_headlines'
'''
training_headlines = [value[0] for value in documents.iloc[0:].values];


# In[ ]:


'''
Preview 'training_headlines'
'''
training_headlines


# In[ ]:


'''
Perform preprocessing on entire dataset using the function defined earlier and save it to 'processed_docs'
'''
processed_docs = [preprocess(doc) for doc in training_headlines]


# In[ ]:


'''
Preview 'processed_docs'
'''
processed_docs


# ## Step 3.1: Bag of words on the dataset ##

# In[ ]:


'''
Create a dictionary containing the number of times a word appears in the training set and call it 'dictionary'
'''
dictionary = gensim.corpora.Dictionary(processed_docs)


# In[ ]:


'''
Checking dictionary created
'''
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# ** Gensim doc2bow **
# 
# `doc2bow(document)`
# 
# * Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples. Each word is assumed to be a tokenized and normalized string (either unicode or utf8-encoded). No further preprocessing is done on the words in document; apply tokenization, stemming etc. before calling this method.

# In[ ]:


'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[ ]:


'''
Checking Bag of Words corpus for our sample document --> (token_id, token_count)
'''
# document_num = 4310
bow_corpus[document_num]


# In[ ]:


'''
Preview BOW for our sample preprocessed document
'''
# Here document_num is document number 4310 which we have checked in Step 2
bow_doc_4310 = bow_corpus[document_num]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                                     dictionary[bow_doc_4310[i][0]], 
                                                     bow_doc_4310[i][1]))


# ## Step 3.2: TF-IDF on our document set ##
# 
# While performing TF-IDF on the corpus is not necessary for LDA implemention using the gensim model, it is recemmended. TF-IDF expects a bag-of-words (integer values) training corpus during initialization. During transformation, it will take a vector and return another vector of the same dimensionality.
# 
# *Please note: The author of Gensim dictates the standard procedure for LDA to be using the Bag of Words model.*

# ** TF-IDF stands for "Term Frequency, Inverse Document Frequency".**
# 
# * It is a way to score the importance of words (or "terms") in a document based on how frequently they appear across multiple documents.
# * If a word appears frequently in a document, it's important. Give the word a high score. But if a word appears in many documents, it's not a unique identifier. Give the word a low score.
# * Therefore, common words like "the" and "for", which appear in many documents, will be scaled down. Words that appear frequently in a single document will be scaled up.
# 
# In other words:
# 
# * TF(w) = `(Number of times term w appears in a document) / (Total number of terms in the document)`.
# * IDF(w) = `log_e(Total number of documents / Number of documents with term w in it)`.
# 
# ** For example **
# 
# * Consider a document containing `100` words wherein the word 'tiger' appears 3 times. 
# * The term frequency (i.e., tf) for 'tiger' is then: 
#     - `TF = (3 / 100) = 0.03`. 
# 
# * Now, assume we have `10 million` documents and the word 'tiger' appears in `1000` of these. Then, the inverse document frequency (i.e., idf) is calculated as:
#     - `IDF = log(10,000,000 / 1,000) = 4`. 
# 
# * Thus, the Tf-idf weight is the product of these quantities: 
#     - `TF-IDF = 0.03 * 4 = 0.12`.

# In[ ]:


'''
Create tf-idf model object on 'bow_corpus' and save it to 'tfidf'
'''
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)


# In[ ]:


'''
Apply transformation to the entire corpus and call it 'corpus_tfidf'
'''
corpus_tfidf = tfidf[bow_corpus]


# In[ ]:


'''
Preview TF-IDF scores for our first document --> --> (token_id, tfidf score)
'''
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# ## Step 4.1: Running LDA using Bag of Words ##
# 
# We are going for 10 topics in the document corpus.
# 
# ** We will be running LDA using all CPU cores to parallelize and speed up model training.**
# 
# Some of the parameters we will be tweaking are:
# 
# * **num_topics** is the number of requested latent topics to be extracted from the training corpus.
# * **id2word** is a mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size, as well as for debugging and topic printing.
# * **workers** is the number of extra processes to use for parallelization. Uses all available cores by default.
# * **alpha** and **eta** are hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. We will let these be the default values for now(default value is `1/num_topics`)
#     - Alpha is the per document topic distribution.
#         * High alpha: Every document has a mixture of all topics(documents appear similar to each other).
#         * Low alpha: Every document has a mixture of very few topics
# 
#     - Eta is the per topic word distribution.
#         * High eta: Each topic has a mixture of most words(topics appear similar to each other).
#         * Low eta: Each topic has a mixture of few words.
# 
# * ** passes ** is the number of training passes through the corpus. For  example, if the training corpus has 50,000 documents, chunksize is  10,000, passes is 2, then online training is done in 10 updates: 
#     * `#1 documents 0-9,999 `
#     * `#2 documents 10,000-19,999 `
#     * `#3 documents 20,000-29,999 `
#     * `#4 documents 30,000-39,999 `
#     * `#5 documents 40,000-49,999 `
#     * `#6 documents 0-9,999 `
#     * `#7 documents 10,000-19,999 `
#     * `#8 documents 20,000-29,999 `
#     * `#9 documents 30,000-39,999 `
#     * `#10 documents 40,000-49,999` 

# In[ ]:


# LDA multicore 
'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                       num_topics=10, 
                                       id2word = dictionary, 
                                       passes = 2, 
                                       workers=2)


# In[ ]:


'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} Word: {}".format(idx, topic))
    print("\n")


# ### Classification of the topics ###
# 
# Using the words in each topic and their corresponding weights, what categories were you able to infer?
# 
# * 0: 
# * 1: 
# * 2: 
# * 3: 
# * 4: 
# * 5: 
# * 6: 
# * 7:  
# * 8: 
# * 9: 

# ## Step 4.2 Running LDA using TF-IDF ##

# In[ ]:


'''
Define lda model using tfidf corpus
'''
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                             num_topics=10, 
                                             id2word = dictionary, 
                                             passes = 2, 
                                             workers=4)


# In[ ]:


'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model_tfidf.print_topics(-1):
    print("Topic: {} Word: {}".format(idx, topic))
    print("\n")


# ### Classification of the topics ###
# 
# As we can see, when using tf-idf, heavier weights are given to words that are not as frequent which results in nouns being factored in. That makes it harder to figure out the categories as nouns can be hard to categorize. This goes to show that the models we apply depend on the type of corpus of text we are dealing with. 
# 
# Using the words in each topic and their corresponding weights, what categories could you find?
# 
# * 0: 
# * 1:  
# * 2: 
# * 3: 
# * 4:  
# * 5: 
# * 6: 
# * 7: 
# * 8: 
# * 9: 

# ## Step 5.1: Performance evaluation by classifying sample document using LDA Bag of Words model##
# 
# We will check to see where our test document would be classified. 

# In[ ]:


'''
Text of sample document 4310
'''
print(train_headlines[4310])


# In[ ]:


'''
Check which topic our test document belongs to using the LDA Bag of Words model.
'''
document_num = 4310
# Our test document is document number 4310
for index, score in sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# ### It has the highest probability (`x`) to be  part of the topic that we assigned as X, which is the accurate classification. ###

# ## Step 5.2: Performance evaluation by classifying sample document using LDA TF-IDF model##

# In[ ]:


'''
Check which topic our test document belongs to using the LDA TF-IDF model.
'''
document_num = 4310
# Our test document is document number 4310
for index, score in sorted(lda_model_tfidf[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# ### It has the highest probability (`x%`) to be  part of the topic that we assigned as X. ###

# ## Step 6: Testing model on unseen document ##

# In[ ]:


unseen_document = "My favorite sports activities are running and swimming."

# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# The model correctly classifies the unseen document with 'x'% probability to the X category.

# In[ ]:




