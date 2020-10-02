#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis with `nltk.sentiment.SentimentAnalyzer` and VADER tools

# ## 1. Exploring the `subjectivity` corpus

# In[ ]:


from nltk.corpus import subjectivity

subjectivity.fileids()


# In[ ]:


subjectivity.sents('plot.tok.gt9.5000')


# In[ ]:


subjectivity.sents('quote.tok.gt9.5000')


# In[ ]:


subjectivity.categories() # The mapping between documents and categories does not depend on the file structure.


# In[ ]:


subjectivity.sents(categories='obj')


# In[ ]:


subjectivity.sents(categories='subj')


# ## 2. Building and testing a classifier with `SentimentAnalyzer`

# In[ ]:


from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer # SentimentAnalyzer is a tool to implement and facilitate Sentiment Analysis.
from nltk.sentiment.util import (mark_negation, extract_unigram_feats) # mark_negation(): Append _NEG suffix to words that appear in the scope between a negation and a punctuation mark. extract_unigram_feats(): Populate a dictionary of unigram features, reflecting the presence/absence in the document of each of the tokens in unigrams.

n_instances = 100
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
len(obj_docs), len(subj_docs)


# In[ ]:


obj_docs[0]


# In[ ]:


train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]

training_docs = train_obj_docs + train_subj_docs
testing_docs = test_obj_docs + test_subj_docs

sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])


# In[ ]:


unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
len(unigram_feats)


# In[ ]:


sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


# In[ ]:


training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
training_set[0]


# In[ ]:


trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


# In[ ]:


for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))


# ## 3. Sentiment analysis with `nltk.sentiment.vader.SentimentIntensityAnalyzer`

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentences = [
    "You are a piece of shit, and I will step on you.",
    "THIS SUX!!!",
    "This kinda sux...",
    "You're good, man",
    "HAHAHA YOU ARE THE BEST!!!!! VERY FUNNY!!!"
            ]


sid = SentimentIntensityAnalyzer()

for sentence in sentences:
    print('\n' + sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')


# Above, `compound` represents the aggregated, final score.
