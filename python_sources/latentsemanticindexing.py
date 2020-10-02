# This script creates topic vectors using the LSI model in gensim. These
# topic vectors can be used to represent the documents (emails) quantitatively
# such that we can perform analyses such as querying, checking similarity, etc.
#
# Note that since the total number of documents we use is really small, the validity
# of this approach is, of course, debatable but I thought it was worth sharing.

import re
import numpy as np
import pandas as pd
import networkx as nx

from gensim import corpora, models, similarities, utils

# It's yours to take from here!
usecols = ["DocNumber", "MetadataSubject", "MetadataTo", "MetadataFrom", "ExtractedBodyText"]
emails = pd.read_csv("../input/Emails.csv", usecols=usecols,)

# remove emails when number of characters < 250
emails = emails[emails["ExtractedBodyText"].str.len() > 250]

# convert emails to list for convenience
emails_body_text = emails["ExtractedBodyText"].tolist()


def cleanEmailText(text):
    
    # Removes any accents
    text = utils.deaccent(text)
    
    # Replace hypens with spaces
    text = re.sub(r"-", " ", text)
    
    # Removes dates
    text = re.sub(r"\d+/\d+/\d+", "", text)
    
    # Removes times
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)
 
    # Removes email addresses
    text = re.sub(r"[\w]+@[\.\w]+", "", text)
    
    # Removes web addresses
    text = re.sub(r"/[a-zA-Z]*[:\/\/]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)
    
    # Remove any bad characters
    clndoc = ''
    
    for eachLetter in text:
        if eachLetter.isalpha() or eachLetter == ' ':
            clndoc += eachLetter
        
    text = ' '.join(clndoc.split())
    
    return text
    
# let's iterate through the list and clean the text
for i, item in enumerate(emails_body_text):
    emails_body_text[i] = cleanEmailText(item)
    
# get emails in a format that gensim can turn into a dictionary and corpus
texts = [ [word for word in document.lower().split() ] for document in emails_body_text]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# perform tf-idf
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# LSI model
num_topics = 10

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
corpus_lsi = lsi[corpus_tfidf]

# print each topic to screen
for i in range(num_topics):
    print("Topic",i)
    print(lsi.show_topic(i))