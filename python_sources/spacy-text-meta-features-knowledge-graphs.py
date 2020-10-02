#!/usr/bin/env python
# coding: utf-8

# 
# ## spaCy : Faster Natural Language Processing Toolkit
# 
# <br>
# 
# In this kernel, I have shared two use-cases, that can be used for different purposes: 
# 
# - 1. Building a Meta Feature Extractor Module for Text Data   
# - 2. Building a Basic Knowledge Graph using spaCy
# 
# 
# ### Quick Intro
# 
# Spacy is written in cython language, (C extension of Python designed to give C like performance to the python program). Hence is a quite fast library. spaCy provides a concise API to access its methods and properties governed by trained machine (and deep) learning models. For a detailed information, one can refer to my blog on spaCy that I wrote in 2017 :
# 
# https://www.analyticsvidhya.com/blog/2017/04/natural-language-processing-made-easy-using-spacy-%E2%80%8Bin-python/
# 
# 
# Different types of features can be generated from the text data. For example - 
# 
# Count Features : Word counts, Character Counts, Punctuation Counts, Stopwords. 
# NLP Attributes : Part of speech tags distribution, grammar relations as features, topic models 
# Word Vectors : TF-IDF, Count Vectors, Word Embeddings  
# 
# ### Uses of meta features of text:  
# 
# 1. Information Extraction   
# &nbsp;&nbsp;&nbsp;&nbsp; 1.1 Social Media  
# &nbsp;&nbsp;&nbsp;&nbsp; 1.2 Chats & Conversations  
# &nbsp;&nbsp;&nbsp;&nbsp; 1.3 News / Articles     
# 2. Machine Learning Models  
# &nbsp;&nbsp;&nbsp;&nbsp; 2.1 Classification / Regression Models     
# &nbsp;&nbsp;&nbsp;&nbsp; 2.2 Recommendation Models     
# &nbsp;&nbsp;&nbsp;&nbsp; 2.3. Building Knowledge Graphs  
# 
# 
# In this kernel, I have shared a framework that uses spacy which can be used to automatically different types of features from a given text. Importing the required libraries. 
# 

# In[ ]:


from collections import Counter 
import pandas as pd
import spacy 
nlp = spacy.load('en')


# Next, lets define the class that will use spacy functions to generate features . In the class, initialize some important variables and lists. 

# In[ ]:


feats = ['char_count', 'word_count', 'word_count_cln',
       'stopword_count', '_NOUN', '_VERB', '_ADP', '_ADJ', '_DET', '_PROPN',
       '_INTJ', '_PUNCT', '_NUM', '_PRON', '_ADV', '_PART', '_amod', '_ROOT',
       '_punct', '_advmod', '_auxpass', '_nsubjpass', '_ccomp', '_acomp',
       '_neg', '_nsubj', '_aux', '_agent', '_det', '_pobj', '_prep', '_csubj',
       '_nummod', '_attr', '_acl', '_relcl', '_dobj', '_pcomp', '_xcomp',
       '_cc', '_conj', '_mark', '_prt', '_compound', '_dep', '_advcl',
       '_parataxis', '_poss', '_intj', '_appos', '_npadvmod', '_predet',
       '_case', '_expl', '_oprd', '_dative', '_nmod']

class AutomatedTextFE:
    def __init__(self):
        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',                         'NUM', 'PRON', 'ADV', 'PART']
        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp',                          'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',                         'expl', 'oprd', 'dative', 'nmod']


# Adding a text cleaning function that removes stopwords, punctuations, performs normalization (lemmatization) and lower cases the words. 

# In[ ]:


def _spacy_cleaning(doc):
    toks = [t for t in doc if (t.is_stop == False)]
    toks = [t for t in toks if (t.is_punct == False)]
    words = [t.lemma_ for token in toks]
    return " ".join(words)


# Next, lets write the function to perform feature engineering. This function will extract following:
# 
# - word counts, char counts, stopword counts   
# - part of speech tags and their counts  
# - dependency relations among words and their counts   

# In[ ]:


def _spacy_features(df):
    df["clean_text"] = df[c].apply(lambda x : _spacy_cleaning(x))
    df["char_count"] = df[textcol].apply(len)
    df["word_count"] = df[c].apply(lambda x : len([_ for _ in x]))
    df["word_count_cln"] = df["clean_text"].apply(lambda x : len(x.split()))
    df["stopword_count"] = df[c].apply(lambda x : len([_ for _ in x if _.is_stop]))
    df["pos_tags"] = df[c].apply(lambda x : dict(Counter([_.head.pos_ for _ in x])))
    df["dep_tags"] = df[c].apply(lambda x : dict(Counter([_.dep_ for _ in x])))
    return df 


# Next, lets wrap this functions in the main class

# In[ ]:


class AutomatedTextFE:
    def __init__(self, df, textcol):
        self.df = df
        self.textcol = textcol
        self.c = "spacy_" + textcol
        self.df[self.c] = self.df[self.textcol].apply( lambda x : nlp(x))
        
        self.pos_tags = ['NOUN', 'VERB', 'ADP', 'ADJ', 'DET', 'PROPN', 'INTJ', 'PUNCT',                         'NUM', 'PRON', 'ADV', 'PART']
        self.dep_tags = ['amod', 'ROOT', 'punct', 'advmod', 'auxpass', 'nsubjpass',                         'ccomp', 'acomp', 'neg', 'nsubj', 'aux', 'agent', 'det', 'pobj',                         'prep', 'csubj', 'nummod', 'attr', 'acl', 'relcl', 'dobj', 'pcomp',                          'xcomp', 'cc', 'conj', 'mark', 'prt', 'compound', 'dep', 'advcl',                         'parataxis', 'poss', 'intj', 'appos', 'npadvmod', 'predet', 'case',                         'expl', 'oprd', 'dative', 'nmod']
        
    def _spacy_cleaning(self, doc):
        tokens = [token for token in doc if (token.is_stop == False)                  and (token.is_punct == False)]
        words = [token.lemma_ for token in tokens]
        return " ".join(words)
        
    def _spacy_features(self):
        self.df["clean_text"] = self.df[self.c].apply(lambda x : self._spacy_cleaning(x))
        self.df["char_count"] = self.df[self.textcol].apply(len)
        self.df["word_count"] = self.df[self.c].apply(lambda x : len([_ for _ in x]))
        self.df["word_count_cln"] = self.df["clean_text"].apply(lambda x : len(x.split()))
        
        self.df["stopword_count"] = self.df[self.c].apply(lambda x : 
                                                          len([_ for _ in x if _.is_stop]))
        self.df["pos_tags"] = self.df[self.c].apply(lambda x :
                                                    dict(Counter([_.head.pos_ for _ in x])))
        self.df["dep_tags"] = self.df[self.c].apply(lambda x :
                                                    dict(Counter([_.dep_ for _ in x])))
        
    def _flatten_features(self):
        for key in self.pos_tags:
            self.df["_" + key] = self.df["pos_tags"].apply(lambda x :                                                            x[key] if key in x else 0)
        
        for key in self.dep_tags:
            self.df["_" + key] = self.df["dep_tags"].apply(lambda x :                                                            x[key] if key in x else 0)
                
    def generate_features(self):
        self._spacy_features()
        self._flatten_features()
        self.df = self.df.drop([self.c, "pos_tags", "dep_tags", "clean_text"], axis=1)
        return self.df


# Finally, create a wrapper function which can be used to call the specific functions for feature generation.

# In[ ]:


def spacy_features(df, tc):
    fe = AutomatedTextFE(df, tc)
    return fe.generate_features()


# Let's look at some examples: 
# 
# ### Ted Talk Transcripts Dataset 

# In[ ]:


path = "../input/ted-talks/transcripts.csv"
df = pd.read_csv(path)[:10]
textcol = "transcript"

feats_df = spacy_features(df, textcol)
feats_df[[textcol] + feats].head()


# ### Seinfeld Chronicles Dataset

# In[ ]:


path = "../input/seinfeld-chronicles/scripts.csv"
df = pd.read_csv(path)[:10]
textcol = "Dialogue"

feats_df = spacy_features(df, textcol)
feats_df[[textcol] + feats].head()


# So these features can be passed to ML model for better classification, or can be used in recommendation engines to improve recommendations. Other use-cases can be to improve the search results, chat bot responses, and better information reterival.
# 
# ### Basics of Knowledge Graphs using spaCy
# 
# In this section, I have explained the basics of building knowledge graphs using spaCy. 
# 
# First, lets understand what are knoweldge graphs. 
# 
# - What are knowlege graphs ? 
# > Knowledge stored in a graph form. The knowledge is captured in entities, attributes, relationships. The Nodes represents entities, NodeLabels represents attributes, and Edges represents Relationships. 
# 
# - Example:  
# > Chris Nolan (Director, Producer, person) ---> born in  ----> London (place) ---> Director of  ----> Interstellar (Movie) ---> shooted in  -----> Iceland (place)  
# 
# - Source of information for building knowledge graphs: 
# > Structured Text: Wikipedia, Dbpedia  
# > Unstructured Text: Social Media, Blogs, Images, Videos, Audios 
# 
# #### Main ideas for building knowlege graphs
# 
# - Entity Extraction   
# In this step, the aim is to extract right entities from the text data. spaCy provides NER (Named Entity Recognition) which can be used for this purpose.  
# 
# - Relationship Extraction    
# In this step, the aim is to identify the relationship between the sentences / entities. Again, by using spaCy one can extract the grammar relations between two words / entities.  
# 
# - Relationship Linking    
# The hard part of knowlege graphs is to identify what kind of relationship exists between the two entities. The idea is to add the contextual sense to the relationship. 
# 
# Let's look at very high level implementation of this idea using spacy. Lets load a news dataset. 

# In[ ]:


path = "../input/news-aggregator-dataset/uci-news-aggregator.csv"
df = pd.read_csv(path)[:1500]
df["spacy_title"] = df["TITLE"].apply(lambda x : nlp(x))


# Now, we will define a grammar pattern / part of speech pattern to identify what type of relations we want to extract from the data. 
# 
# Let's we are interested in finding an action relation between two named entities. so we can define a pattern using part of speech tags as : 
# 
# Proper Noun - Verb - Proper Noun

# In[ ]:


pos_chain_1 = "NNP-VBZ-NNP"


# Using spaCy, we can now iterate in text and identify what are the relevant triplets (governer, relation, dependent) or in other terms, what are the entities and relations.

# In[ ]:


df["named_entities"] = df["spacy_title"].apply(lambda x : x.ents)
for i, r in df.iterrows():
    pos_chain = "-".join([d.tag_ for d in r['spacy_title']])
    if pos_chain_1 in pos_chain:
        if len(r["named_entities"]) == 2:
            print (r["TITLE"])
            print (r["named_entities"])
            print ()
    


# So from these examples, one can see different entities and relations for example: 
# 
# - Honda --- **restructures** ---> US operations  
# - Carl Icahn --- **slams** ---> eBay CEO
# - Google --- **confirms** ---> Android SDK 
# - GM --- **hires** ---> Lehman Brothers 
# 
# References : https://kgtutorial.github.io/
