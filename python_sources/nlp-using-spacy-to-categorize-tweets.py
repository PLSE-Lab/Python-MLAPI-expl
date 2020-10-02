#!/usr/bin/env python
# coding: utf-8

# Hello everyone, if you like my kernel don't forget to vote. 

# * <a id='section1' href=#sec1>Inspection of the data. Plotting with Seaborn </a>
# 
# * <a id='section2' href=#sec2>Introduction to SpaCy</a>
# 
# * <a id='section3' href=#sec3> Building the model to classify the tweets</a>

# In[ ]:


#importing necesary tools to work with
import pandas as pd
import spacy
from spacy import displacy
import seaborn as sns


# <div id='sec1'>1-Inspection of the datasets.  I'm going to read both train and test data. I will focus on train dataset</div>

# In[ ]:


dtrain=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
sample=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
dtest=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# In[ ]:


#inspection of train data
dtrain.info()


# In[ ]:


sns.countplot(y='target',data=dtrain,palette='Set3')#are there more tweets classified as 0 or as 1?


# There are more tweets classified as non disaster(0). 

# In[ ]:


#from where are these tweets?
sns.countplot(y='location',data=dtrain,palette='Set3',order=dtrain['location'].value_counts().iloc[:6].index)


# This is the top-5 of locations where someone sent a tweet. We can improve this graph.

# In[ ]:


#this is for replace the cities by their country
dtrain['location']=dtrain['location'].replace(['United States','New York','Los Angeles','Los Angeles, CA', 'Washington, DC'],'USA')
dtrain['location']=dtrain['location'].replace(['London'],'UK')
dtrain['location']=dtrain['location'].replace(['Mumbai'],'India')


# In[ ]:


sns.countplot(y='location',data=dtrain,palette='Set3',order=dtrain['location'].value_counts().iloc[:6].index)


# In[ ]:


sns.countplot(y='keyword',data=dtrain,palette='Set2',order=dtrain['keyword'].value_counts().iloc[:3].index)#Top 3 of keywords more used


# <div id='sec2'>2-Introduction to spaCy.</div>

# SpaCy is an alternative to using NLTK. They are pretty similar but spaCy has cool utilities like these ones:

# First we have to load the english model for spaCy.

# In[ ]:


nlp = spacy.load("en_core_web_sm")


# In[ ]:


#First one: Entity Recognition
doc=nlp(dtrain['text'][58])
displacy.render(doc,style='ent')


# In[ ]:


doc=nlp(dtrain['text'][10])
displacy.render(doc,style='ent')


# In[ ]:


#linguistic annotations
tokenized_text = pd.DataFrame()
#describe the words in the sentence before
for i, token in enumerate(doc):
    tokenized_text.loc[i, 'text'] = token.text
    tokenized_text.loc[i, 'type'] = token.pos_
    tokenized_text.loc[i, 'lemma'] = token.lemma_,
    tokenized_text.loc[i, 'is_alphabetic'] = token.is_alpha
    tokenized_text.loc[i, 'is_stop'] = token.is_stop
    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct
    tokenized_text.loc[i, 'sentiment'] = token.sentiment
    
    

tokenized_text[:30]


# * text: the text of the word
# 
# * type: type of the word. Is it an adverb? Is it a preposition?
# 
# * lemma: the base form of the word.
# 
# * is_alpha: does the word consist of alphabetic characters? 
# 
# * is_stop: is the word part of a stop list?
# 
# * is_puntuaction: is the word puntuaction?
# 
# * sentiment: A scalar value indicating the positivity or negativity of the token.
# 
# 

# In[ ]:


#dependency parser- see the relations between the words 
displacy.render(doc,style='dep',jupyter='true')


# In[ ]:


#if you don't understand a tag displayed
spacy.explain('ADP')


# <div id='sec3'>3-Building the model to classify the tweets</div>

# For building the model I am following the guide by Susan Li: https://towardsdatascience.com/machine-learning-for-text-classification-using-spacy-in-python-b276b4051a49. 

# In[ ]:


import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[ ]:


#A-FIRST STEP: TOKEN THE DATA. We are going to remove stopwords and puntuaction from each sentence.

# Create a list of punctuation marks
punctuations = string.punctuation

# Create a list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer
tokenizer = English()

# Creating a tokenizer function with the ones defined before
def text_tokenizer(sentence):
    # Creating the token object
    tokens = tokenizer(sentence)

    # Lemmatizing each token if it is not a pronoun and converting each token into lowercase
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens ]
    
    # Remove stop words
    tokens = [ word for word in tokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return tokens


# In[ ]:


#we want to clean more our data. For that, we will be creating a class predictors which inherits from sklearn TransformerMixin
from sklearn.base import TransformerMixin


# In[ ]:


# Custom transformer using spaCy
class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text


# In[ ]:


#CountVectorizer converts a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
vectorizer = CountVectorizer(tokenizer = text_tokenizer, ngram_range=(1,1))


# In[ ]:


#Now we need to split our train dataset into train and validation data
X = dtrain['text'] 
y = dtrain['target'] 


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_vald, y_train, y_vald = train_test_split(X, y, test_size=0.15)


# In[ ]:


from sklearn.pipeline import Pipeline
# we are going to use Linear Support Vector Classification
from sklearn.svm import LinearSVC
classifier = LinearSVC()

# Create a pipeline
pipeline = Pipeline([("cleaner", CleanTextTransformer()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# model generation
pipeline.fit(X_train,y_train)


# In[ ]:


from sklearn import metrics
# predict the X_vald
predictions = pipeline.predict(X_vald)

# model Accuracy
print("Linear Support Vector Classification Accuracy:",metrics.accuracy_score(y_vald, predictions))


# In[ ]:


#the code bellow is to create the submission file with the predictions made using the test dataset


# In[ ]:


predictionsFinal=pipeline.predict(dtest['text'])


# In[ ]:


sample['target'] = predictionsFinal


# In[ ]:


sample


# In[ ]:


sample.to_csv("submissionNLP.csv", index=False)

