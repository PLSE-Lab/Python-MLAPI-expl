#!/usr/bin/env python
# coding: utf-8

# # Natural language preprocessing

# In[ ]:


import nltk


# In[ ]:


from nltk.corpus import brown


# In[ ]:


brown.words()


# In[ ]:


brown.categories()


# In[ ]:


print(len(brown.categories()))


# In[ ]:


data = brown.sents(categories = 'adventure')


# In[ ]:


len(data)


# In[ ]:


data = brown.sents(categories = 'fiction')


# In[ ]:


data


# In[ ]:


" ".join(data[1])


# ## bag of words pipeline
# get the data/corpus
# 
# tokenisation,stopward removal
# 
# stemming
# 
# building a vocab
# 
# classification

# ## 1 Tokenisation & Stopword Removal

# In[ ]:


document = '''it was a very pleasent day. the weather was cool and there were light showers.
i went to the market to buty some fruits'''

sentence = "Send all the 50 document related to chapters 1,2,3,4 at prateek@cb.com"


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize


# In[ ]:


sents = sent_tokenize(document)
print(sents)


# In[ ]:


print(len(sents))


# In[ ]:


sentence.split()


# In[ ]:


word = word_tokenize(sentence)


# In[ ]:


word


# ## Stopwords

# In[ ]:


from nltk.corpus import stopwords
sw = set(stopwords.words('english'))


# In[ ]:


print(sw)


# In[ ]:


def remove_stopwords(text,stopwords):
    useful_words = [w for w in text if w not in stopwords]
    return useful_words


# In[ ]:


text = "i am not bothered about her very much".split()
useful_text = remove_stopwords(text,sw)
print(useful_text)


# In[ ]:


'not' in sw


# ## Tokenization using Regular Expression 

# In[ ]:


sentence = "Send all the 50 document related to chapters 1,2,3,4 at prateek@cb.com"
from nltk.tokenize import RegexpTokenizer


# In[ ]:


tokenizer = RegexpTokenizer('[a-zA-Z]')
useful_text = tokenizer.tokenize(sentence)


# In[ ]:


useful_text


# In[ ]:


tokenizer = RegexpTokenizer('[a-zA-Z@.]+')
useful_text = tokenizer.tokenize(sentence)


# In[ ]:


useful_text


# ## Stemming
# 1 process that transform particular words (verbs, plurals) into their radical form
# 
# 2 preserve the semantics of the sentence without increasing the number of unique tokens
# 
# examples - jumps, jumping,jumped ,jump ==>jump

# In[ ]:


text   = """Foxes love to make jumpes. the quick brown fox was seen jumping over the
lovely dog from a 6th feet high wall"""


# In[ ]:


from nltk.stem.snowball import SnowballStemmer,PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
#Snowball Stemmer ,Porter ,Lancaster Stemmer


# In[ ]:


ps = PorterStemmer()


# In[ ]:


ps.stem('jumping')


# In[ ]:


ps.stem('lovely')


# In[ ]:


ps.stem('loving')


# In[ ]:


ps.stem('jumped')


# In[ ]:


# let's work with snowball stemmer
ss = SnowballStemmer('english')


# In[ ]:


ss.stem('jumping')


# In[ ]:


'''# Lemitization
from nltk.stem import WordNetLemmatizer
wn = WordNetLemmatizer()
wn.lemmatize('jumping')'''


# ## Building a vocab &vectorization

# In[ ]:


# Sample Corpus - Contains 4 Documents, each document can have 1 or more sentences
corpus = [
        'Indian cricket team will wins World Cup, says Capt. Virat Kohli. World cup will be held at Sri Lanka.',
        'We will win next Lok Sabha Elections, says confident Indian PM',
        'The nobel laurate won the hearts of the people.',
        'The movie Raazi is an exciting Indian Spy thriller based upon a real story.'
]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv = CountVectorizer()


# In[ ]:


vectorized_corpus = cv.fit_transform(corpus)


# In[ ]:


vectorized_corpus = vectorized_corpus.toarray()


# In[ ]:





# In[ ]:


vectorized_corpus[0]


# In[ ]:


print(cv.vocabulary_)


# In[ ]:


len(cv.vocabulary_.keys())


# In[ ]:


#reverse maping
numbers = vectorized_corpus[2]
numbers


# In[ ]:


s = cv.inverse_transform(numbers)
print(s)


# ## Vectorization with Stopword Removal

# In[ ]:


def myTokenizer(document):
    words = tokenizer.tokenize(document.lower())
    # Remove Stopwords
    words = remove_stopwords(words,sw)
    return words
    


# In[ ]:


#myTokenizer(sentence)
#print(sentence)


# In[ ]:


cv = CountVectorizer(tokenizer = myTokenizer)


# In[ ]:


vectorized_corpus = cv.fit_transform(corpus).toarray()


# In[ ]:


print(vectorized_corpus)


# In[ ]:


print(len(vectorized_corpus[0]))


# In[ ]:


cv.inverse_transform(vectorized_corpus)


# In[ ]:



# For Test Data
test_corpus = [
        'Indian cricket rock !',        
]


# In[ ]:


cv.transform(test_corpus).toarray()


# ### More ways to Create features 
# 1 unigram -every word as a feature 
# 
# 2 Bigram 
# 
# 3 Trigram
# 
# 4 n-grams
# 
# 5 TF-IDF Normalisation 

# In[ ]:


sent_1  = ["this is good movie"]
sent_2 = ["this is good movie but actor is not present"]
sent_3 = ["this is not good movie"]


# In[ ]:


cv = CountVectorizer(ngram_range=(1,3))


# In[ ]:


docs = [sent_1[0],sent_2[0]]
cv.fit_transform(docs).toarray()


# In[ ]:


cv.vocabulary_


# ## Tf-idf Normalisation
# 1 Avoid features that occur very often,because they contain less information 
# 
# 2 information decreases as the number of occurrences increases across different type of document 
# 
# 3 so we define another term - term-document-frequency which associates a weight with every term

# In[ ]:


sent_1  = "this is good movie"
sent_2 = "this was good movie"
sent_3 = "this is not good movie"

corpus = [sent_1,sent_2,sent_3]


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer()


# In[ ]:


vc = tfidf.fit_transform(corpus).toarray()


# In[ ]:



print(vc)


# In[ ]:


tfidf.vocabulary_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




