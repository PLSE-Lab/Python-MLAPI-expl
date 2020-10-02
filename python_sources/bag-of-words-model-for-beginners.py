#!/usr/bin/env python
# coding: utf-8

# # <u> Bag-of-Words Model </u>
# 
# Bag-of-words model is a way of representing text data when modeling text with machine learning algorithms. Machine learning algorithms cannot work with raw text directly; the text must be converted into well defined fixed-length(vector) numbers.
# 
# ## <u> What is a Bag-of-Words? </u>
# 
# A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:
# - A vocabulary of known words.
# - A measure of the presence of known words.
# 
# It is called a bag-of-words , because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. The complexity comes both in deciding how to design the vocabulary of known words (or tokens) and how to score the presence of known words.
# 
# ## <u> Example:</u>
# 
# Below is a snippet of the first few lines of text from the book A Tale of Two Cities by Charles Dickens.
# 
# $\text {It was the best of times} $ <br/>
# $\text {it was the worst of times,}  $  <br/>
# $\text {it was the age of wisdom,} $ <br/>
# $\text {it was the age of foolishness} $
# 
# ## <u>Bag-of-Words Model in SkLearn</u>
# 
# ### <u> Design the Vocabulary </u>
# 
# Make a list of all of the words in our model vocabulary. The CountVectorizer provides a simple way to tokenize a collection of text documents and build a vocabulary of known words.
# - Create an instance of the CountVectorizer class.
# - Call the fit() function in order to learn a vocabulary from one or more documents.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# Multiple documents
text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"] 
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(sorted(vectorizer.vocabulary_))


# That is a vocabulary of 10 words from a corpus containing 24 words.

# 
# 
# ### <u> Create Document Vectors </u>
# 
# #### <u> Document Vectors with CountVectorizer </u>
# 
# Next step is to score the words in each document. Because we know the vocabulary has 10 words, we can use a fixed-length document representation of 10, with one position in the vector to score each word. The simplest scoring method is to mark the presence of words as a boolean value, 0 for absent, 1 for present.
# 
# - Call the transform() function on one or more documents as needed to encode each as a vector.

# In[ ]:


# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())


# The same vectorizer can be used on documents that contain words not included in the vocabulary. These words are ignored and no count is given in the resulting vector.

# In[ ]:


# encode another document
text2 = ["the the the times"]
vector = vectorizer.transform(text2)
print(vector.toarray())


# The encoded vectors can then be used directly with a machine learning algorithm.

# #### <u> Document Vectors with TfidfVectorizer </u>
# 
# Word counts are very basic. One issue with simple counts is that some words like "the" will appear many times and their large counts will not be very meaningful in the encoded vectors. An alternative is to calculate word frequencies.
# 
# - Term Frequency: This summarizes how often a given word appears within a document.
# - Inverse Document Frequency: This downscales words that appear a lot across documents.
# 
# TF-IDF are word frequency scores that try to highlight words that are more frequent in a document but not across documents.If we already have a learned CountVectorizer, we can use it with a TfidfTransformer to just calculate the inverse document frequencies and start encoding documents. The same create, fit, and transform process is used as with the CountVectorizer.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(sorted(vectorizer.vocabulary_))
# encode document
vector = vectorizer.transform([text[0]])


# In[ ]:


print(vectorizer.idf_)


# A vocabulary of 10 words is learned from the documents and each word is assigned a unique integer index in the output vector. The inverse document frequencies are calculated for each word in the vocabulary, assigning the lowest score of 1.0 to the most frequently observed words: "it", "of", "the" , "was".

# In[ ]:


# summarize encoded vector
print(vector.shape)
print(vector.toarray())


# The scores are normalized to values between 0 and 1 and the encoded document vectors can then be used directly with most machine learning algorithms.

# #### <u> Document Vectors with HashingVectorizer </u>
# 
# One limitation with CountVectorizer and TfidfVectorizer is that the encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. Because these vectors contains a lot of zeros(sparse), vocabulary can become very large. This, will require large vectors for encoding documents and impose large requirements on memory and slow down algorithms.
# 
# HashingVectorizer uses a one way hash of words to convert them to integers. No vocabulary is required and we can choose an arbitrary long fixed length vector. A downside is that the hash is a one-way function so there is no way to convert the encoding back to a word (which may not matter for many supervised learning tasks).
# 
# HashingVectorizer hash words, then tokenize and encode documents as needed.

# In[ ]:


from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]
# create the transform small number of "n_features"  may result in hash collisions
vectorizer = HashingVectorizer(n_features=6)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())


# Running the example encodes the sample document as a 6-element sparse array. The values of the encoded document correspond to normalized word counts by default in the range of -1 to 1, but could be made simple integer counts by changing the default configuration.

# ## <u> Bag-of-Words Model in Keras </u>
# 
# ### <u> Design the Vocabulary </u>
# "text_to_word_sequence" Split text into a list of words.

# In[ ]:


from keras.preprocessing.text import text_to_word_sequence
# define the document
# text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)


# ### <u> Document Vectors with hashing trick </u>
# 
# To hashing_trick, in addition to the text, the vocabulary size (total words) must be specified. This could be the total number of words in the document or more if you intend to encode additional documents that contains additional words. The size of the vocabulary defines the hashing space from which words are hashed.

# In[ ]:


from keras.preprocessing.text import hashing_trick

text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)


# ### <u>Keras Tokenizer API </u>
# 
# Keras provides the Tokenizer class for preparing text documents for deep learning. The Tokenizer must be constructed and then fit on either raw text documents or integer encoded text documents.

# In[ ]:


from keras.preprocessing.text import Tokenizer # define 5 documents
docs = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"] 
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(docs)


# Once fit, the Tokenizer provides 4 attributes that we can use to query what has been learned about our documents:
# - <b> word counts </b>: A dictionary of words and their counts.
# - <b> word docs </b>: An integer count of the total number of documents that were used to fit the Tokenizer.
# - <b> word index </b>: A dictionary of words and their uniquely assigned integers.
# - <b> document count </b>: A dictionary of words and how many documents each appeared in.

# In[ ]:


tokenizer.word_counts, tokenizer.document_count, tokenizer.word_index, tokenizer.word_docs


# Once the Tokenizer has been fit on training data, it can be used to encode documents in the train or test datasets.
# The modes available include:
# - binary: Whether or not each word is present in the document. This is the default.
# - count: The count of each word in the document.
# - tfidf: The Text Frequency-Inverse DocumentFrequency (TF-IDF) scoring for each word in the document.
# - freq: The frequency of each word as a ratio of words within each document.

# In[ ]:


encoded_docs = tokenizer.texts_to_matrix(docs, mode='count')
encoded_docs


# ### Bonus
# 
# ### <u> N - Grams </u>
# 
# A vocabulary of grouped words can be created. This allows the bag-of-words to capture a little bit more meaning from the document. In this approach, each word or token is called a gram. Creating a vocabulary of two-word pairs is called a bigram model. An n-gram is an n-token sequence of words.
# 
# Example Text : It was the best of times
# 
# $\text {it was }  $  <br/>
# $\text {was the }$ <br/>
# $\text {the best }$ <br/>
# $\text {best of }$ <br/>
# $\text {of times }$

# ### End
# If you reached this far please comment and upvote this kernel, feel free to make improvements on the kernel and please share if you found anything useful !
