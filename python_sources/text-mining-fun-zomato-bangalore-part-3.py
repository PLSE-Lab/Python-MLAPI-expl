#!/usr/bin/env python
# coding: utf-8

# Fish fry, chicken fried rice, masala dosa, paneer tikka. Can you guess which of these food items made it to the top 50 mentions in Zomato Bangalore customer reviews?
# 
# This is Part Three of my analysis of the Zomato Bangalore dataset. In Part One we explored the data and predicted restaurant ratings with six selected features using Linear Regression, while in Part Two we predicted the ratings (split into classes) using classification models. 
# 
# In this kernel we will apply text mining / NLP techniques to extract insights from textual features like customer reviews. Then we will try to predict restaurant ratings by feeding the transformed text to a neural network.
# 
# This kernel consists of:
# 
# - Data cleaning (identifying and dropping duplicates, selecting features)
# - Text mining and insights (ngrams, bigrams, trigrams and FreqDist plots)
# - Text processing (regex, tokenizing, stopword removal, lemmatizing, vectorizing)
# - Building an LSTM Neural Network 
# - Model evaluation
# - Results summary

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import RegexpTokenizer as regextoken
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams, trigrams
from nltk import WordNetLemmatizer
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout, LSTM, GRU
from keras.regularizers import l1, l2
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

zomato = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv", na_values = ["-", ""])
# Making a copy of the data to work on
data = zomato.copy()


# In[ ]:


# Dropping duplicates - see Part One for explanation

grouped = data.groupby(["name", "address"]).agg({"listed_in(type)" : list})
newdata = pd.merge(grouped, data, on = (["name", "address"]))
newdata["listed_in(type)_x"] = newdata["listed_in(type)_x"].astype(str) # converting unhashable list to a hashable type
newdata.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)
newdata = newdata.reset_index(drop = True)


# In[ ]:


# Transforming the ratings column 

newdata["rating"] = newdata["rate"].str[:3] # Extracting the first three characters of each string in "rate"
# Removing rows with "NEW" in ratings as it is not a predictable level
newdata = newdata[newdata.rating != "NEW"] 
# Dropping rows that have missing values in ratings 
newdata = newdata.dropna(subset = ["rating"])
# Converting ratings to a numeric column so we can discretize it
newdata["rating"] = pd.to_numeric(newdata["rating"])
# Discretizing the ratings into a categorical feature with 4 levels

newdata["rating"] = pd.cut(newdata["rating"], bins = [0, 3.0, 3.5, 4.0, 5.0], labels = ["0", "1", "2", "3"])


# Our four rating bins (classes) will be 0 to 3 < 3 to 3.5 < 3.5 to 4 < 4 to 5. To make label encoding easier later, we'll label these classes 0, 1, 2, 3. We can think of these as Very Low, Low, Medium and High.

# In[ ]:


# Visualizing the rating class distribution
plt.figure(figsize = (10, 5))
sns.countplot(newdata["rating"])
plt.show()


# In[ ]:


# Summary statistics
newdata.describe(include = "all")


# ### Agenda
# We will use the reviews_list, menu_item, dish_liked and cuisines columns for our analysis.
# 
# First, we will look at the customer reviews and pull out the most common words and phrases. Next, we will analyse cuisine listings and identify cuisines that are rare in Bangalore. Finally we will build a neural network with all four features to predict restaurant ratings.

# In[ ]:


# Creating a new dataset that has only customer reviews and restaurant ratings
reviews_data = newdata[["reviews_list", "rating"]]
# Examining the reviews for the first restaurant in the dataset
reviews_data["reviews_list"][0]
# The text needs cleaning up


# In[ ]:


# Converting all the text to lowercase
reviews_data["reviews_list"] = reviews_data["reviews_list"].apply(lambda x: x.lower())

# Creating a regular expression tokenizer that matches only alphabets
# This will return separate words (tokens) from the text
tokenizer = regextoken("[a-zA-Z]+") 
# Applying the tokenizer to each row of the reviews
review_tokens = reviews_data["reviews_list"].apply(tokenizer.tokenize)
# Examining the tokens created for the first row / restaurant
print(review_tokens[0])


# In[ ]:


# Importing and examining the English stopwords directory 
# These are common words that typically don't add meaning to the text and can be removed
stop = stopwords.words("english")
print(stop)


# In[ ]:


# Adding custom words to stopwords 
stop.extend(["rated", "n", "nan", "x"])
# Removing stopwords from the tokens
review_tokens = review_tokens.apply(lambda x: [token for token in x if token not in stop])
# Concatenating all the reviews 
all_reviews = review_tokens.astype(str).str.cat()
cleaned_reviews = tokenizer.tokenize(all_reviews)

# Getting the frequency distribution of individual words in the reviews
fd = FreqDist()
for word in cleaned_reviews:
    fd[word] += 1
    
# Examining the top 5 most frequent words
fd.most_common(5)


# In[ ]:


# Plotting the top 50 most frequent words
plt.figure(figsize = (10, 5))
fd.plot(50)
plt.show()


# ### Observations
# Of the 50 most frequent words across customer reviews, six reveal food preferences: **chicken, biryani, veg, pizza, rice, paneer**. The only negative word in the top 50 is "bad".
# 
# Factors contributing to restaurant experience are mentioned in the following (descending) order of frequency: place > taste > service > time > ambience > staff > quality > delivery > menu > quantity > friendly.
# 
# Now let us repeat the analysis on a bi-gram level. Bi-grams are pairs of words which can provide better context than individual words.

# In[ ]:


# Generating bigrams from the reviews
bigrams = bigrams(cleaned_reviews)
# Getting the bigram frequency distribution
fd_bigrams = FreqDist()
for bigram in bigrams:
    fd_bigrams[bigram] += 1
# Examining the top 5 most frequent bigrams
fd_bigrams.most_common(5)


# In[ ]:


# Plotting the top 50 most frequent bigrams
plt.figure(figsize = (10, 5))
fd_bigrams.plot(50)
plt.show()


# ### Observations
# 
# We have some new insights! Food items/preferences mentioned in the top 50 bigrams are **ice cream, non veg, North Indian, chicken biryani, fried rice, chicken and South Indian**. Top six bigrams related to restaurant experience: good food > good place > good service > value (for) money > pocket friendly > ambience good. 
# 
# There's a key insight here: **the expense factor, which was missed by individual word frequency counts, was picked up by the bigram frequency counts.**
# 
# Zomato might also be happy to know their membership program "Zomato Gold" is in the top 50 bigrams, with 2593 mentions in the customer reviews.
# 
# What about trigrams? 

# In[ ]:


# Generating trigrams from the reviews
trigrams = trigrams(cleaned_reviews)

fd_trigrams = FreqDist()
for trigram in trigrams:
    fd_trigrams[trigram] += 1

fd_trigrams.most_common(5)


# In[ ]:


plt.figure(figsize = (10, 5))
fd_trigrams.plot(50)
plt.show()


# ### Observations
# There appears to be some bad data (strings of "xa xa xa") somewhere in the reviews, but we'll ignore that. The specific food preferences we can see here are **paneer butter masala, chicken fried rice, chicken biryani, peri peri chicken and chicken ghee roast**. Bangalore is really into chicken.
# 
# On restaurant experience: a specific insight revealed by the trigrams is that **many people are looking for places to hang out with their friends**. 
# 
# We also see a variety of positive trigrams like "must visit place", "food really good", "service also good" and "worth every penny". However, there is only one negative trigram in the top 50 - "worst food ever".
# 
# We now have plenty of insights into customer preferences and experiences, and will move onto an analysis of Bangalore's cuisines.

# In[ ]:


# Creating a new dataset with cuisines info and restaurant ratings
cuisines = newdata[["cuisines", "rating"]]
cuisines["cuisines"] = cuisines["cuisines"].astype(str)
# Converting to lowercase
cuisines["cuisines"] = cuisines["cuisines"].apply(lambda x: x.lower())
# Tokenizing the cuisines
cuisine_tokens = cuisines["cuisines"].apply(tokenizer.tokenize)
# Concatenating all the cuisine names into one text document
all_cuisines = cuisine_tokens.astype(str).str.cat()
cleaned_cuisines = tokenizer.tokenize(all_cuisines)

# Generating cuisine frequencies 
fd_cuisine = FreqDist()
for cuisine in cleaned_cuisines:
    fd_cuisine[cuisine] += 1
    
# Printing the 50 most common cuisines (top 50)
print(fd_cuisine.most_common(50))


# ### Observations
# One must be careful when interpreting these lists. For example, "dogs" can't be a cuisine but the preceding word "hot" tells us that the cuisine is "hot dogs". Another tricky one is Cantonese, which comes under Chinese and so might not really be rare.
# 
# We've done our reviews and cuisines analysis and will now prepare all the text in the dataset for feeding into a neural network.

# ## Text Preprocessing

# In[ ]:


# Converting all the text to strings
newdata[["reviews_list", "menu_item", "dish_liked", "cuisines"]] = newdata[["reviews_list", "menu_item", "dish_liked", "cuisines"]].astype("str")
# Combining all the text data into a single feature called "text"
newdata["text"] = newdata["reviews_list"] + " " + newdata["menu_item"] + " " + newdata["dish_liked"] + " " + newdata["cuisines"]
# Creating a new dataset with text and restaurant ratings
text_data = newdata[["text", "rating"]]
# Converting text to lowercase
text_data["text"] = text_data["text"].apply(lambda x: x.lower())
# Tokenizing the text
tokens = text_data["text"].apply(tokenizer.tokenize) 
# Removing stopwords 
tokens = tokens.apply(lambda x: [token for token in x if token not in stop])


# In[ ]:


print(tokens[0])


# In[ ]:


# Writing a function to lemmatize words
lmtzr = WordNetLemmatizer()
def lem(text):
    return [lmtzr.lemmatize(word) for word in text]

# Applying the function to each row of the text
# i.e. reducing each word to its lemma
tokens_new = tokens.apply(lem)


# In[ ]:


# Applying label encoding and one hot encoding to the restaurant rating classes 
le = LabelEncoder()
target = le.fit_transform(text_data["rating"])
target = to_categorical(target)


# In[ ]:


# Splitting the data into train and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(tokens_new, target, test_size = 0.3, random_state = 0, stratify = target)

# Processing the text with the Keras tokenizer
t = Tokenizer() 
t.fit_on_texts(X_train)
# Setting a vocabulary size that we will specify in the neural network
vocab_size = len(t.word_index) + 1
# The t.word_index contains each unique word in our text and an integer assigned to it
print(vocab_size)


# In[ ]:


# Encoding the text as sequences of integers
train_sequences = t.texts_to_sequences(X_train)
test_sequences = t.texts_to_sequences(X_test)
# Adding zeros so each sequence has the same length 
train_padded = pad_sequences(train_sequences, maxlen=500)
test_padded = pad_sequences(test_sequences, maxlen=500)


# ### Word embedding 
# We'll use Google's pre-trained Word2Vec word embeddings.

# In[ ]:


# Loading Word2Vec word embeddings 

word_vectors = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True)

embedding_dim = 300 # each word will become a 300-d vector

# Creating an empty matrix 
embedding_matrix = np.zeros((vocab_size, embedding_dim)) 
# Each row is a word with 300 dimensions

# Populating the matrix
for word, i in t.word_index.items(): # for each word in the customer reviews vocabulary
    try:
        # get the Word2Vec vector representation for that word
        embedding_vector = word_vectors[word] 
        # add it to the embedding matrix
        embedding_matrix[i] = embedding_vector 
        # handle new words by generating random vectors for them
    except KeyError: 
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25), embedding_dim)


# In[ ]:


embedding_matrix.shape


# In[ ]:


# Examining the words embeddings - vector representations of words
embedding_matrix


# ## Model Building

# In[ ]:


# Building an LSTM neural network

warnings.filterwarnings("ignore")
max_length = 500 # maximum length of each input string (movie review)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length, weights = [embedding_matrix], trainable = False))
model.add(LSTM(100, activation = "tanh"))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
model.fit(train_padded, y_train, validation_data=(test_padded, y_test), epochs=15, batch_size=512)


# ### Evaluation

# In[ ]:


# Predicting on the train data
pred_train = model.predict(train_padded)
pred_train = np.argmax(pred_train, axis=1)
y_train = np.argmax(y_train, axis=1)
# Printing evaluation metrics
print(classification_report(y_train, pred_train))


# In[ ]:


# Predicting on the test data
pred_test = model.predict(test_padded)
pred_test = np.argmax(pred_test, axis=1)
y_test = np.argmax(y_test, axis = 1)
# Printing evaluation metrics
print(classification_report(y_test, pred_test))


# ## Results summary
# By applying text mining techniques to customer reviews and other data, **we discovered common food preferences which became more specific as we progressed (dish names)**. **We also noted which aspects of a restaurant people care about and in what order of priority. Then we identified the most and least common cuisines in the city along with their prevalence.**
# 
# The text mining activity was interesting from both a data science perspective and a business perspective, as it showed the usefulness of different NLTK tools and revealed actionable insights.
# 
# After processing the text, we fed it to an LSTM network with pre-trained Word2Vec word vectors to see if we could predict the four rating classes we created. Accuracy and average F1 scores were lower than what we got with XGBoost in Part Two of my analysis. 
# 
# The conclusion from this three-part analysis is that non-text features and tree-based classification models do a better job of predicting the restaurant ratings than LSTM with text features.
# 
