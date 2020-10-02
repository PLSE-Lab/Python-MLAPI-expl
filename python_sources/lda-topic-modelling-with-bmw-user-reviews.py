#!/usr/bin/env python
# coding: utf-8

# # EXTRACTING TOPICS FROM CUSTOMER REVIEWS 
# 
# I was interested in learning about Topic Modelling so I decided to use LDA topic modelling to extract common 
# 
# topics in customer reviews.
# 
# Since I really like BMW's, I decided to 
# use their customer reviews. 
# 
# There are basically three main steps: 
# 1. Preprocessing the Reviews
# 2. Building the LDA model 
# 3. Inferring the topics 

# ## Collecting the Reviews 

# In[ ]:


# importing all the necessary packages 
import gensim
from gensim import corpora
import re
import nltk
from nltk.corpus import stopwords 
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# I start by reading in the file containing all the reviews. I am using the [OpinRank](http://http://archive.ics.uci.edu/ml/datasets/opinrank+review+dataset) review dataset that contains 
# 
# both hotel and car reviews. 
# 
# I will just be using the reviews for the BMW 3 Series (2009) model. 

# In[ ]:


#reading the file in:  
f = open('../input/bmw_3_series_review/2009_bmw_3_series.txt',
         encoding = "ISO-8859-1")
content = f.read()


# Lets see how the information is stored. 
# We notice that the actual reviews are within the `<TEXT> <\TEXT>` tags. 
# 
# We also notice there is some more comments by the customer within the 
# `<FAVORITE> <\FAVORITE>` tags. 

# In[ ]:


print(content[:950])


# For our purposes, we only need the reviews, so in the next steps, I 
# extract the reviews 
# 
# as well as the 'favorites' comment. 
# 
# I will append the favorite comment 
# to the review it corresponds to, to make things a little simpler. 

# In[ ]:


pattern = "<TEXT>(.*?)</TEXT>"
pattern2 = "<FAVORITE>(.*?)</FAVORITE>"
matches = re.findall(pattern, content)
matches2 = re.findall(pattern2, content)
reviews = []
counter = 0 
# adding all the reviews to a list 
for match in matches:
    reviews.append(match)
# appending favourite comment to the review. 
for match2 in matches2:
    reviews[counter] += ' ' + match2
    counter += 1


# In[ ]:


#Testing to see if it worked 
print(len(reviews))
print(reviews[0])


# ## 1. Preprocessing the reviews
# 
# Now that we have our reviews, we need to do some pre-processing. 
# 
# I will use the genism package to peform a simple pre-processing and then do the following:
# 
# 1. Remove stopwords like 'a', 'the', 'an', etc. I also chose to remove common words like 'car' and 'bmw' 
# 
#     since they might come up very frequently.
# 
#     I removed "drive" (all forms) because it is understood that car reviews 
# 
#     are written by people who've driven the car. Flitering this word out will help us get better results. 
# 
# 2. Lemmatize the text. That is, convert it to its root form (driving -> drive). 
# 
#      Another option is Stemming the text. However, stemming results in some words that have no 
# 
#      meaning so I chose not to use it. (fairly -> fairli)

# Here are some helper functions for preprocessing: 

# In[ ]:


# performs a simple preprocessing using gensism 
def create_tokens(review): 
    return gensim.utils.simple_preprocess (review)
    


# In[ ]:


# removes stopwords and commonly occuring words in car reviews 
def remove_stopWrd(tokenized_review):
    filtered_review = []
    common = ['car', 'bmw', 'drive', 'driving', 'drives', 'drove', 'driven']
    stop_words = set(stopwords.words('english')) 
    for word in tokenized_review:
        if word not in stop_words and word not in common:
            filtered_review.append(word)
    return filtered_review


# I chose to use spacy for lemmatizing because it doesnt require you to specify a POS (parts of speech)  tag like 
# 
# WordNetLemmatizer does in order to convert verbs and adjectives. 

# In[ ]:


# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp= spacy.load('en', disable=['parser', 'ner'])
def apply_stemming(filtered_review):
    for i in range(len(filtered_review)):
        lemmatized = nlp(filtered_review[i])
        filtered_review[i] = lemmatized[0].lemma_
    return filtered_review


# I then preprocess each of the reviews in the list using the helper function. ( takes a little while ~ 10secs) 

# In[ ]:


filtered_rev = []
for rev in reviews:
    tk_review = create_tokens(rev.lower())
    stp_review = remove_stopWrd(tk_review)
    stem_review = apply_stemming(stp_review)
    
    filtered_rev.append(stem_review)   


# Now that we have our cleaned reviews, I create a word cloud to see the types of words that are occuring 
# 
# frequently. Also, I want to check that no stop words or punctuation shows up in our word cloud. 
# 
# The wordcloud below contains words that you would typically find in a car review. 
# 
# We also notice that the cloud contains mostly positive words. 

# In[ ]:


text = str(filtered_rev)
text = text.replace("\'" , " ") # to avoid trailing ' in the words

wordcloud = WordCloud(
    max_font_size= 50, 
    max_words=200, 
    background_color="white",
    scale = 3
).generate(text)
plt.figure(1, figsize = (10, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# ## 2. Building the LDA model

# Now that the reviews are all nice and cleaned up, I use the list of reviews to create a term dictionary
# 
# for our corpus or document term matrix. (Every unique term is assigned an index)

# In[ ]:


dictionary = corpora.Dictionary(filtered_rev)
print(dictionary)


# Using the Dictionary above, I then create the Document Term Matrix which we will feed to the LDA model. 

# In[ ]:


docterm_mat = [dictionary.doc2bow(rev) for rev in filtered_rev]


# ## Inferring the topics 
# Now lets use the LDA model to find the two most common topics in the reviews. We notice the the first topic 
# 
# is about the "look and feel" of the car or in other words, the exterior and interiors of the car. The second topic
# 
# seems to be about the performance of the car. In both cases, the reviews tend to be positive because words 
# 
# like "good" "love" and "great" appear frequently. 

# In[ ]:


# Creating the object for LDA model
LDA = gensim.models.ldamodel.LdaModel

# LDA model
lda_model = LDA(corpus=docterm_mat, id2word=dictionary, num_topics= 2, 
                random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()


# In[ ]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda_model, docterm_mat, dictionary)


# 
