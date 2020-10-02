#!/usr/bin/env python
# coding: utf-8

# ## Categorizing Tweets Using Natural Language Processing
# 
# *adapted from https://github.com/hundredblocks/concrete_NLP_tutorial.git*

# ## Helping the Police Identify Disasters
# 
# People post on Twitter very frequently. Sometimes these posts are about disasters. Wouldn't it be a good idea if the police could leverage this information to identify and respond quickly to disasters happening in the area?
# 
# But that's a lot of data to go through. How could we automate this process? 
# 
# Let's write code that does it for us!
# 
# 
# ## Workshop Overview
# 
# We are going to show you the pipeline that we will follow in this notebook:
# 
# ![Pipeline](https://raw.github.com/loryfelnunez/nlp_workshop/master/newpipeline.png)
# 
# 
# -------
# 
# **Below are the NLP and ML specific tasks in the pipeline:**
# 
# 
# ### **Preprocess**:  Clean and normalize your data for better results
# 
# Examples: making all letters lowercase, stemming, removing URLs
# 
#  *We have included code to make all letters lowercase and remove URLs, but you can do some additional work to increase the accuracy of your model*
# 
# 
# 
# *Tweet 1:*  HELP!! Fire in John St.  @annab is there!           
# 
# *Cleaned Tweet 1:  help fire in john st is there*
# 
# *Tweet 2: * Fire and Fury book set bookshelves on fire at http://torch.com !!!
# 
# *Cleaned Tweet 2: fire and fury book set bookshelf on fire at*
# 
# 
# 
# ### **Embedding**: transform your text data into numerical vectors (because machine learning algorithms need numbers as inputs!)
# *You have 3 different ways to turn your text into numbers; pick one*
# 
# |                 | help | fire | fury | book | etc... |
# |-----------------|------|------|------|------|--------|
# | Cleaned Tweet 1 | 1    | 1    | 0    | 0    | ...    |
# | Cleaned Tweet 2 | 0    | 2    | 1    | 1    | ...    |
# 
# ![Embeddings](https://raw.github.com/loryfelnunez/nlp_workshop/master/embedtfnew.png)
# 
# ### **Classification**: Identify to which category a tweet belongs
# 
# Once you select an embedding and a classification algorithm, you will first use training data to help your model learn how to analyze tweets to decide if they are relevant or not. Your model checks the tweet, checks whether it's relevant or not, and learns.
# 
# 
# ![Models](https://raw.github.com/loryfelnunez/nlp_workshop/master/embedtfnewmodels.png)
# 
# *We have provided you with 4 different classification algorithms; select the best one*
# 
# ### Evaluation:  how good is your model?
# 
# You then use testing data (data that your model has never seen before) to test the accuracy of your trained model. Your model checks the tweet, decides whether it's relevant or not, then we check if the machine made the right decision since we know already whether the tweet is relevant or not. 
# 
# Your model is going to be evaluated on **F1 score** which is the harmonic mean of:
# 
# **precision**:  Was your predicted 'Relevant' Tweet really relevant?  (same for irrelevant)
# 
# **recall**:  Did your model classify all actual 'Relevant' Tweets as relevant? (same for irrelevant)
# 
# -------
# 
# ### Let's submit
# 
# Using your chosen embedding and saved model, you are going to make predictions on unseen competition data.  Then you are going to download and submit your predictions and see your team name in the Leaderboard!
# 
# 

# ## Let's start coding!
# 
# Let's import all the libraries we will need upfront

# In[ ]:


## MANDATORY 
## [IMPORT NECESSARY LIBRARIES]

import gensim
import nltk
import sklearn
import pandas as pd
import numpy as np
import matplotlib

import re
import codecs
import itertools
import matplotlib.pyplot as plt

print ('DONE [IMPORT NECESSARY LIBRARIES]')


# We also need to import our dataset.

# In[ ]:


## MANDATORY 
## [ETL] Import Data

input_file = codecs.open("../input/nlp-starter-test/socialmedia_relevant_cols.csv", "r",encoding='utf-8', errors='replace')

# read_csv will turn CSV files into dataframes
questions = pd.read_csv(input_file)

#let's give names to the columns of our dataframe
questions.columns=['text', 'choose_one', 'class_label']

print ('DONE - [ETL] Import Data')


# ## Exploring the Dataset
# 
# Let's create a function to clean up our data.
# 
# For now all we are going to do is to convert everything to lower case and remove URLs, but after examining the data you might want to add code to remove punctuation marks or other irrelevant characters or words.

# In[ ]:


## [EDA] Explore Imported Data

questions.head()
#questions.head(10)
#questions.tail()
#questions.describe()


# ## Cleaning the dataset
# 
# You took a look at the data... When cleaning the dataset, think of the following questions:
# 
# What could confuse a computer? It probably doesn't know the difference between 'This' and 'this'. So let's turn everything into lowercase.
# 
# Which words or phrases are irrelevant? Would a URL tell us much? Probably not, so let's remove all URLs.

# In[ ]:


## MANDATORY 
## [PREPROCESS] Text Cleaning

def standardize_text(df, text_field):
    # normalize by turning all letters into lowercase
    df[text_field] = df[text_field].str.lower()
    # get rid of URLS
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r"http\S+", "", elem))  
    return df

# call the text cleaning function
clean_questions = standardize_text(questions, "text")

print ('DONE - [PREPROCESS] Text Cleaning')


# Let's take another look at our data to see if anything has changed...

# In[ ]:


## [EDA] Explore Cleaned Data

clean_questions.head()
#clean_questions.tail()


# ## Label Representation 
# 
# Let's look at our class balance; are all labels represented fairly?

# In[ ]:


## [EDA] Explore Class Labels

clean_questions.groupby("class_label").count()


# We can see our classes are pretty balanced, with a slight oversampling of the "Irrelevant" class.

# ## Tokenization
# What is the unit of analysis for this exercise? We should probably analyze words instead of sentences...
# 
# We'll use regex to tokenize sentences to a list of words. 
# 
# We'll then analyze Tweet lengths and inspect our data a little more to validate results.

# In[ ]:


## MANDATORY
## [PREPROCESS] Tokenize

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
clean_questions.head()


# In[ ]:


## [EDA] Explore words and sentences

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]

sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]

VOCAB = sorted(list(set(all_words)))

print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))


# In[ ]:


# [EDA] Explore Vocabulary

# What are the words in the vocabulary
print (VOCAB[0:100])

# What are the most commonly occuring words
from collections import Counter
count_all_words = Counter(all_words)

# get the top 100 most common occuring words
count_all_words.most_common(100)


# ## Preparing Training and Test Data
# We need to split our data into a training set and a test set. Later on we will 'fit' our model using the training set. We will use the test set to ask the model to predict the labels, and we'll then compare the predicted labels against the actual test set labels

# In[ ]:


## MANDATORY 
## [CLASSIFY] Train test Split

from sklearn.model_selection import train_test_split

list_corpus = clean_questions["text"]
list_labels = clean_questions["class_label"]

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)

print("Training set: %d samples" % len(X_train))
print("Test set: %d samples" % len(X_test))


# In[ ]:


## [CLASSIFY] Check Data to be Trained

print (X_train[:10])


# In[ ]:


## [CLASSIFY] Check the Training Labels

print (y_train[:10])


# ##  Embedding
# Machine Learning on images can use raw pixels as inputs. Fraud detection algorithms can use customer features. What can NLP use?
#  
# A natural way to represent text for computers is to encode each character individually, this seems quite inadequate to represent and understand language. Our goal is to first create a useful embedding for each sentence (or tweet) in our dataset, and then use these embeddings to accurately predict the relevant category.
# 
# Here, you are given 3 different methods. Which one is best?

# ### 1.  Bag of Words Counts
# The simplest approach we can start with is to use a bag of words model. A bag of words just associates an index to each word in our vocabulary, and embeds each sentence as a list of 0s, with a 1 at each index corresponding to a word present in the sentence.

# In[ ]:


## MANDATORY FOR BOW EMBEDDING
## [EMBEDDING] Tranform Tweets to BOW Embedding

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+')

bow = dict()
bow["train"] = (count_vectorizer.fit_transform(X_train), y_train)
bow["test"]  = (count_vectorizer.transform(X_test), y_test)
print(bow["train"][0].shape)
print(bow["test"][0].shape)


# ### 2. TFIDF Bag of Words
# Let's try a slightly more subtle approach. On top of our bag of words model, we use a TF-IDF (Term Frequency, Inverse Document Frequency) which means weighing words by how frequent they are in our dataset, discounting words that are too frequent, as they just add to the noise.

# In[ ]:


## MANDATORY FOR TFIDF EMBEDDING
## [EMBEDDING] Transform Tweets to TFIDF Embedding

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

tfidf = dict()
tfidf["train"] = (tfidf_vectorizer.fit_transform(X_train), y_train)
tfidf["test"]  = (tfidf_vectorizer.transform(X_test), y_test)

print(tfidf["train"][0].shape)
print(tfidf["test"][0].shape)


# ### 3. Word2Vec - Capturing semantic meaning
# Our first models have managed to pick up on high signal words. However, it is unlikely that we will have a training set containing all relevant words. To solve this problem, we need to capture the semantic meaning of words. Meaning we need to understand that words like 'good' and 'positive' are closer than apricot and 'continent'.
# 
# Word2vec is a model that was pre-trained on a very large corpus, and provides embeddings that map words that are similar close to each other. A quick way to get a sentence embedding for our classifier, is to average word2vec scores of all words in our sentence.

# In[ ]:


## MANDATORY FOR WORD2VEC EMBEDDING
## [EMBEDDING] Load Word2Vec Pretrained Corpus

word2vec_path = "../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

print ('DONE [Load Word2Vec Pretrained Corpus]')


# In[ ]:


## MANDATORY FOR WORD2VEC EMBEDDING
## [EMBEDDING] Get Word2Vec values for a Tweet

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions_tokens, generate_missing=False):
    embeddings = clean_questions_tokens.apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

# Call the functions
embeddings = get_word2vec_embeddings(word2vec, clean_questions['tokens'])

print ('[EMBEDDING] Get Word2Vec values for a Tweet')


# In[ ]:


## MANDATORY FOR WORD2VEC EMBEDDING
## [CLASSIFY] Word2Vec Train Test Split

X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(embeddings, list_labels, 
                                                                    test_size=0.2, random_state=40)

w2v = dict()
w2v["train"] = (X_train_w2v, y_train_w2v)
w2v["test"]  = (X_test_w2v, y_test_w2v)

print ('DONE - [CLASSIFY] Word2Vec Train Test Split]')


# ## The Classifiers
# We are providing you with 4 different classification models; which one is best?

# ### 1. Logistic Regression classifier
# Starting with a logistic regression is a good idea. It is simple, often gets the job done, and is easy to interpret.

# In[ ]:


## MANDATORY FOR LOGISTIC REGRESSION CLASSIFIER
## [CLASSIFY] Initialize Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', random_state=40)

print ('DONE - [CLASSIFY] Initialize Logistic Regression')


# ### 2. Linear Support Vector Machine classifier
# Common alternative to logistic regression

# In[ ]:


## MANDATORY FOR SUPPORT VECTOR MACHINE CLASSIFIER
## [CLASSIFY] Initialize Support Vector Machine Classifier

from sklearn.svm import LinearSVC

lsvm_classifier = LinearSVC(C=1.0, class_weight='balanced', multi_class='ovr', random_state=40)

print ('[CLASSIFY] Initialize Support Vector Machine Classifier')


# ### 3. Naive Bayes
# 
# A probabilistic alternative.

# In[ ]:


## MANDATORY FOR NAIVE BAYES CLASSIFIER
## [CLASSIFY] Initialize Naive Bayes
## NOTE - Does not work with Word2Vec Embedding

from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB()

print ('DONE - [CLASSIFY] Initialize Naive Bayes')


# ### 4. Decision Tree
# 
# Classifier that  partitioning the data into subsets that contain instances with similar values (homogenous).

# In[ ]:


## MANDATORY FOR DECISION TREE
## [CLASSIFY] Initialize Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)

print ('DONE - [CLASSIFY] Initialize Decision Tree')


# ## Evaluation - Preparing your metrics
# Now that our data is clean and prepared and trasformed into a format the machine can understand, let's dive in to the machine learning part.
# 
# **But** before anything else, let us define some functions that will help us assess the accuracy of our trained models.

# In[ ]:


## MANDATORY 
## [EVALUATE] Prepare Metrics

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

print ('DONE - [EVALUATE] Prepare Metrics')


# Also define a function that plots a *Confusion Matrix* which helps us see our false positives and false negatives

# In[ ]:


## MANDATORY
## [EVALUATE] Confusion Matrix

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt

print ('DONE - [EVALUATE] Confusion Matrix')


# ## Let's run!!!

# ### First we choose the Embedding.  For this Kernel we choose Bag of Words (*bow*)
# Other choices are tfidf (Term Frequency Inverse Document Frequency) and w2v (Word2Vec)

# In[ ]:


## MANDATORY 
## [EMBEDDING] CHOOSE EMBEDDING

embedding = bow                  # bow | tfidf | w2v

print ('DONE - [EMBEDDING] CHOOSE EMBEDDING')


# ### Then we choose the Classifier.  For this Kernel we choose Logistic Regression (*lr_classifier*)
# Other choices are lsvm_classifier (Linear Support Vector Machine) and nb_classifier (Naive Bayes)

# In[ ]:


## MANDATORY 
## [CLASSIFY] CHOOSE CLASSIFIER

classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier| dt_classifier

print ('DONE - [CLASSIFY] CHOOSE CLASSIFIER')


# ### Then we Fit and Predict on our Test Data so we can score our Model.

# In[ ]:


## MANDATORY 
## [CLASSIFY] Train Classifier on Embeddings

classifier.fit(*embedding["train"])
y_predict = classifier.predict(embedding["test"][0])

print ('DONE - [CLASSIFY] Train Classifier on Embeddings')


# #### We score our model.
# 
# We use the mean F1 score to score our Model.
# 
# **Recall** is the ability of the classifcation model  to find all the data points of interest in a dataset.
# 
# **Precision** is  the ability of a classification model to identify only the relevant data points. 
# 
# While recall expresses the ability to find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant actually were relevant.
# 
# Taking both metrics into account, we have the **mean F1 score**.

# In[ ]:


## MANDATORY 
## [EVALUATE] Score chosen model

accuracy, precision, recall, f1 = get_metrics(embedding["test"][1], y_predict)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


# A metric is one thing, but in order to make an actionnable decision, we need to actually inspect the kind of mistakes our classifier is making. Let's start by looking at the **confusion matrix**.

# In[ ]:


## MANDATORY 
## [EVALUATE] Confusion matrix for chosen model

cm = confusion_matrix(embedding["test"][1], y_predict)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant','Disaster', 'Unsure'], normalize=False, title='Confusion Matrix')
plt.show()


# ## Now let us run this model on the competition data set
# - We need to read the test competion data
# - We need to Vectorize the Tweets with our Embedding of Choice (Count Vectorizer, TFIDF, Word2Vec)
# - We need to classify the vectorized Tweets with our Classifier of choice (Logistic, Linear SVM, , Naive Bayes)

# In[ ]:


## MANDATORY for COMPETITION
## [ETL] Load competition Test Data

test_X = pd.read_csv('../input/nlp-starter-test/test.csv')
test_corpus = test_X["Tweet"]
test_Id = test_X["Id"]

print ('DONE [ETL] Load competition Test Data')


# In[ ]:


## MANDATORY for COMPETITION
## [PREPROCESS] Tokenize Competition Data

# tokenize the test_corpus
test_corpus_tokens = test_corpus.apply(tokenizer.tokenize)

print ('[PREPROCESS] Tokenize Competition Data')


# ### Apply Word Embeddings to the Tweets
# 
# #### For Bag of Words:
# count_vectorizer.transform(test_corpus)
# #### For TFIDF
# tfidf_vectorizer.transform(test_corpus)
# #### For Word2Vec
# get_word2vec_embeddings(word2vec, test_corpus_tokens)

# In[ ]:


## MANDATORY for COMPETITION 
## [EMBEDDING] Apply Chosen Embeddings to the Tweets

vectorized_text = dict()
vectorized_text['test']  = (count_vectorizer.transform(test_corpus))  # see options in the above cell

print ('DONE - [EMBEDDING] Apply Chosen Embeddings to the Tweets')


# ### Classify the Vectorized Tweets with our Classifier of choice

# In[ ]:


## MANDATORY for COMPETITION  
## [CLASSIFY] Apply Chosen Classifier to the Embedding

embedding = vectorized_text                
classifier = lr_classifier     # lr_classifier | lsvm_classifier | nb_classifier | dt_classifier
predicted_sentiment = classifier.predict(embedding['test']).tolist()

print ('DONE - [CLASSIFY] Apply Chosen Classifier to the Embedding')


# In[ ]:


## MANDATORY for COMPETITION  
## [PREPARE SUBMISSION]


results = pd.DataFrame(
    {'Id': test_Id,
     'Expected': predicted_sentiment
    })

# Write your results for submission.
# Make sure to put in a meaningful name for the 'for_submission.csv 
# to distinguish your submission from other teams.

results.to_csv('for_submission_sample.csv', index=False)

print ('DONE - [PREPARE SUBMISSION]')

