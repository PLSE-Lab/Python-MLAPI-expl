#!/usr/bin/env python
# coding: utf-8

# # <center>Spam Filtering with Naive Bayes</center>
# ## <center>Pavel Bogdanov</center>
# ## <center>Software University</center>

# 
# ### Abstract
# A step by step guide for building a basic spam filter using Naive Bayes. We try not to rely on external libraries and build everything from scratch. All formulas used are based on the Bayes Theorem and no additional methods are used for accuracy improvement.
# 
# ### 1 Introduction
# 
# Unwanted emails are a problem we face daily in modern society. "Spam" as we call it, is something that is encountered by almost anyone using the internet. But how can we deal with this problem? Sure, we could open each individual email and manually assess if it is spam or not, but what if there was another way? What if we could teach the computer to look at the emails for us and tell us ahead of time if it's spam or not? Something like a filter. A spam filter! Now that sounds really nice, but how could we do that? Fortunately for us, humans have had emails for decades, and such a filter already exists. Not only that, but it is in use by many email clients to this day. It is called Naive Bayes spam filter and what we'll do is try to recreate it from scratch and explain how it works along the way.
# 
# ### 2 In theory
# 
# #### 2.1 Bayes Theorem  
# 
# The foundation on which Naive Bayes Filter is based on is a theorem called...you guessed it, Bayes' theorem. Expressed as a mathematical equation it looks like this.
#     
# $$ P(A | B)=\frac{P(B | A) P(A)}{P(B)}, $$
#     
# where  $ A $ and $ B $ are events and $ P(B)\neq 0 $
#  
# * $ P(A | B) $ is a conditional probability: the likelihood of event $ A $ occurring given that $ B $ is true;
# 
# * $ P(B | A) $ is also a conditional probability: the likelihood of event $ B $ occurring given that $ A $ is true;
# 
# * $ P(A) $ and $ P(B) $ are the probabilities of observing $ A $ and $ B $ independently of each other; this is known as the marginal probability.
# 
# #### 2.2 Bayes Theorem for spam filtering
# 
# But what does this theorem have to do with our spam filter? We want to find out what is the likelihood of a specific message to be spam. But a message consists of multiple words. In order to find the combined probability of the words we first have to find the probability of each separate word being a spam word. This is also known as the 'spaminess' of a word and we can calculate it by using one special case of Bayes Theorem where the event is a binary variable.
# 
# $$ P(S | W) = \frac{P(W | S)\cdot P(S)}{P(W | S)\cdot P(S) + P(W | H)\cdot P(H)}, $$
# 
# where,
# * $ P(S|W) $ is the probability that a message is a spam, knowing that a specific word is in it;
# * $ P(W|S) $ is the probability that the specific word appears in spam messages;
# * $ P(S) $ is the overall probability that any given message is spam;
# * $ P(W|H)  $ is the probability that the specific word appears in ham messages;
# * $ P(H) $  is the overall probability that any given message is ham.
# 
# #### 2.3 Unbiased approach
# 
# But we want our filter to be unbiased toward incoming email, so we are going to assume that the probabilities of receiving both spam and ham are equal to 50%. This allows us to simplify the formula to the following:
# 
# $$ P(S | W) = \frac{P(W | S)}{P(W | S) + P(W | H)}, $$
# 
# Our training dataset already has some marked mail, so we know the total number of spam and ham messages. All that's left to do is count in how many of the spam and ham messages respectively, the specific word is present.
# 
# $$ P( W | S ) = \frac{spam\space messages\space containing\space the\space word}{all\space messages\space containing\space the\space word} $$
# 
# $$ P( W | H ) = \frac{ham\space messages\space containing\space the\space word}{all\space messages\space containing\space the\space word} $$ 
# 
# #### 2.4 Combining probabilities
# 
# Now, words in a message are usually not independent of each other, but for simplicity's sake we're going to assume that they are. Bearing that in mind, we can get the spam probability of the message if we combine the spam probabilities of all the words in it like this:
# 
# $$ p=\frac{p_{1} p_{2} \cdots p_{N}}{p_{1} p_{2} \cdots p_{N}+(1-p_{1})(1-p_{2}) \cdots(1-p_{N})} $$
# 
# where,
# * $ p $ is the probability that the suspect message is spam;
# * $ p_{1} $ is the probability $ p(S|W_{1}) $ that it is a spam knowing it contains some first word;
# * $ p_{2} $ is the probability $ p(S|W_{2}) $ that it is a spam knowing it contains some second word;
# * $ p_{n} $ is the probability $ p(S|W_{N}) $ that it is a spam knowing it contains some Nth word.
# 
# #### 2.5 Corrections
# 
# In order to avoid floating-point underflow, we are going to use the alternative form of the formula which looks like this:
# $$ p=\frac{1}{1+e^{\eta}} $$
# where,
# $$ \eta = \sum_{i=1}^{N}[\ln(1-p_{i})-\ln p_{i}] $$
# 
# What happens if we get a word that was never encountered in the learning phase? We get $ \frac{0}{0} $. That's why we're going to ignore such words and carry on.
# Rare words also cause problems to our calculations and a way to deal with this is to use a corrected probability:
# 
# $$ P'(S | W) = \frac{s \cdot P( S ) + n \cdot P(S | W)}{s  + n}, $$
# where,
# 
# * $ P'(S | W) $ is the corrected probability for the message to be spam, knowing that it contains a given word;
# * $ s $  is the strength we give to background information about incoming spam;
# * $ P(S) $ is the probability of any incoming message to be spam;
# * $ n $ is the number of occurrences of this word during the learning phase;
# * $ P(S|W) $ is the spaminess of this word.
# 
# We use this corrected probability instead of the spaminess of the word and finally we get the probability that a message is spam. If it's greater than 50%, we classify the message as spam.

# ---
# ### 3 In practice
# #### 3.1 Loading dependencies

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from nltk.corpus import stopwords


# #### 3.2 Loading dataset
# Our dataset consists of 5574 English SMS messages, 425 of them tagged as spam and 3,375 tagged as ham.
# Ideally, we want a bigger dataset with 50% spam and 50% ham, but this is what we have and it should be enough for a simple proof of concept.

# In[ ]:


mails = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding = 'latin-1')
mails.head()


# #### 3.3 Cleaning dataset
# Of course what we do have is not perfect either, so we'll have to trim it up a little.
# For example we don't need these columns, they are empty:

# In[ ]:


mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
mails.head(1)


# We can also rename these columns appropriately:

# In[ ]:


mails.rename(columns = {'v1': 'spam', 'v2': 'message'}, inplace = True)
mails.head(1)


# And let's turn our spam or ham column into one with boolean values:

# In[ ]:


mails['spam'] = mails['spam'].map({'ham': False, 'spam': True})
mails.head()


# #### 3.4 Splitting dataset
# Now let's just check for ourselves how much of our mail is spam and how much is ham:

# In[ ]:


mails['spam'].value_counts()


# Later we'll need some data that is not part of our training model for testing the accuracy of our spam filter. So lets's split our data in two - one for training our filter, and one for testing it afterwards. 

# In[ ]:


all_mails_count = mails['spam'].count()

train_data = mails.loc[:all_mails_count*0.70]
train_data['spam'].value_counts()


# In[ ]:


test_data = mails.loc[all_mails_count*0.70 + 1:]
test_data.reset_index(inplace = True)
test_data['spam'].value_counts()


# #### 3.5 Visualizing data
# All of this data seems a bit abstract, it would be useful to make some visual representation that helps us see what we're dealing with. Say we want see the top 15 most common words of each category.
# 
# (If we didn't know the difference between a histogram and a bar chart, we probably would have spend way too much time trying to make a function that takes each word and adds it into a list as many times as we've already counted and then used plt.hist with that list to plot a histogram.)
# 
# Luckily we know that a bar chart takes two arguments and we can easily make one like this:

# In[ ]:


def count_words(data):
    counter = collections.OrderedDict()
    for message in data:
        for word in message.split(' '):
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter


# In[ ]:


spam_messages = set(train_data[train_data['spam'] == True]['message'])
spam_words = count_words(spam_messages)
ham_messages = set(train_data[train_data['spam'] == False]['message'])
ham_words = count_words(ham_messages)


# In[ ]:


def bar_chart_words(words, top=10, messages_type="", color="#1f77b4"):
    top_spam = np.array(sorted(words.items(), key=lambda x: -x[1]))[:top]
    top_words = top_spam[ : :-1, 0]
    top_words_count = [int(i) for i in top_spam[ : :-1, 1]]
    # aesthetics
    if messages_type:
        messages_type = messages_type + " " 
    plt.title(f"Top {top} most common words in {messages_type}messages")
    plt.xlabel(f"Number of words")
    plt.barh(top_words, top_words_count, color=color)
    plt.show()


# In[ ]:


bar_chart_words(spam_words, top=15, messages_type="spam", color="orange")


# In[ ]:


bar_chart_words(ham_words, top=15, messages_type="ham", color="green")


# What do we see? Although it looks pretty, it seems like our word counting method is quite flawed. Thanks to the bar chart we could see that we have to make a few adjustments before going further with our filter.

# #### 3.6 Processing data
# 
# How can we improve our method:
# * First off, we can lowercase all the words, because each word means the same thing whether it has capital letters or not.
# * Then we can tokenize each message, that is - we split it up into words and we get rid of all the punctuation characters.
# * After that we can safely get rid of all the words that are only single characters. There isn't really a meaningfull word with one letter anyway.
# * Next, we can remove all the stop words from the message. These are common words that occur in any text, like 'the', 'any', 'such', 'this'. They don't give us any information and we can get rid of them.
# * Now, since we're dealing with english text, we can check if our words are actually valid english words. That way we get rid of all the rare words in the model that don't help us anyway.
# * Finally, comes a thing called stemming. This is an algorithm that detects the "core" meaning of the word and gets rid of the rest. That way words like  'go', 'goes', 'going' will be replaced by a single 'go'. We can use the famous Porter Stemmer algorithm.
# 

# First we have to get our valid english words.

# In[ ]:


words = pd.read_csv('../input/english-words/words.csv', encoding = 'UTF-8', delimiter='\n')
words.rename(columns = {'a': 'words'}, inplace = True)
wordlist = set(words['words'])


# And our stop words which are imported from the Natural Language Toolkit(nltk).

# In[ ]:


stop_words = stopwords.words('english')


# Now the processing itself:

# In[ ]:


def process_message(message):  
    words = message.lower() # lowercase
    words = word_tokenize(words) # tokenization
    words = [word for word in words if len(word) > 1] # non absurd words          
    words = [word for word in words if word not in stop_words] # non stop words
    words = [word for word in words if word in wordlist] # english words
    words = [PorterStemmer().stem(word) for word in words] # stemming
    return words


# Let's make a new function for counting words in a dataset.

# In[ ]:


def count_processed_words(data):
    counter = collections.OrderedDict()
    for message in data:
        words = process_message(message)
        for word in set(words):           
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    return counter


# #### 3.7 Visualizing data (again)

# Now how about a different type of visualization? Again we want to see the top 15 words but this time we can use WordCloud instead of our bar chart function.

# In[ ]:


spam_words = count_processed_words(spam_messages)
ham_words = count_processed_words(ham_messages)


# Top spam words in the training dataset:

# In[ ]:


spam_wc = WordCloud(width = 1024,height = 1024, max_words=15).generate_from_frequencies(spam_words)
plt.figure(figsize = (8, 6), facecolor='k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# Top ham words in the training dataset:

# In[ ]:


spam_wc = WordCloud(width = 1024,height = 1024, max_words=15).generate_from_frequencies(ham_words)
plt.figure(figsize = (8, 6), facecolor='k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# Much better! Now that we are satisfied with our processed words, we can continue with the fun part.

# #### 3.8 Spam classification

# We discard any empty messages formed as a result of our processing, because they will interfere with the accuracy of our filter.

# In[ ]:


spam_messages = [i for i in spam_messages if len(process_message(i)) >= 1]
ham_messages = [i for i in ham_messages if len(process_message(i)) >= 1]


# In[ ]:


all_messages = spam_messages + ham_messages
all_words = count_processed_words(all_messages)


# The following function takes a message as a parameter and applies all the formulas for finding its spam probability. It returns a boolean value, but we can choose to return a percentage if we want. The message paramatere

# In[ ]:


def spam(message, s=1, p=0.5, percentage=False):
    '''
    message - needs to be a non-empty string value for valid result
    s - the strength we give to background information about incoming spam, default is 1 
    p - the probability of any incoming message to be spam, default is 0.5
    percentage - returns result as boolean or a percentage, default is True
    '''
    n = 0
    spam_freq = 0
    ham_freq = 0
    for word in process_message(message):
        
        if word in spam_words.keys(): 
            # count of spam messages containing the word / count of all messages containing the word
            spam_freq = (spam_words[word] / all_words[word])
            
        if word in ham_words.keys():
            # count of ham messages containing the word / count of all messages containing the word
            ham_freq = (ham_words[word] / all_words[word])
            
        # if word is not in trained dataset we ignore it    
        if not (spam_freq + ham_freq) == 0 and word in all_words.keys(): 
            spaminess_of_word = (spam_freq ) / (spam_freq  + ham_freq )
            corr_spaminess = (s * p + all_words[word] * spaminess_of_word) / (s + all_words[word])            
            n += np.log(1 - corr_spaminess) - np.log(corr_spaminess)

    spam_result = 1 / (1 + np.e**n)
    
    if percentage:
        print(f'Spam probability: {spam_result*100:.2f}%')
    elif spam_result > 0.5:    
        return True
    else:
        return False


# #### 3.9 Testing the spam filter
# Before trying it out on all our test messages, let's see how it works on a case by case basis with some simple messages that we just came up with.

# In[ ]:


spam("Join SoftUni, FREE Programming Basics course! Don't miss out on this offer! Become a code wizard GUARANTEED!")


# In[ ]:


spam("Hey man, SoftUni is pretty great. This Data Science course? Highly recommend.")


# So far so good, let's try some progressively spammier messages.

# In[ ]:


spam("Call me when you get home", percentage=True)


# In[ ]:


spam("Call me now to win a FREE home!", percentage=True)


# In[ ]:


spam("Call now to win a FREE prize!", percentage=True)


# Our spam filter appears to be working, but lets see how accurate it is on our testing dataset.

# In[ ]:


def test(spam_test, ham_test, s=1, p=0.5, details=False):
    '''
    spam_test - list of spam messages to be tested
    ham_test - list of ham messages to be tested
    details - displays additional information
    '''
    spam_count = 0
    ham_count = 0
    for message in spam_test:
        if spam(message, s, p):
            spam_count += 1
        else:
            ham_count += 1

    true_positive = spam_count
    false_negative = ham_count
    

    spam_count = 0
    ham_count = 0
    for message in ham_test:
        if spam(message, s, p):
            spam_count += 1
        else:
            ham_count += 1
    
    false_positive = spam_count
    true_negative = ham_count

    # How many selected messages are spam?
    spam_precision = true_positive / (true_positive + false_positive)
    
    # How many spam messages are selected?
    spam_recall = true_positive / (true_positive + false_negative)
    
    # Harmonic mean between precision and recall.
    spam_fscore = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall) 
    
    
    # How many selected messages are ham?
    ham_precision = true_negative / (true_negative + false_negative)
    
    # How many ham messages are selected?
    ham_recall = true_negative / (true_negative + false_positive)
    
    # Harmonic mean between precision and recall.
    ham_fscore = 2 * (ham_precision * ham_recall) / (ham_precision + ham_recall)
    
    # If the data was ballanced.
    # accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    # For unballanced data.
    ballanced_accuracy = ( spam_recall + ham_recall ) / 2
    
    if details:
        print("True Positive: ", true_positive)
        print("False Negative:", false_negative)
        print("True Negative: ", true_negative)
        print(f"False Positive: {false_positive}\n")
        
        print(f"Spam precision: {spam_precision*100:.2f}%")
        print(f"Spam recall: {spam_recall*100:.2f}%")
        print(f"Spam F-score: {spam_fscore*100:.2f}%\n")
        
        print(f"Ham precision: {ham_precision*100:.2f}%")
        print(f"Ham recall: {ham_recall*100:.2f}%")
        print(f"Ham F-score: {ham_fscore*100:.2f}%\n")
        
    print(f"Accuracy: {ballanced_accuracy*100:.2f}%\n")


# Loading our test messages we split in the beginning:

# In[ ]:


test_spam_messages = set(test_data[test_data['spam'] == True]['message'])
test_spam_messages = [i for i in test_spam_messages if len(process_message(i)) >= 1]


test_ham_messages = set(test_data[test_data['spam'] == False]['message'])
test_ham_messages = [i for i in test_ham_messages if len(process_message(i)) >= 1]


# Evaluating each message in the dataset:

# In[ ]:


test(spam_test=test_spam_messages, ham_test=test_ham_messages, details=True)


# All things considered, the result is pretty good for a dataset of this size. We can see that our filter identifies ham significantly more accurately than spam and that can be explained by our uneven ratio of spam to ham in the trained dataset. If we want to increase the overall accuracy of the filter we just need to keep adding more data in the trained model aiming for optimal results at a ratio of 50% spam and 50% ham.

# ### 4 Conclusion
# We did not create the perfect universal spam filter. We made a lot of assumptions that limit our model to a specific type of messages. For example we only consider English words as valid, but around 3/4 of the world population doesn't speak English. Or the fact that we took all the words from a message and assumed they have no connection to each other, but "not okay" is not the same as "not","okay". Or what if a clever spammer uses the knowledge of our word processing and tries to change a few letter in each word so that we classify them as new words instead of spam. "Ca11 n0w to have a gr8 t1me" is not going to give us the same probability as "Call now to have a great time".
# 
# We made a simple proof of concept. However, all great spam filters have to start from somewhere and this is the foundation on which most of them are built on.
# 
# 
# ### 5 References:
# 
# Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset
# 
# English words: https://github.com/dwyl/english-words
# 
# https://en.wikipedia.org/wiki/Bayes%27_theorem
# 
# https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
# 
# https://www.linuxjournal.com/article/6467
# 
# https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73
# 
# https://www.seas.upenn.edu/~cis391/Lectures/naive-bayes-spam-2015.pdf
# 
# http://www.cs.ubbcluj.ro/~gabis/DocDiplome/Bayesian/000539771r.pdf
# 
# https://en.wikipedia.org/wiki/Precision_and_recall
# 
# https://softuni.bg/trainings/2313/math-concepts-for-developers-april-2019
# 
# https://www.quora.com/What-percentage-of-the-world-speaks-no-English
