#!/usr/bin/env python
# coding: utf-8

# ## POS Tagging, HMMs, Viterbi
# 
# Let's learn how to do POS tagging by Viterbi Heuristic using tagged Treebank corpus. Before going through the code, let's first understand the pseudo-code for the same. 
# 
# 1. Tagged Treebank corpus is available (Sample data to training and test data set)
#    - Basic text and structure exploration
# 2. Creating HMM model on the tagged data set.
#    - Calculating Emission Probabaility: P(observation|state)
#    - Calculating Transition Probability: P(state2|state1)
# 3. Developing algorithm for Viterbi Heuristic
# 4. Checking accuracy on the test data set
# 
# 
# ## 1. Exploring Treebank Tagged Corpus

# In[ ]:


#Importing libraries
import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


# In[ ]:


# reading the Treebank tagged sentences
wsj = list(nltk.corpus.treebank.tagged_sents())


# In[ ]:


# first few tagged sentences
wsj[:3]


# In[ ]:


# Splitting into train and test
random.seed(1234)
train_set, test_set = train_test_split(wsj,test_size=0.2)

print(len(train_set))
print(len(test_set))
print(train_set[:40])


# In[ ]:


# Getting list of tagged words from train set
train_tagged_words = [(word,tag) for sent in train_set for word,tag in sent]
len(train_tagged_words)


# In[ ]:


# tokens in train set
tokens = [word for word,tag in train_tagged_words]
tokens[:10]


# In[ ]:


# unique vocabulary from train set
V = set(tokens)
print(len(V))


# In[ ]:


# number of tags in train set
T = set([tag for word,tag in train_tagged_words])
len(T)


# In[ ]:


# . -> represent the start of a sentence
# $ -> represent the end of the sentence
print(T)


# ## 2. POS Tagging Algorithm - HMM
# 
# We'll use the HMM algorithm to tag the words. Given a sequence of words to be tagged, the task is to assign the most probable tag to the word. 
# 
# In other words, to every word w, assign the tag t that maximises the likelihood P(t/w). Since P(t/w) = P(w/t). P(t) / P(w), after ignoring P(w), we have to compute P(w/t) and P(t).
# 
# 
# P(w/t) is basically the probability that given a tag (say NN), what is the probability of it being w (say 'building'). This can be computed by computing the fraction of all NNs which are equal to w, i.e. 
# 
# P(w/t) = count(w, t) / count(t). 
# 
# 
# The term P(t) is the probability of tag t, and in a tagging task, we assume that a tag will depend only on the previous tag. In other words, the probability of a tag being NN will depend only on the previous tag t(n-1). So for e.g. if t(n-1) is a JJ, then t(n) is likely to be an NN since adjectives often precede a noun (blue coat, tall building etc.).
# 
# 
# Given the penn treebank tagged dataset, we can compute the two terms P(w/t) and P(t) and store them in two large matrices. The matrix of P(w/t) will be sparse, since each word will not be seen with most tags ever, and those terms will thus be zero. 
# 

# ### Emission Probabilities

# In[ ]:


t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))
w_given_t.shape


# In[ ]:


# compute word given tag: Emission Probability
# p(w|t) = (#word w tagged with tag t in the corpus) / (#tag t appearing in the corpus)

def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [(w,t) for w,t in train_bag if t==tag]
    count_tag = len(tag_list) # count of tag t present in the corpus
    w_given_tag_list = [w for w,t in tag_list if w==word] #word w with the tag t present in the corpus
    count_w_given_tag = len(w_given_tag_list) #count of word w with the tag t in the corpus
    
    return (count_w_given_tag, count_tag)


# In[ ]:


# examples

# large
print("\n", "large")
print(word_given_tag('large', 'JJ'))
print(word_given_tag('large', 'VB'))
print(word_given_tag('large', 'NN'), "\n")

# will
print("\n", "will")
print(word_given_tag('will', 'MD'))
print(word_given_tag('will', 'NN'))
print(word_given_tag('will', 'VB'))

# book
print("\n", "book")
print(word_given_tag('book', 'NN'))
print(word_given_tag('book', 'VB'))

# Android
print("\n", "android")
print(word_given_tag('android', 'NN'))


# In[ ]:


"""word_with_tag_matrix = np.zeros((len(T), len(V)), dtype='float32')
for i, t in enumerate(list(T)):
    for j, w in enumerate(list(V)): 
        word_with_tag_matrix[i, j] = word_given_tag(w, t)[0]/word_given_tag(w, t)[1]"""


# In[ ]:


# convert the matrix to a df for better readability
#word_with_tags_df = pd.DataFrame(word_with_tag_matrix, columns = list(V), index=list(T))
#word_with_tags_df


# ### Transition Probabilities

# In[ ]:


# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability
# p(t2|t1)= (#tag t1 is followed by tag t2)/ (#tag t1 appearing in corpus)

def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [t for w,t in train_bag] #get all the tags from training set
    count_t1 = len([t for t in tags if t==t1]) #count of t1 appearing in the corpus
    count_t2_t1 = 0  #count of t2 coming after t1 -> t1 followed by t2
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1 #increment count if t1 is followed by t2
    return (count_t2_t1, count_t1)


# In[ ]:


def t2_given_t1_prob(t2,t1,train_bag = train_tagged_words):
    count_t2_t1, count_t1 = t2_given_t1(t2,t1,train_bag)
    return count_t2_t1/count_t1


# In[ ]:


# examples
print(t2_given_t1(t2='NNP', t1='JJ'))
print(t2_given_t1('NN', 'JJ'))
print(t2_given_t1('NN', 'DT'))
print(t2_given_t1('NNP', 'VB'))
print(t2_given_t1(',', 'NNP'))
print(t2_given_t1('PRP', 'PRP'))
print(t2_given_t1('VBG', 'NNP'))
print(t2_given_t1('VB', 'MD'))


# In[ ]:


#Please note P(tag|start) is same as P(tag|'.')
print(t2_given_t1('DT', '.'))
print(t2_given_t1('VBG', '.'))
print(t2_given_t1('NN', '.'))
print(t2_given_t1('NNP', '.'))


# In[ ]:


# creating t x t transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)

tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)): 
    for j, t2 in enumerate(list(T)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]


# In[ ]:


# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
tags_df #row->t1, col->t2


# In[ ]:


# Let's see the prob for tags appearing at start of the sentence represented by tag .
tags_df.loc['.', :]


# In[ ]:


# heatmap of tags matrix
# T(i, j) means P(tag j given tag i)
plt.figure(figsize=(18, 12))
sns.heatmap(tags_df)
plt.show()


# In[ ]:


# frequent tags
# filter the df to get P(t2, t1) > 0.5
tags_frequent = tags_df[tags_df>0.5]
plt.figure(figsize=(18, 12))
sns.heatmap(tags_frequent)
plt.show()


# ## 3. Viterbi Algorithm
# 
# Let's now use the computed probabilities P(w, tag) and P(t2, t1) to assign tags to each word in the document. We'll run through each word w and compute P(tag/w)=P(w/tag).P(tag) for each tag in the tag set, and then assign the tag having the max P(tag/w).
# 
# We'll store the assigned tags in a list of tuples, similar to the list 'train_tagged_words'. Each tuple will be a (token, assigned_tag). As we progress further in the list, each tag to be assigned will use the tag of the previous token.
# 
# Note: P(tag|start) = P(tag|'.') 

# In[ ]:


len(train_tagged_words)


# ### Steps for Viterbi algorithm
# 1. Calculate Transition probability - p(tag|prev tag) <br>
#    - If the word is at the start of the sentence i.e at index 0 :-> t2 given start -> p(t2|start) -> p(t2|.) -> for all t2 in corpus
#    - If word is not at the start of the sentence that is not at index 0 :-> t2 given t1 -> p(t2|prev state) -> p(t2|(tag of previous word in sen)) -> p(t2|state[-1]) -> for all t2 in corpus
# 2. Calculate Emission probability :-> p(word|tag) -> for all tags in the corpus
# 3. Calculate state probability :-> p(tag|word) -> emission_p * transition_p 
# 4. pick the state which have maximum state probability for each word in the sentence
# 
# We basically iterate over the words collection (comes from sentence) and calculate the transition and emission probability for each tag in the corpus (iterating over tags) and based on these two probabilities we record the state probabilities and picks the tag which have the maximun state probility for the given word.

# In[ ]:


# Viterbi Heuristic
def Viterbi(words, train_bag = train_tagged_words):
    state = [] #state/tag for each word
    T = list(set([tag for word,tag in train_bag])) #tags in the corpus
    
    for index, word in enumerate(words):
        #initialise list of probability column for a given observation
        state_probalities = [] #prob for each state/word in corpus for each word
        for t2 in T:
            if index == 0:
                transition_p = t2_given_t1_prob(t2, '.') #transition prob. for start tag
            else:
                t1 = state[-1]
                transition_p = t2_given_t1_prob(t2,t1) #transition prob. for tag t1 followed by t2
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[index], t2)[0]/word_given_tag(words[index], t2)[1]  # p(w|tag) -> count of word with tag t2 / total number of t2
            state_probalities.append(emission_p * transition_p) 
            
        # getting state for which probability is maximum
        state_with_max_prob = T[state_probalities.index(max(state_probalities))] 
        state.append(state_with_max_prob)
    return list(zip(words, state))


# ## 4. Evaluating on Test Set

# In[ ]:


# Running on entire test dataset would take more than 3-4hrs. 
# Let's test our Viterbi algorithm on a few sample sentences of test dataset

random.seed(100)

# choose random 5 sents index from test set
rndom_index = [random.randint(1,len(test_set)) for x in range(50)]

# get the 5 sent from test set using the 5 random_index we picked above
test_run = [test_set[i] for i in rndom_index]

# list of tagged words - this we will use for evaluation purpose
test_run_base = [(word,tag) for sent in test_run for word,tag in sent]

# list of untagged words
test_words = [word for word,tag in test_run_base] 
test_words


# In[ ]:


# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_words)
end = time.time()
difference = end-start


# In[ ]:


print("Time taken in seconds: ", difference)
print(tagged_seq)
#print(test_run_base)


# In[ ]:


# accuracy
check = [(i,j) for i, j in zip(tagged_seq, test_run_base) if i == j] 
accuracy = len(check)/len(tagged_seq)
accuracy


# In[ ]:


incorrect_tagged_cases = [(test_run_base[tagged_seq.index(i)-1],i,j) for i, j in zip(tagged_seq, test_run_base) if i != j] 
incorrect_tagged_cases


# In[ ]:


## Testing
sentence_test = 'Twitter is the best networking social site. Man is a social animal. Data science is an emerging field. Data science jobs are high in demand.'
words = word_tokenize(sentence_test)

start = time.time()
tagged_seq = Viterbi(words)
end = time.time()
difference = end-start


# In[ ]:


print(tagged_seq)
print(difference)


# In[ ]:


sentence = "Donald Trump is the current President of US. Before entering politics, he was a domineering businessman and television personality."
words = word_tokenize(sentence)

tagged_seq = Viterbi(words)
tagged_seq

