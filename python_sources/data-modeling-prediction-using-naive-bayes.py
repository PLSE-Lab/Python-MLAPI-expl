#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Download data from google drive. You need not mess with this code.

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                
if __name__ == "__main__":
    file_id = '1e_Azf9zGvSWsDhM9PP2sfMNKC72-iWAK'
    destination = 'data.txt'
    download_file_from_google_drive(file_id, destination)


# In[ ]:


with open('data.txt', 'r') as f:
  data_raw = f.readlines()


# 1. Data preparation
# Now the entire data is stored in the list ```data_raw```. 
# Every line in the file is a different element of the list.
# First let us look at the first five elements of the list. 
# 

# We will first write a function that returns first five elements of the list if length of list is greater than or equal to 5 and None value otherwise.

# In[ ]:


def first_five_in_list(l):
  """
  Inputs: 
  l: Python list

  Outputs:
  l_5 : python list, first five elements of list if length of list greater than 5; None otherwise
  """
  l_5 = [] 
  if len(l) >= 5:
    for i in range(0,5):
      l_5.append(l[i])
    return l_5
  else:
    return None


# ####1.2

# You can see that the first five elements in our list look like this - 
# <img src="https://drive.google.com/uc?id=1JnA0TxI-jWR4mJHYztnQQZZNRRTacsaa"> \\
# <br>
# You can clearly see that each line ends with a newline character. We want to remove these new line characters. \\
# Now we will write a function that removes all extra newline characters (number of newline characters maybe greater than or equal to 0) at the end of any string that is passed to it.

# In[ ]:


def remove_trailing_newlines(s):
  """
  Function that removes all trailing newlines at the end of it
  Inputs:
    s : string

  Outputs:
    s_clean : string, string s but without newline characters at the end 
  """
  s_clean = s.strip('\n')
  return s_clean


# If we apply ```remove_trailing_newlines``` to first element of data_raw, we get <br>
# <img src="https://drive.google.com/uc?id=1vu-awFwqGC9sNgk-QgHMIamnFp6d996R"> <br>
# You can see that the newline at the end has disappeared. <br>
# 

# ####1.3

# But we now we need to apply this function to the whole list. \\
# We will write a function named mapl, that takes two arguments - a function on elements of type $t$ and a list $l$ of elements of type $t$ and applies the function over all elements of the list $l$ and returns them as a list. 

# In[ ]:


def mapl(f, l):
  """
  Function that applies f over all elements of l
  Inputs:
    f : function, f takes elements of type t1 and returns elements of type t2
    l : list, list of elements of type t1

  Ouptuts:
    f_l : list, list of elements of type t2 obtained by applying f over each element of l
  """
  f_l = []
  for i in range(0,len(l)):
    f_l.append(f(l[i]))

  return f_l


# Now we can use mapl to apply remove_trailing_newlines to all lines in data_raw

# In[ ]:


data_clean = mapl(remove_trailing_newlines, data_raw)


# First five elements of data_clean look like this: <br>
# <img src = "https://drive.google.com/uc?id=17U7h87_7VjZs5CpRtJawHUfM5O1EztcS">

# This is a dataset of text messages. We have to classify this into spam or ham. Ham means non-spam relevant text messages. More details can be found here - http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection <br>
# <br>
# You can see that each line starts by specifying whether the message is ```ham``` or ```spam``` and then there is a tab character, ```\t``` followed by actual text message.
# <br>
# Now we need to split the lines to extract the two components - data label (```ham``` or ```spam```) and data sample (the text message).

# ####1.4

# Now we will write a function ```split_at_s``` that takes two strings - ```text``` and ```s```. <br>
# It splits the string text into two parts at the first occurence of s. <br>
# Then it wraps both parts in a tuple and returns it.

# In[ ]:


def split_at_s(text, s):
  """Function that splits string text into two parts at the first occurence of string s
  Inputs:
    text: string, string to be split
    s : string, string of length 1 at which to split
  
  Outputs:
    split_text: tuple of size 2, contains text split in two (do not include the string s at which split occurs in any of the split parts) 
  """
  s1 = text[0:text.index(s)]
  s2 = text[text.index(s) + 1:len(text)]
  split_text = (s1,s2)
  return split_text


# Python has a very handy feature used to define short functions called lambda expressions. This is from official python docs <br>
# <img src = "https://drive.google.com/uc?id=1kBLTdhWT6SNrhlk7vYH7ql2KxcSKvu29">

# Use lambda expressions and ```split_at_s``` to write a function, ```split_at_tab``` that takes only one argument - ```text``` and splits at the first occurence of ```'\t'``` character. (If you can't understand lambda expressions, just define the function in the ususal way)

# In[ ]:


split_at_tab = lambda text: split_at_s(text,'\t')


# After splitting at '\t' character, one data point looks like this - <br>
# <img src = "https://drive.google.com/uc?id=1TVblRl9K_4HFLOncSWZW20u3ztsfoCzY">
# 

# ####1.5

# Now apply split_at_tab function over elements of list ```data_clean``` and assign it to variable named ```data_clean2```

# In[ ]:


data_clean2 = []
for i in range(0,len(data_clean)):
  data_clean2.append(split_at_tab(data_clean[i]))


# Now let us remove the punctuations in an sms.

# In[ ]:


import string
def remove_punctuations_and_lower(text):
  """Function that removes punctuations in a text
  Inputs:
    text: string
  Outputs:
    text_wo_punctuations
  """
  return (text.translate(str.maketrans("","", string.punctuation))).lower()


# ####1.6

# Now use the function remove_punctuations to remove punctuations from the text part of all of the tuples in ```data_clean2``` and assign it to a variable named ```dataset```

# In[ ]:


dataset = []
for i in range(0,len(data_clean2)):
  t = data_clean2[i]
  f = []
  for j in range(0,2):
   f.append(remove_punctuations_and_lower(t[j]))
  dataset.append(tuple(f))


# First 5 elements of ```dataset``` look like this now. <br>
# <img src="https://drive.google.com/uc?id=19TomF6uXvsFALRsX6KLOPseKzxADMgRp">

# Now let us count number of occurences of ```ham``` and ```spam``` in our dataset.

# ####1.7 

# Now we will write a function ```counter``` that takes two arguments - 
# - a list $l$ of elements of type $t$
# - a function $f: t \rightarrow u$ (means $f$ takes an argument of type $t$ and returns values of type $u$)
# 
# Counter returns a dictionary whose keys are $u_1, u_2, \ldots etc$ - unique values of type $u$ obtained by applying $f$ over elements of $l$. <br>
# The values corresponding to the keys are the the number of times a particular key say $u_1$ is obtained when we apply $f$ over elements of $l$

# In[ ]:


def counter(l, f):
  """
  Function that returns a dictionary of counts of unique values obtained by applying f over elements of l
  Inputs:
    l: list; list of elements of type t
    f: function; f takes arguments of type t and returns values of type u
  
  Outputs:
    count_dict: dictionary; keys are elements of type u, values are ints
  """
  count_dict = {}
  t = []
  for i in l:
    t.append(f(i))
  r = set(t)
  for i in r:
    count_dict[i] = t.count(i)
  return count_dict


# ####1.8

# A function named ```aux_func``` can be passed to ```counter``` along with the list ```dataset``` to get a dictionary containing counts of ```ham``` and ```spam``` 

# In[ ]:


def aux_func(i):
  return(i[0])
counter(dataset,aux_func)


# The counts of ```ham``` and ```spam``` as we can see are ```{'ham': 4827, 'spam': 747}```

# Now let us split our dataset into training and test sets. We'll first shuffle the elements of the dataset, then we'll use 80% of data for training and 20% for testing.

# ####1.9

# Now we will write a function that takes a list, randomly shuffles it and then returns it. <br>
# (Use the random library of python - https://docs.python.org/3/library/random.html)

# In[ ]:


def random_shuffle(l):
  import random
  """Function that returns a randomly shuffled list
  Inputs:
    l: list
  Outputs:
    l_shuffled: list, contains same elements as l but randomly shuffled
  """
  random.shuffle(l)
  l_shuffled = [] 
  for i in l:
    l_shuffled.append(i)
  return l_shuffled


# 
# 1.10

# Now split the shuffled list. Take 80% (4459) samples and assign them to a variable called ```data_train``` . Put the rest in a variable called ```data_test```

# In[ ]:


n = random_shuffle(dataset)
l1 = (int)(0.8 * len(n))
data_train = []
data_test = []
for i in range(0,l1):
  data_train.append(n[i])
for i in range(l1,len(n)):
  data_test.append(n[i])


# ### 2.Data Modeling
# We shall use Naive Bayes for modelling our classifier. You can read about Naive Bayes from here (https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes). But you don't actually need to read it, because we are going to move step by step in building this classifier.

# First we need to find the probabilities $P(w_i | C)$ <br>
# We read $P(A | B)$ as probability of event A, given event B. <br>
# $P(w_i | C)$ is probability that word $w_i$ occurs in the sms given that the sms belongs to class $C$ where $C$ can be either ```spam``` or ```ham``` .
# <br>
# But we will be finding $\tilde{P}(w_i|C)$ which is the smoothed probability function to take care of words with 0 probabilities that may cause problems.

# $\tilde{P}(w_i|C) = \frac{\text{Number of occurences of } w_i \text{ in all samples of class C} + 1}{\text{Total number of words in all samples of class C } + \text{ Vocabulary size}}$

# ####2.1

# Find the vocabulary - list of unique words in all smses of ```data_train``` and assign it to the variable ```vocab```

# In[ ]:


vocab = []
for i in data_train:
  j = i[1].split(" ")
  for k in j:
    if k not in vocab:
      vocab.append(k)


# #### 2.2

# For every word $w_i$ in vocab, find the count (total number of occurences) of $w_i$ in all smses of type ```spam```. Put these counts in a dictionary and assign it to a variable named ```dict_spam``` where key is the word $w_i$ and value is the count. <br>
# In a similar way, create a variable called ```dict_ham``` which contains counts of each word in vocabulary in smses of type ```ham```. (This is only w.r.t samples in ```data_train```) 

# In[ ]:


dict_spam = {}
dict_ham = {}

for i in vocab:
  dict_spam[i] = 0
  dict_ham[i] = 0

for i in vocab:
  count = 0
  for j in data_train:
    if(j[0] == "spam"):
      t = j[1].split(" ")
      for k in t:
        if(i == k):
          count += 1
  dict_spam[i] = count

for i in vocab:
  count = 0
  for j in data_train:
    if(j[0] == "ham"):
      t = j[1].split(" ")
      for k in t:
        if(i == k):
          count +=1
  dict_ham[i] = count


# ####2.3

# For every word $w_i$ in vocab, find the smoothed probability $\tilde{P}(w_i | \text{ spam })$ and put in a dictionary named ```dict_prob_spam```. 
# In a similar way, define the dictionary ```dict_prob_ham``` which contains smoothed probabilities $\tilde{P}(w_i | \text{ ham })$

# In[ ]:


dict_prob_spam = {}
dict_prob_ham = {}
sumspam = 0
sumham = 0
sumspam = sum(dict_spam.values())
sumham = sum(dict_ham.values())
for i in vocab:
  dict_prob_spam[i] = (dict_spam[i] + 1)/(len(vocab) + sumspam) 
for i in vocab:
  dict_prob_ham[i] = (dict_ham[i] + 1)/(len(vocab) + sumham)


# ###3. Prediction

# We need to test our model on ```data_test``` . 
# For each sample of ```data_test```, prediction procedure is as follows: 
# - For all words common to the sample and vocabulary, find ```spam_score``` and ```ham_score```
# - If ```spam_score``` is higher than ```ham_score```, then we predict the sample to be spam and vice versa.
# - ```spam_score``` = $P(spam)*\tilde{P}(w_1 | \text{ spam }) *  \tilde{P}(w_2 | \text{ spam }) * \ldots$ where $w_1, w_2, \ldots$ are words which occur both in the test sms and vocabulary.
# - Similary, ```ham_score``` = $P(ham)*\tilde{P}(w_1 | \text{ ham }) *  \tilde{P}(w_2 | \text{ ham }) * \ldots$ where $w_1, w_2, \ldots$ are words which occur both in the test sms and vocabulary. <br>
# Here $P(spam) = \frac{\text{Number of samples of type spam in training set}}{\text{Total number of samples in training set}}$ <br>
# Similarly, $P(ham) = \frac{\text{Number of samples of type ham in training set}}{\text{Total number of samples in training set}}$ <br>
# (Note: The above is prediction procedure for a single sample in data_test) <br>
# Now we will write a function ```predict``` which does this.

# ####3.1

# In[ ]:


def predict(text, dict_prob_spam, dict_prob_ham, data_train):
  """Function which predicts the label of the sms
  Inputs:
    text: string, sms
    dict_prob_spam: dictionary, contains dict_prob_spam as defined above
    dict_prob_spam: dictionary, contains dict_prob_ham as defined above
    data_train: list, list of tuples of type(label, sms), contains training dataset

  Outputs:
    prediction: string, one of two strings - either 'spam' or 'ham'
  """
  prediction = ''
  f = text.split(" ")
  d = counter(data_train,aux_func)
  spam_score = d["spam"]/len(data_train)
  ham_score = d["ham"]/len(data_train)
  for j in f:
    if j in vocab:
      spam_score = spam_score*dict_prob_spam[j]
      ham_score = ham_score*dict_prob_ham[j]
  if spam_score > ham_score:
    prediction = 'spam'
  else:
    prediction = 'ham'
  return prediction


# ####3.2

# Now find accuracy of the model. Apply function predict to all the samples in data_test. <br>
# $\text{accuracy} = \frac{\text{number of correct predictions}}{\text{size of test set}}$ <br>
# Now we will write a function accuracy which applies predict to all ```samples``` in data_test and returns ```accuracy```

# In[ ]:


def accuracy(data_test, dict_prob_spam, dict_prob_ham, data_train):
  """Function which finds accuracy of model
  Inputs:
    data_test: list, contains tuples of data (label, sms) 
    dict_prob_spam: dictionary, contains dict_prob_spam as defined above
    dict_prob_spam: dictionary, contains dict_prob_ham as defined above
    data_train: list, list of tuples of type(label, sms), contains training dataset


  Outputs:
    accuracy: float, value of accuracy
  """
  c = 0
  for i in data_test:
    o = i[0]
    p = predict(i[1],dict_prob_spam, dict_prob_ham, data_train)
    if p == o:
      c += 1
  accuracy = c/len(data_test)
  return accuracy
accuracy(data_test,dict_prob_spam, dict_prob_ham, data_train)

