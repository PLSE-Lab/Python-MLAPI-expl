#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# To solve this problem we can go back to the basics of Machine Learning, the Naive Bayes Classifier. Specifically, the categorical variant. We will use the classifier in conjunction with counting word distributions. This will give us about 0.51 Public Score and 0.46 Private Score, not that bad for a simple classifier.
# 
# ## Data
# 
# In the dataset, each row holds some text and the author who wrote it. We want to group the rows by author, so that we have what each author wrote stored neatly in one place. To store our data we will use the `pandas` library. Let's go!

# In[ ]:


import pandas as pd

train_x = pd.read_csv("../input/train.csv", sep=',')
test_x = pd.read_csv("../input/test.csv", sep=',')


# We will store everything that the authors wrote in three variable, `EAP`, `HPL` and `MWS`:

# In[ ]:


EAP, HPL, MWS = "", "", ""

for i, row in train_x.iterrows():
    a, t = row['author'], row['text']
    if a == 'EAP':
        EAP += " " + t.lower()
    elif a == 'HPL':
        HPL += " " + t.lower()
    elif a == 'MWS':
        MWS += " " + t.lower()


# Each variable now holds the text written by the respective author in lowercase. Here are the first 50 characters written by Edgar Allan Poe:

# In[ ]:


EAP[:50]


# We now want to convert these long strings into lists of words. To accomplish that, we will make use of the Natural Language Toolkit, `nltk`. Specifically, the function `word_tokenize`:

# In[ ]:


from nltk.tokenize import word_tokenize

EAP = word_tokenize(EAP)
HPL = word_tokenize(HPL)
MWS = word_tokenize(MWS)


# Let's now read the first 10 words written by Poe:

# In[ ]:


EAP[:10]


# ## Counting Distributions
# 
# Now that we have a collection of each author's words, we will create the word count distributions. These, in a nutshell, are dictionaries in the form `{word: word_prob}`.
# 
# For this task we will make use of the default Python library, `collections` and its `Counter` function. The function takes as input a list and outputs a dictionary in the form `{word: word_count}`, where `word_count` is the number of times the word appears in the list.
# 
# If we kept the distributions with the count of each word Naive Bayes would have to operate with very large numbers, which might throw an error. Instead, we will use the frequency of each word which is a number between 0 and 1 denoting the probability of the word appearing from the given author.
# 
# What will happen though if a sentence we want to classify contains a word not in an author's distribution? The word's probability will be 0 which means the whole sentence will have a probability of 0. One single word made the sentence impossible to have come from the particular author. We need to take care of that. So, if a word does not appear in the distribution, we will assign to it a very small probability so that the calculations are not thrown off. We can do that with Python's `defaultdict`, which assigns a default value to keys not contained in the dictionary. In this case, the default value will be the minimum value in the distribution up to that point.

# In[ ]:


from collections import Counter, defaultdict

def create_dist(text):
    c = Counter(text)

    least_common = c.most_common()[-1][1]
    total = sum(c.values())
    
    for k, v in c.items():
        c[k] = v/total

    return defaultdict(lambda: min(c.values()), c)


# Now we will pass the texts through the function to get the counting distributions for each author:

# In[ ]:


c_eap, c_hpl, c_mws = create_dist(EAP), create_dist(HPL), create_dist(MWS)


# ## Naive Bayes
# 
# Now comes the part where we create our Naive Bayes Classifier. Before you continue, I suggest you have a basic understanding on how the algorithm works. I have written a short blog on the subject [here](https://mrdupin.github.io/naive-bayes-cat-intro/).
# 
# Basically, this algorithm works as follows:
# 
# 1. Given input `sentence`.
# 2. Break `sentence` into words.
# 3. Calculate the product of all the words' probabilities for each author.
# 4. Return the products in percentile form
#     1. Add all the authors' products (`total`)
#     2. Divide the individual products by `total`
#     3. Return the results
# 
# So, given a sentence we will return a list in the form `[0.25, 0.65, 0.1]`, where each element shows the probability of the sentence coming from the corresponding author.
# 
# The classifier:

# In[ ]:


import decimal
from decimal import Decimal
decimal.getcontext().prec = 1000

def precise_product(numbers):
    result = 1
    for x in numbers:
        result *= Decimal(x)
    return result

def NaiveBayes(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    Counter distributions and can then be used to find the probability
    of a given item belonging to each class.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the probabilities for each class."""
        def class_prob(target, e):
            attr = attr_dist[target]
            return precise_product([attr[a] for a in e])

        pred = {t: class_prob(t, example) for t in dist.keys()}

        total = sum(pred.values())
        for k, v in pred.items():
            pred[k] = v / total

        return pred

    return predict


# The input to `NaiveBayes` is a dictionary in the form `{Author: CounterDist}`.

# In[ ]:


dist = {'EAP': c_eap, 'HPL': c_hpl, 'MWS': c_mws}


# The classifier returns another function, `predict`, which we can use to make predictions.

# In[ ]:


nBS = NaiveBayes(dist)


# ## Predictions
# 
# Now we can finally start making predictions. We will build a function that takes as input a sentence, converts it to lowercase, breaks it up into words and feeds it to the classifier.

# In[ ]:


def recognize(sentence, nBS):
    return nBS(word_tokenize(sentence.lower()))


# Let's classify a random sentence:

# In[ ]:


recognize("The blood curdling scream echoed across the mansion.", nBS)


# As we can see it is most likely that the sentence came from Lovecraft.
# 
# We now need to automate this process to make predictions for all the rows in `test_x`.

# In[ ]:


def predictions(test_x, nBS):
    d = []
    for index, row in test_x.iterrows():
        i, t = row['id'], row['text']
        p = recognize(t, nBS)
        d.append({'id': i, 'EAP': p['EAP'], 'HPL': p['HPL'], 'MWS': p['MWS']})
    
    return pd.DataFrame(data=d)


# The `predictions` function builds a new `pandas` DataFrame with columns `['id', 'EAP', 'HPL', 'MWS']`, where the values are the probabilities returned from the classifier for a specific `id`.
# 
# Let's make our submission:

# In[ ]:


submission = predictions(test_x, nBS)
submission.to_csv('submission.csv', index=False)


# ## Conclusion
# 
# And that's about it! Without much ado we managed to make a respectable ~0.5 submission. With a little tweaking, we could increase this slightly (for example, we could say that if an author has a probability >0.95, we will go on a limp and assume that author has a probability of 1.0; most likely we are correct and we decrease our error by about 0.05).
# 
# We could go a bit more in-depth and instead of creating one-word distributions, we created two-word distributions. A lot of authors might use two words together very often, which is a useful feature. But for this example that will do.
# 
# Cheers!
