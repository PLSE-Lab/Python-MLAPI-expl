#!/usr/bin/env python
# coding: utf-8

# # Creating a POS Tagger

# We can train a classifier to work out which suffixes are most informative for POS tagging. We can begin by finding out what the most common suffixes are

# In[ ]:


from nltk.corpus import brown
from nltk import FreqDist

suffix_fdist = FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
    
suffix_fdist


# In[ ]:


common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
common_suffixes[:10]


# Next, we'll define a feature extractor function which checks a given word for these suffixes:

# In[ ]:


def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

pos_features('test')


# Now that we've defined our feature extractor, we can use it to train a new decision tree classifier:

# In[ ]:


tagged_words = brown.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
featuresets[0]


# In[ ]:


from nltk import DecisionTreeClassifier
from nltk.classify import accuracy

cutoff = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]


# In[ ]:


classifier = DecisionTreeClassifier.train(train_set) # NLTK is a teaching toolkit which is not really optimized for speed. Therefore, this may take forever. For speed, use scikit-learn for the classifiers.


# In[ ]:


accuracy(classifier, test_set)


# In[ ]:


classifier.classify(pos_features('cats'))


# In[ ]:


classifier.pseudocode(depth=4)


# To improve the classifier, we can add contextual features:
# 
# ```py
# def pos_features(sentence, i): [1]
#     features = {"suffix(1)": sentence[i][-1:],
#                 "suffix(2)": sentence[i][-2:],
#                 "suffix(3)": sentence[i][-3:]}
#     if i == 0:
#         features["prev-word"] = "<START>"
#     else:
#         features["prev-word"] = sentence[i-1]
#     return features
# ```
# 
# Then, instead of working with tagged words, we work with tagged sentences:
# ```py
# tagged_sents = brown.tagged_sents(categories='news')
# ```
# 
# We can then improve this further by adding more features such as `prev-tag` etc.
