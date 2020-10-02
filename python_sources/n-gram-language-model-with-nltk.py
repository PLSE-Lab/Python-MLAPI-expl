#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U pip')
get_ipython().system('pip install -U dill')
get_ipython().system('pip install -U nltk==3.4')


# # N-grams Language Models (N-grams LM)
# 
# Nowadays, everything seems to be going neural... 
# 
# Traditionally, we can use n-grams to generate language models to predict which word comes next given a history of words. 
# 
# We'll use the `lm` module in `nltk` to get a sense of how non-neural language modelling is done.
# 
# (**Source:** The content in this notebook is largely based on [language model tutorial in NLTK documentation by Ilia Kurenkov](https://github.com/nltk/nltk/blob/develop/nltk/lm/__init__.py))

# In[ ]:


from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten


# If we want to train a bigram model, we need to turn this text into bigrams. Here's what the first sentence of our text would look like if we use the `ngrams` function from NLTK for this.

# In[ ]:


text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]


# In[ ]:


list(bigrams(text[0]))


# In[ ]:


list(ngrams(text[1], n=3))


# Notice how "b" occurs both as the first and second member of different bigrams but "a" and "c" don't? 
# 
# Wouldn't it be nice to somehow indicate how often sentences start with "a" and end with "c"?
# 
# 
# A standard way to deal with this is to add special "padding" symbols to the sentence before splitting it into ngrams. Fortunately, NLTK also has a function for that, let's see what it does to the first sentence.
# 

# In[ ]:


from nltk.util import pad_sequence
list(pad_sequence(text[0],
                  pad_left=True, left_pad_symbol="<s>",
                  pad_right=True, right_pad_symbol="</s>",
                  n=2)) # The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc. 


# In[ ]:


padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>", 
                                pad_right=True, right_pad_symbol="</s>", n=2))
list(ngrams(padded_sent, n=2))


# In[ ]:


list(pad_sequence(text[0],
                  pad_left=True, left_pad_symbol="<s>",
                  pad_right=True, right_pad_symbol="</s>",
                  n=3)) # The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc. 


# In[ ]:


padded_sent = list(pad_sequence(text[0], pad_left=True, left_pad_symbol="<s>", 
                                pad_right=True, right_pad_symbol="</s>", n=3))
list(ngrams(padded_sent, n=3))


# Note the `n` argument, that tells the function we need padding for bigrams.
# 
# Now, passing all these parameters every time is tedious and in most cases they can be safely assumed as defaults anyway.
# 
# Thus the `nltk.lm` module provides a convenience function that has all these arguments already set while the other arguments remain the same as for `pad_sequence`.

# In[ ]:


from nltk.lm.preprocessing import pad_both_ends
list(pad_both_ends(text[0], n=2))


# Combining the two parts discussed so far we get the following preparation steps for one sentence.

# In[ ]:


list(bigrams(pad_both_ends(text[0], n=2)))


# To make our model more robust we could also train it on unigrams (single words) as well as bigrams, its main source of information.
# NLTK once again helpfully provides a function called `everygrams`.
# 
# While not the most efficient, it is conceptually simple.

# In[ ]:


from nltk.util import everygrams
padded_bigrams = list(pad_both_ends(text[0], n=2))
list(everygrams(padded_bigrams, max_len=2))


# We are almost ready to start counting ngrams, just one more step left.
# 
# During training and evaluation our model will rely on a vocabulary that defines which words are "known" to the model.
# 
# To create this vocabulary we need to pad our sentences (just like for counting ngrams) and then combine the sentences into one flat stream of words.
# 

# In[ ]:


from nltk.lm.preprocessing import flatten
list(flatten(pad_both_ends(sent, n=2) for sent in text))


# In most cases we want to use the same text as the source for both vocabulary and ngram counts.
# 
# Now that we understand what this means for our preprocessing, we can simply import a function that does everything for us.

# In[ ]:


from nltk.lm.preprocessing import padded_everygram_pipeline
train, vocab = padded_everygram_pipeline(2, text)


# So as to avoid re-creating the text in memory, both `train` and `vocab` are lazy iterators. They are evaluated on demand at training time.
# 
# For the sake of understanding the output of `padded_everygram_pipeline`, we'll "materialize" the lazy iterators by casting them into a list.

# In[ ]:


training_ngrams, padded_sentences = padded_everygram_pipeline(2, text)
for ngramlize_sent in training_ngrams:
    print(list(ngramlize_sent))
    print()
print('#############')
list(padded_sentences)


# ## Lets get some real data and tokenize it

# In[ ]:


try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    # Testing whether it works. 
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize


# In[ ]:


import os
import requests
import io #codecs


# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)


# In[ ]:


# Tokenize the text.
tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(text)]


# In[ ]:


tokenized_text[0]


# In[ ]:


print(text[:500])


# In[ ]:


# Preprocess the tokenized text for 3-grams language modelling
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


# # Training an N-gram Model

# Having prepared our data we are ready to start training a model. As a simple example, let us train a Maximum Likelihood Estimator (MLE).
# 
# We only need to specify the highest ngram order to instantiate it.

# In[ ]:


from nltk.lm import MLE
model = MLE(n) # Lets train a 3-grams model, previously we set n=3


# Initializing the MLE model, creates an empty vocabulary

# In[ ]:


len(model.vocab)


# ... which gets filled as we fit the model.

# In[ ]:


model.fit(train_data, padded_sents)
print(model.vocab)


# In[ ]:


len(model.vocab)


# The vocabulary helps us handle words that have not occurred during training.

# In[ ]:


print(model.vocab.lookup(tokenized_text[0]))


# In[ ]:


# If we lookup the vocab on unseen sentences not from the training data, 
# it automatically replace words not in the vocabulary with `<UNK>`.
print(model.vocab.lookup('language is never random lah .'.split()))


# Moreover, in some cases we want to ignore words that we did see during training but that didn't occur frequently enough, to provide us useful information. 
# 
# You can tell the vocabulary to ignore such words using the `unk_cutoff` argument for the vocabulary lookup, To find out how that works, check out the docs for the [`nltk.lm.vocabulary.Vocabulary` class](https://github.com/nltk/nltk/blob/develop/nltk/lm/vocabulary.py)

# **Note:** For more sophisticated ngram models, take a look at [these objects from `nltk.lm.models`](https://github.com/nltk/nltk/blob/develop/nltk/lm/models.py):
# 
#  - `Lidstone`: Provides Lidstone-smoothed scores.
#  - `Laplace`: Implements Laplace (add one) smoothing.
#  - `InterpolatedLanguageModel`: Logic common to all interpolated language models (Chen & Goodman 1995).
#  - `WittenBellInterpolated`: Interpolated version of Witten-Bell smoothing.

# # Using the N-gram Language Model

# When it comes to ngram models the training boils down to counting up the ngrams from the training corpus.

# In[ ]:


print(model.counts)


# This provides a convenient interface to access counts for unigrams...

# In[ ]:


model.counts['language'] # i.e. Count('language')


# ...and bigrams for the phrase "language is"

# In[ ]:


model.counts[['language']]['is'] # i.e. Count('is'|'language')


# ... and trigrams for the phrase "language is never"

# In[ ]:


model.counts[['language', 'is']]['never'] # i.e. Count('never'|'language is')


# And so on. However, the real purpose of training a language model is to have it score how probable words are in certain contexts.
# 
# This being MLE, the model returns the item's relative frequency as its score.

# In[ ]:


model.score('language') # P('language')


# In[ ]:


model.score('is', 'language'.split())  # P('is'|'language')


# In[ ]:


model.score('never', 'language is'.split())  # P('never'|'language is')


# Items that are not seen during training are mapped to the vocabulary's "unknown label" token.  This is "<UNK>" by default.
# 

# In[ ]:


model.score("<UNK>") == model.score("lah")


# In[ ]:


model.score("<UNK>") == model.score("leh")


# In[ ]:


model.score("<UNK>") == model.score("lor")


# To avoid underflow when working with many small score values it makes sense to take their logarithm. 
# 
# For convenience this can be done with the `logscore` method.
# 

# In[ ]:


model.logscore("never", "language is".split())


# # Generation using N-gram Language Model

# One cool feature of ngram models is that they can be used to generate text.

# In[ ]:


print(model.generate(20, random_seed=7))


# We can do some cleaning to the generated tokens to make it human-like.

# In[ ]:


from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


# In[ ]:


generate_sent(model, 20, random_seed=7)


# In[ ]:


print(model.generate(28, random_seed=0))


# In[ ]:


generate_sent(model, 28, random_seed=0)


# In[ ]:


generate_sent(model, 20, random_seed=1)


# In[ ]:


generate_sent(model, 20, random_seed=30)


# In[ ]:


generate_sent(model, 20, random_seed=42)


# # Saving the model 
# 
# The native Python's pickle may not save the lambda functions in the  model, so we can use the `dill` library in place of pickle to save and load the language model.
# 

# In[ ]:


import dill as pickle 

with open('kilgariff_ngram_model.pkl', 'wb') as fout:
    pickle.dump(model, fout)


# In[ ]:


with open('kilgariff_ngram_model.pkl', 'rb') as fin:
    model_loaded = pickle.load(fin)


# In[ ]:


generate_sent(model_loaded, 20, random_seed=42)


# # Lets try some generating with Donald Trump data!!!
# 
# 
# **Dataset:** https://www.kaggle.com/kingburrito666/better-donald-trump-tweets#Donald-Tweets!.csv
# 
# 
# In this part, I'll be munging that data as how I would be doing it at work. 
# I've really no seen the data before but I hope this session would be helpful for you to see how to approach new datasets with the skills you have.

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/Donald-Tweets!.csv')
df.head()


# In[ ]:


trump_corpus = list(df['Tweet_Text'].apply(word_tokenize))


# In[ ]:


# Preprocess the tokenized text for 3-grams language modelling
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, trump_corpus)


# In[ ]:


from nltk.lm import MLE
trump_model = MLE(n) # Lets train a 3-grams model, previously we set n=3
trump_model.fit(train_data, padded_sents)


# In[ ]:


generate_sent(trump_model, num_words=20, random_seed=42)


# In[ ]:


generate_sent(trump_model, num_words=10, random_seed=0)


# In[ ]:


generate_sent(trump_model, num_words=50, random_seed=10)


# In[ ]:


print(generate_sent(trump_model, num_words=100, random_seed=52))


# In[ ]:




