#!/usr/bin/env python
# coding: utf-8

# # Wine2Vec Exploration
# ##### By Zack Thoutt
# 
# Here is a little data exploration of my new wine review dataset using word2vec. My theory is that the words a sommelier would use to describe a wine (oaky, tannic, acidic, berry, etc.) can be used to predict the type of wine (Pinot Noir, Cabernet Sav., etc.). Let's see if we can extract some interesting relationships from the data and somewhat validate this theory.

# In[1]:


from collections import Counter
import numpy as np
import nltk
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v


# ---
# 
# ### Get the Data
# The dataset can be found on [Kaggle](https://www.kaggle.com/zynicide/wine-reviews) or you can run my sraper on [Github](https://github.com/zackthoutt/wine-deep-learning).

# In[2]:


data = pd.read_csv('../input/winemag-data_first150k.csv')


# In[3]:


labels = data['variety']
descriptions = data['description']


# ### Explore the Data
# There are several hundred fairly common varietals of wine and probably thousands of other niche varietals. It will be difficult to be able to identify them all, but I hypothesize that it should be possible to classify the most common, say, 50 or 100 wine varietals with this wine review dataset. 
# 
# Let's take a look at a few reviews and see if we as humans can tell a difference in the descriptive words used for different types of wine.

# In[4]:


print('{}   :   {}'.format(labels.tolist()[0], descriptions.tolist()[0]))
print('{}   :   {}'.format(labels.tolist()[56], descriptions.tolist()[56]))
print('{}   :   {}'.format(labels.tolist()[93], descriptions.tolist()[93]))


# Even if you're not someone who knows wine, I think that there is a pretty clear distinction in the descriptions of these different types of wines. The Cabernet Sauvignon (a red wine) was described with words like cherry, tannin and carmel. The next two reviews are white wines, but even they show differences in their description. The sauvignon blanc is described as minerally, citrus, and green fruits while the chardonnay is described as smokey, earthy, crisp-apple, and buttery. This provides us with good motivation to move forward and explore the data more.
# 
# One of the limitations that I think we will have with this dataset is that there will be a lot more reviews for popular wine varietals than less popular wine varietals. This isn't bad neccissarily, but it means that we will probably only be able to classify the most popular N varietals.

# In[5]:


varietal_counts = labels.value_counts()
print(varietal_counts[:50])


# If you drink wine regularly you will probably recognize the most reviewed wines listed above. The value counts for different wine varietals does verify my theory that less popular wines might not have enough reviews to classify them. The most popular wine varietals have thousands of reviews, but even towards the bottom end of the top 50 wine varietals there are only a few hundred reviews. This isn't a problem for building a word2vec model like we are going to do next, but it is something to keep in mind as we move forward trying to create a wine classifier.

# ### Word2Vec Model
# ##### Formatting the Data
# In order to train a word2vec model, all of the description data will need to be concatenated into one giant string. 

# In[6]:


corpus_raw = ""
for description in descriptions:
    corpus_raw += description


# Next, we need to tokenize the wine corpus using NLTK. This process will essentially break the word corpus into an array of sentences and then break each sentence into an array of words stripping out less usefull characters like commas and hyphens in the process. In this way, we are able to train the word2vec model with the context of sentences and relative word placement. 

# In[7]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[8]:


raw_sentences = tokenizer.tokenize(corpus_raw)


# In[9]:


def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[10]:


sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


# In[11]:


print(raw_sentences[234])
print(sentence_to_wordlist(raw_sentences[234]))


# In[12]:


token_count = sum([len(sentence) for sentence in sentences])
print('The wine corpus contains {0:,} tokens'.format(token_count))


# For some context, all of the GOT books combined make up only ~1,800,000 tokens, so this dataset is nearly 4x as large as the GOT book series.
# 
# ##### Training the Model
# It took some experimenting to get the model to train well. The main things hyperparameters that I had to tune were `min_word_count` and `context_size`. 
# 
# I usually train word2vec models with a `min_word_count` closer to 3-5, but since this dataset is so large I had to bump it up to 10. When I was training the model on a smaller `min_word_count` I was getting a lot of winery and vinyard noise in my word similarities (ie the words most similar to "cherry" were a bunch of foreign vinyards, wineries, regions, etc.). After looking through some of the descriptions I came to the conclusion that most of the wine descriptions don't mention the wine varietal, vinyard, or winery, but some do. So I played with the `min_word_count` until those rare instances had less of an effect on the model.
# 
# I also had to play with the `context_size` quite a bit. 10 is a pretty large context size, but it makes sense here because really all of the words in a sentence are related to each other in the context of wine descriptions and what were are trying to accomplish. I might even experiment with bumping the `context_size` up higher at some point, but even now most of the words in each sentence will be associated with each other in the model.

# In[13]:


num_features = 300
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993


# In[14]:


wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


# In[15]:


wine2vec.build_vocab(sentences)


# In[16]:


print('Word2Vec vocabulary length:', len(wine2vec.wv.vocab))


# In[17]:


print(wine2vec.corpus_count)


# In[18]:


wine2vec.train(sentences, total_examples=wine2vec.corpus_count, epochs=wine2vec.iter)


# ### Playing with the Model
# Now that we have a trained model we can get to the fun part and start playing around with the results. As you can tell from the outputs below, there is definitely still some noise in the data that could be worked out by tuning the parameters further, but overall we are getting pretty good results.
# 
# ##### Words closest to a given word
# "melon," "berry," and "oak" are words that someone might use to describe the taste/smell of a wine.

# In[19]:


wine2vec.most_similar('melon')


# In[20]:


wine2vec.most_similar('berry')


# In[21]:


wine2vec.most_similar('oak')


# Another thing that someone might use to describe a wine is how acidic it is

# In[22]:


wine2vec.most_similar('acidic')


# Or what the body is like. "full-bodied" would be something that is thick like whole milk while "light-bodied" would be something that is thin like skim milk.

# In[23]:


wine2vec.most_similar('full')


# Finally, you can also feel in your mouth how much tannin a wine has. Wines with lots of tannis give you a dry, furry feeling on your tounge.

# In[24]:


wine2vec.most_similar('tannins')


# ##### Linear relationships between word pairs

# In[25]:


def nearest_similarity_cosmul(start1, end1, end2):
    similarities = wine2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


# In[26]:


nearest_similarity_cosmul('oak', 'vanilla', 'cherry');


# In[27]:


nearest_similarity_cosmul('full', 'berry', 'light');


# In[28]:


nearest_similarity_cosmul('tannins', 'plum', 'fresh');


# In[29]:


nearest_similarity_cosmul('full', 'bodied', 'acidic');


# ### Conclusion
# I think that exploring this wine2vec model has helped validate the theory that there is a lot of useful data in these wine descriptions that can probably be used to classify wine varietals. I have not yet trained any classifiers, but we saw early on that descriptions of different wines used different words to describe the wine varietals, and based on our wine2vec model there is definitley enough context to link these descriptive words together and come up with something to classify them when they are used in certain combinations.
# 
# That's all I have for now. As always, let me know if anyone has any questions, comments, insights, ideas, etc. I'll be posting more of my analyses and models soon!
