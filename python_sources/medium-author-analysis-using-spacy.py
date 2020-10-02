#!/usr/bin/env python
# coding: utf-8

# # Background
# - I want to analyze the text with SpaCy.
# - The documentation of SpaCy is very helpful! You can check it! https://spacy.io/\
# - I referred to this kernel, https://www.kaggle.com/enerrio/scary-nlp-with-spacy-and-keras. Thanks to [Aaron Marques](https://www.kaggle.com/enerrio)

# # Read dataset

# In[ ]:


#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#print(os.listdir("../input"))
import spacy
import random 
from collections import Counter #for counting
import seaborn as sns #for visualization
import matplotlib.pyplot as plt

plt.style.use('seaborn')
sns.set(font_scale=2)


# In[ ]:


articles = pd.read_csv('../input/articles.csv')


# # Read text using spacy and extract tokens

# In[ ]:


nlp = spacy.load('en')


# - The article is a bit long. Let's use some part of the article.

# In[ ]:


doc = nlp(articles['text'][0][:500]) 


# ## Store the informations of tokens

# In[ ]:


df_token = pd.DataFrame()

for i, token in enumerate(doc):
    df_token.loc[i, 'text'] = token.text
    df_token.loc[i, 'lemma'] = token.lemma_,
    df_token.loc[i, 'pos'] = token.pos_
    df_token.loc[i, 'tag'] = token.tag_
    df_token.loc[i, 'dep'] = token.dep_
    df_token.loc[i, 'shape'] = token.shape_
    df_token.loc[i, 'is_alpha'] = token.is_alpha
    df_token.loc[i, 'is_stop'] = token.is_stop


# In[ ]:


df_token


# - As you can see, the tokens and relevant information are extraced very easily.

# # Visualize the structure of sentence

# - Using displacy with keyword "dep",  we can visulize the structure of sentences easily.

# In[ ]:


from spacy import displacy


# In[ ]:


sentence_spans = list(doc.sents)
displacy.render(sentence_spans, style='dep', jupyter=True)


# - As you can see, sentences are well-divided.

# # Find entity

# - Spacy have built-in entity-types
# 
# 
# | Type | Description | 
# |:--------|:--------|
# | PERSON | People, including fictional. | 
# | NORP | Nationalities or religious or political groups. | 
# | FAC | Buildings, airports, highways, bridges, etc. | 
# | ORG | Companies, agencies, institutions, etc. | 
# | GPE | Countries, cities, states. | 
# | LOC | Non-GPE locations, mountain ranges, bodies of water. | 
# | PRODUCT | Objects, vehicles, foods, etc. (Not services.) | 
# | EVENT | Named hurricanes, battles, wars, sports events, etc. | 
# | WORK_OF_ART | Titles of books, songs, etc. | 
# | LAW | Named documents made into laws. | 
# | LANGUAGE | Any named language. | 
# | DATE | Absolute or relative dates or periods. | 
# | TIME | Times smaller than a day. | 
# | PERCENT | Percentage, including "%". | 
# | MONEY | Monetary values, including unit. | 
# | QUANTITY | Measurements, as of weight or distance. | 
# | ORDINAL | "first", "second", etc. | 
# | CARDINAL | Numerals that do not fall under another type | 

# - You can see the tables, in this URL. https://spacy.io/usage/linguistic-features#section-named-entities
# - Ok, let's find the entities using SpaCy and visualize.

# In[ ]:


spacy.displacy.render(doc, style='ent',jupyter=True)


# # Authors 

# In[ ]:


articles['author'].value_counts()


# - As you can see, there are many authors. 
# - Let's analyze the top 5 authors.

# In[ ]:


from nltk.corpus import stopwords
import string
stopwords = stopwords.words('english')
punctuations = string.punctuation


# In[ ]:


# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 100 == 0:
            print('Processed {} out of {}'.format(counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# ## Adam Geitgey

# In[ ]:


def make_barplot_for_author(Author):
    author_text = [text for text in articles.loc[articles['author'] == Author]['text']]

    author_clean = cleanup_text(author_text)
    author_clean = ' '.join(author_clean).split()
    author_clean = [word for word in author_clean if word not in '\'s']
    author_counts = Counter(author_clean)

    NUM_WORDS = 25
    author_common_words = [word[0] for word in author_counts.most_common(NUM_WORDS)]
    author_common_counts = [word[1] for word in author_counts.most_common(NUM_WORDS)]

    plt.figure(figsize=(15, 12))
    sns.barplot(x=author_common_counts, y=author_common_words)
    plt.title('Words that {} use frequently'.format(Author), fontsize=20)
    plt.show()


# In[ ]:


Author = 'Adam Geitgey'
make_barplot_for_author(Author)


# - Most frequent words are image and network. 
# - Have he frequently written the articles about image for neural network?

# In[ ]:


for title in articles.loc[articles['author'] == 'Adam Geitgey']['title']:
    print(title)


# - Yes, as you can see, he has written many articles of the face recognition and image recognition using deep learning.

# ## Slav Ivanov

# In[ ]:


Author = 'Slav Ivanov'
make_barplot_for_author(Author)


# - Most frequent words are gpu, use and cpu. 
# - Because the 'network' is shown, we can think he wrote some articles about deep learing with GPU.

# In[ ]:


for title in articles.loc[articles['author'] == 'Slav Ivanov']['title']:
    print(title)


# - Good! 

# ## Arthur Juliani

# In[ ]:


Author = 'Arthur Juliani'
make_barplot_for_author(Author)


# - Most frequent words are 'q', 'network' and 'action'. 
# - There are some words which are relevant with Reinforcement learning.
# - Let's see the titles.

# In[ ]:


for title in articles.loc[articles['author'] == 'Arthur Juliani']['title']:
    print(title)


# - SpaCy works well!

# ## Milo Spencer-Harper

# In[ ]:


Author = 'Milo Spencer-Harper'
make_barplot_for_author(Author)


# - Top 3 words are neuron, neural and network. Is he author about deep learning?
# - Let's see the titles!

# In[ ]:


for title in articles.loc[articles['author'] == 'Milo Spencer-Harper']['title']:
    print(title)


# - Most of his articles deals the neural networks.
# - Using SpaCy, we can infer the main subject of articles.

# ## Dhruv Parthasarathy

# In[ ]:


Author = 'Dhruv Parthasarathy'
make_barplot_for_author(Author)


# # Spectial guest!

# - Do you know the 'William Koehrsen'? I know 'William Koehrsen' because of his amazing kernels!
# - He is a kernel master, 8th ranker for now!
# - Let's find the words he likes.

# In[ ]:


Author = 'William Koehrsen'
make_barplot_for_author(Author)


# - Oh, feature is the most frequent used word!

# - As you know that, many his kernel deals the much stuff of features. Below are his kernels.
# - https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
# - https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics
# - https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2

# - Using SpaCy, we can extract his JOB! because he is working in "Feature Labs". :)

# # Conclusion

# - Tokenization using SpaCy works well. 
# - How about using SpaCy?
