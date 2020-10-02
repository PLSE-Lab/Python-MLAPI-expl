#!/usr/bin/env python
# coding: utf-8

# We generate simple, frequency-based wordclouds of both the abstracts and the titles, after some text normalization. We will identify bigrams and tigrams, lemmatize, and remove stopwords. We need a handful of tools from nltk:

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import string
from matplotlib import rcParams
from nltk import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk import pos_tag, sent_tokenize, word_tokenize, BigramAssocMeasures,    BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# The following functions do the text normalization

# In[ ]:


def get_bitrigrams(full_text, threshold=30):
    if isinstance(full_text, str):
        text = full_text
    else:
        text = " ".join(full_text)
    bigram_measures = BigramAssocMeasures()
    trigram_measures = TrigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(text.split())
    finder.apply_freq_filter(3)
    bigrams = {" ".join(words): "_".join(words)
               for words in finder.above_score(bigram_measures.likelihood_ratio, threshold)}
    finder = TrigramCollocationFinder.from_words(text.split())
    finder.apply_freq_filter(3)
    trigrams = {" ".join(words): "_".join(words)
                for words in finder.above_score(trigram_measures.likelihood_ratio, threshold)}
    return bigrams, trigrams


def replace_bitrigrams(text, bigrams, trigrams):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
    new_texts = []
    for t in texts:
        t_new = t
        for k, v in trigrams.items():
            t_new = t_new.replace(k, v)
        for k, v in bigrams.items():
            t_new = t_new.replace(" " + k + " ", " " + v + " ")
        new_texts.append(t_new)
    if len(new_texts) == 1:
        return new_texts[0]
    else:
        return new_texts


def process_text(text, lemmatizer, translate_table, stopwords):
    processed_text = ""
    for sentence in sent_tokenize(text):
        tagged_sentence = pos_tag(word_tokenize(sentence.translate(translate_table)))
        for word, tag in tagged_sentence:
            word = word.lower()
            if word not in stopwords:
                if tag[0] != 'V':
                    processed_text += lemmatizer.lemmatize(word) + " "
    return processed_text


def get_all_processed_texts(texts, lemmatizer, translate_table, stopwords):
    processed_texts = []
    for index, doc in enumerate(texts):
        processed_texts.append(process_text(doc, wordnet_lemmatizer, translate_table, stop))
    bigrams, trigrams = get_bitrigrams(processed_texts)
    very_processed_texts = replace_bitrigrams(processed_texts, bigrams, trigrams)
    return " ".join(very_processed_texts)


# Next we read the dataset and initialize what we need for language processing. We restrict the analysis to 2016.

# In[ ]:


records = pd.read_csv("../input/scirate_quant-ph.csv", dtype={"id": str}, index_col=0)
records = records[records["year"] == 2016]
wordnet_lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
translate_table = dict((ord(char), " ") for char in string.punctuation)


# Using all words
# -------------------
# The following image is the wordcloud for the titles. Unfortunately, very few of the bi- and trigrams appear.

# In[ ]:


wordcloud = WordCloud(background_color="white").    generate(get_all_processed_texts(records["title"], wordnet_lemmatizer, translate_table, stop))
plt.figure(figsize=(8, 5))
plt.axis("off")
plt.imshow(wordcloud)


# The following one is the same for the abstracts. Despite excessive stopword removal, we still see a lot of junk.

# In[ ]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").    generate(get_all_processed_texts(records["abstract"], wordnet_lemmatizer, translate_table, stop))
plt.figure(figsize=(8, 5))
plt.axis("off")
plt.imshow(wordcloud)


# Using bi- and trigrams alone
# -----------------------------------
# We get something more interesting if we restrict the analysis to the bi- and trigrams. There is a reassuring overlap between the two, although curiously authors are more keen to point out tensor networks in the title than in the abstracts.

# In[ ]:


def use_ngrams_only(texts, lemmatizer, translate_table, stopwords):
    processed_texts = []
    for index, doc in enumerate(texts):
        processed_texts.append(process_text(doc, wordnet_lemmatizer, translate_table, stop))
    bigrams, trigrams = get_bitrigrams(processed_texts)
    indexed_texts = []
    for doc in processed_texts:
        current_doc = []
        for k, v in trigrams.items():
            c = doc.count(k)
            if c > 0:
                current_doc += [v] * c
                doc = doc.replace(k, v)
        for k, v in bigrams.items():
            current_doc += [v] * doc.count(" " + k + " ")
        indexed_texts.append(" ".join(current_doc))
    return " ".join(indexed_texts)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").    generate(use_ngrams_only(records["title"], wordnet_lemmatizer, translate_table, stop))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").    generate(use_ngrams_only(records["abstract"], wordnet_lemmatizer, translate_table, stop))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# I love how "et al" becomes a leading topic in the abstracts.

# Alternative representation of author histogram
# -----------------------------------------------------------
# We can plot a word cloud for the authors as well, instead of a [histogram](https://www.kaggle.com/peterwittek/d/peterwittek/scirate-quant-ph/top-authors-in-quant-ph-in-2016).

# In[ ]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").    generate(";".join(records["authors"]).replace(" ", "_"))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

