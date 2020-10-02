#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
#test_df = pd.read_csv("ProjectData/test.csv")


# In[ ]:


train_df.head(5)


# In[ ]:


train_text = train_df['question_text']


# In[ ]:


type(train_text)


# In[ ]:


train_label = train_df['target']


# In[ ]:


train_label.value_counts()


# **WordCloud for Insincere Questions**

# In[ ]:


from wordcloud import WordCloud


# In[ ]:


insincere_text = train_df[train_df.target ==1 ]['question_text']


# In[ ]:


insincere_text = " ".join(text for text in insincere_text)


# In[ ]:


insincere_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(insincere_text)


# In[ ]:


# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(insincere_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# **Examples of Insincere questions**

# In[ ]:


insincere_questions_corpus = train_df[train_df.target ==1 ]['question_text'].values.tolist()


# In[ ]:


insincere_questions_corpus[:11]


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer


# In[ ]:


def clean_text(insincere_words):
    #print(insincere_words)
    insincere_words_tokenized = word_tokenize(insincere_words)
    #insincere_words_tokenized = [i for i in insincere_words_tokenized if nltk.pos_tag ]
    insincere_words_cleaned = [i.lower() for i in insincere_words_tokenized]
    insincere_words_cleaned = [WordNetLemmatizer().lemmatize(i) for i in insincere_words_cleaned]
    insincere_words_cleaned = [i for i in insincere_words_cleaned if i not in string.punctuation]
    insincere_words_cleaned = [i for i in insincere_words_cleaned if i not in stopwords.words('english')]
    return(insincere_words_cleaned)


# In[ ]:


insincere_questions_cleaned =[clean_text(doc) for doc in insincere_questions_corpus]


# In[ ]:


insincere_questions_cleaned[:2]


# In[ ]:


insincere_questions_text = " ".join(str(i) for i in insincere_questions_cleaned)


# **Another iteration of wordcloud after some text preprocessing**

# In[ ]:


insincere_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(insincere_questions_text)


# In[ ]:


# Display the generated image:
plt.figure(figsize=(15,10))
plt.imshow(insincere_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# **Topic Modeling using LDA**

# In[ ]:


import gensim


# In[ ]:


from gensim.models import LdaModel


# In[ ]:


from gensim.corpora import Dictionary


# In[ ]:


dictionary = Dictionary(insincere_questions_cleaned)


# In[ ]:


len(dictionary)


# In[ ]:


insincere_bow = [dictionary.doc2bow(doc) for doc in insincere_questions_cleaned]


# In[ ]:


from gensim.models.ldamodel import LdaModel


# In[ ]:


lda_model = LdaModel(corpus = insincere_bow, num_topics=5, id2word=dictionary, passes = 10,random_state = 1)


# In[ ]:


lda_model.show_topics(num_topics= 5)


# Following topics seem to generate most insincere questions:
# * America and American politics
# * Religion 
# * Gender and Race

# In[ ]:


import pyLDAvis.gensim


# In[ ]:


lda_visualization = pyLDAvis.gensim.prepare(lda_model,insincere_bow,dictionary,sort_topics = False)


# In[ ]:


pyLDAvis.display(lda_visualization)

