#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Hello World..!!")

#print("Hello \nWorld..!!")


# In[ ]:


name = "kaggle"
year = "2010"

print("Hello my name is " + name + ",")
print("and I am founded in " + year + ".")


# In[ ]:


from math import *

num = 625
print(sqrt(num))


# In[ ]:


# wordcloud using python with the help of Lucid programming

from wordcloud import WordCloud


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import nltk


# In[ ]:


text = "One of the key areas of artificial intelligence is Natural Language Processing (NLP) or text mining as it is generally known that deals with teaching computers how to extract meaning from text."


# In[ ]:


basecloud = WordCloud().generate(text)


# In[ ]:


plt.imshow(basecloud)
plt.axis("off")
plt.show()


# In[ ]:


# The above three lines of code is used very frequently so define it as function

def plot_wordcloud(WordCloud):
    plt.imshow(basecloud)
    plt.axis("off")
    plt.show()


# In[ ]:


from wordcloud import STOPWORDS


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.add("key")
stopwords.add("known")
stopwords.add("generally")


# In[ ]:


wordcloud = WordCloud(stopwords = stopwords, relative_scaling= 1.0).generate(text)


# In[ ]:


plot_wordcloud(wordcloud)

