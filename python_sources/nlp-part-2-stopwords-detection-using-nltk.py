#!/usr/bin/env python
# coding: utf-8

# # Stopwords detecion using NLTK

# Import required libraries

# stopwords from nltk.corpus provides us with a list of stopwords

# In[ ]:


from nltk.corpus import stopwords


# 
# tokenize sentence into words so that we can match and remove stopwords

# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


sentence = '''Barack Hussein Obama II is an American attorney and politician who served as the 
        44th president of the United States from 2009 to 2017. A member of the Democratic Party, 
        he was the first African American to be elected to the presidency.'''


# In[ ]:


stop_words = set(stopwords.words('english')) 
word_tokens = word_tokenize(sentence) 
  
sentence_without_stopwords = [w for w in word_tokens if not w in stop_words] 
  
sentence_without_stopwords = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        sentence_without_stopwords.append(w) 
  
print("Word Tokens: " + str(word_tokens)+ '\n') 
print("Sentence without stopwords: "+ str(sentence_without_stopwords))


# # End of Notebook
