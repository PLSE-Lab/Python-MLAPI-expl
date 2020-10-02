#!/usr/bin/env python
# coding: utf-8

# ## <font color=#10A2DF> <p style="text-align: center;">&#9733; Natural Language Processing  &#9733;</p></font> 

# ##### $\color{#10A2DF}{\text{Instal NLTK toolkit for text processing.}}$

# In[ ]:


pip install nltk


# ##### $\color{#10A2DF}{\text{Import NLTK and necessary modules and components.}}$

# In[ ]:


nltk.download("punkt")
import string
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TreebankWordTokenizer, RegexpTokenizer, WhitespaceTokenizer


# ##### $\color{#10A2DF}{\text{Create a body of text}}$

# In[ ]:


text = "I have been working at amazon full-time for more than two years. Yet, I don't have a close relationship with my manager Dr. Smith. " +        "No work/life balance. " +        "so realistically you only will have 8 days off a year when you combine the two together. " +        "I live on the eastside of New York (18mi from office), and my manager isnot flexible on my start time. " +        "salaries in Minimum wage, Benefits capped to $100,000 per calendar year! " +        "there is no Principle around treating employees well. Managers' attitude is negative and treat employees with no respect. "       
text


# In[ ]:


len(text)


# ##### $\color{#10A2DF}{\text{Trained: Convert the text to sentence tokens}}$

# In[ ]:


Sentence_tkns = nltk.sent_tokenize(text)
Sentence_tkns


# ##### $\color{#10A2DF}{\text{Untrained: Create a Punkt sentence tokenizer.}}$

# In[ ]:


pnkt_sntnce_tknzr = nltk.PunktSentenceTokenizer()
pnkt_sntnce_tknzr.tokenize(text)


# In[ ]:


len(Sentence_tkns)


# In[ ]:


print(Sentence_tkns[0])
print(Sentence_tkns[7])


# ##### $\color{#10A2DF}{\text{Tokenize the text by words.}}$

# In[ ]:


print(word_tokenize(text))


# ##### $\color{#10A2DF}{\text{Word_tokenize a specific sentence with index of tokens.}}$

# In[ ]:


print(nltk.word_tokenize(Sentence_tkns[4]))


# ##### $\color{#10A2DF}{\text{Tokenize all sentences in Sentence_tkns.}}$

# In[ ]:


Word_tkns = [nltk.word_tokenize(Sentence_tkns) for Sentence_tkns in Sentence_tkns]

for element in Word_tkns:
    print(element)


# ##### $\color{#10A2DF}{\text{Use Treebank to split standard contractions and split off commas.}}$

# In[ ]:


TBW_tkns = nltk.TreebankWordTokenizer()
print(TBW_tkns.tokenize(Sentence_tkns[4]))


# ##### $\color{#10A2DF}{\text{Use Regexp Tokenizer to segment text by word characters and remove punctuations.}}$

# In[ ]:


Ptrn_words = r'\w+' 
RGX_tkns = nltk.RegexpTokenizer(pattern=Ptrn_words, gaps=False)
print(RGX_tkns.tokenize(Sentence_tkns[4]))


# ##### $\color{#10A2DF}{\text{Use Regexp Tokenizer to segment text by whitespace characters.}}$

# In[ ]:


Ptrn_whiteSp = r'\s+'
RGX_tkns = nltk.RegexpTokenizer(pattern=Ptrn_whiteSp, gaps=True)
print(RGX_tkns.tokenize(Sentence_tkns[4]))


# ##### $\color{#10A2DF}{\text{Use Whitespace Tokenizer to explicitly segment text by whitespace characters (space, tab, newline).}}$

# In[ ]:


WST = nltk.WhitespaceTokenizer()
print(WST.tokenize(Sentence_tkns[5]))


# ##### $\color{#10A2DF}{\text{Convert upercase words to lower case.}}$

# In[ ]:


def lower_tokens(tokens):
    return [token.lower() for token in tokens]

print(lower_tokens(nltk.word_tokenize(Sentence_tkns[5])))


# ##### $\color{#10A2DF}{\text{Use string punctuation to remove all punctuation from t.}}$

# In[ ]:


string.punctuation


# ##### $\color{#10A2DF}{\text{Create a function that removes punctuation.}}$

# In[ ]:


# let's create a function to remove all punctuation in each token in a list

def remove_punct(tokens):
    punct_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return [a for a,b in zip(tokens, [punct_regex.sub('', token) for token in tokens]) if b != '']


# ##### $\color{#10A2DF}{\text{Use the function for a text.}}$

# In[ ]:


print(remove_punct(lower_tokens(nltk.word_tokenize(Sentence_tkns[6]))))

