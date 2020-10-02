#!/usr/bin/env python
# coding: utf-8

# ## Task 1: 
# 
# 
# Lets start with importing the necessary modules, so we will import the three main modules.

# In[ ]:


#the necessary imports
from bs4 import BeautifulSoup
import requests
import nltk


# ## Task 2:
# 
# We need to find a book that we like in HTML format. Gutenberg.org is a website with a bunch of free books, so this is a good start.
# 
# Personally I am fond of Kafka's "Metamorphosis", so I pick this one. The link to my book would be as follows 'http://www.gutenberg.org/files/5200/5200-h/5200-h.htm'. Once picked lets get it into Python by a request.

# In[ ]:


#this gets the book from a certain website
r = requests.get('http://www.gutenberg.org/files/5200/5200-h/5200-h.htm')

#whenever you have a book you need to set the encoding correctly, most of the cases this is 'utf-8'
r.encoding = 'utf-8'

# Now lets extract the text from the html which is placed in our variable r
html = r.text

# Lets do a sanity and check the first 1000 words
print(html[:1000])


# ## Task 3:
#     
# Well that all seems to work. There are a few things that we should take into consideration, for example you can ssee the HTML code at the top. We don't want to keep this so we need to look for a solution, and BeautifulSoup comes in handy on that matter. BeautifulSoup creates from HTML soup humanly readable soup.

# In[ ]:


# Lets first create the soup from our HTML file
soup = BeautifulSoup(html)

# Then we're getting the text out of it
text = soup.get_text()

# Lets print a random area to make sure that everything is working fine
print(text[10000:11000])


# ## Task 4:
# 
# Well that seems about right although there is still a lot of text at the beginning and at the end. Since it's not that much we just leave it as it is. Now it's time to use the nltk- natural language tool kit. First we will remove whitespaces and dots and commas. Else this could be seen by Python as a word. This is called tokenizing, bellow is the way I have done it.

# In[ ]:


# First we create the tokenizer (\w+) means all non-word characters
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')

# Then we will fill in the text
tokens = tokenizer.tokenize(text)

# Lets do a sanity check again
print(tokens[:4])


# ## Task 5:
# 
# We're making good progress. The next step is to make everything lowercase so that won't cause any difficulties when python is going through the text. I did this as follows.

# In[ ]:


# Lets make a new list with lowercase words
words = []

# Looping through the words and appending them in the new list.
for word in tokens:
    words.append(word.lower())

# Sanity, sanity, sanity
print(words[:4])


# ## Task 6:
#     
# We don't want to have any stop words since they will probably be the first ones, we are more interested in the real content. So lets remove the stopwords, luckily there is a package for that in nltk.

# In[ ]:


#For the stopwords we have to install nltk.corpus, else you will get an error. Lets place the stopwords in sw
from nltk.corpus import stopwords
sw = stopwords.words('english')

# We create a new list without any stopword, called words_ns
words_ns = []

# Appending to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)

# Lets make sure that the stopwords are gone, as you can see 'by' is now gone
print(words_ns[:4])


# ## Task 7:
# 
# As last we need to figure out how to the first words. We are going to use matplotlib inline, this might differ when you're not using Jupyter Notebooks. 

# In[ ]:


# Displays figures inline in Jupyter Notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Create a frequency distribution
freqdist = nltk.FreqDist(words_ns)

# Lets plot and see if it has paid off!
freqdist.plot


# The word found most is 'Gregor'

