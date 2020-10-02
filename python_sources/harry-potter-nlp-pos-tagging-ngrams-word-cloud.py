#!/usr/bin/env python
# coding: utf-8

# ## Harry Potter and the Philosopher's Stone NLP analysis

# In[ ]:


import PIL.Image
from IPython.display import Image
Image('/kaggle/input/private-dataset/img.jpg',width=500, height=300)


# In[ ]:


import numpy as np
import pandas as pd
import string
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


file = open('/kaggle/input/private-dataset/hp1.txt')
text = file.read()
re.findall('Page\s.\s\d*\s.*Rowling',text)[:2]


# ## Text cleaning and removing stop words

# In[ ]:


# removing page number pattern (Page | 2 Harry Potter and the Philosophers Stone - J.K. Rowling)
text = re.sub('Page\s.\s\d*\s.*Rowling','',text)
text = re.sub('\n','',text)


# In[ ]:


# using regex tokenizer
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

text = "".join([word for word in text if word not in string.punctuation])
tokenizer=RegexpTokenizer("['\w]+")
tokens = tokenizer.tokenize(text)
words = [word.lower() for word in tokens if word.lower() not in stop_words]


# In[ ]:


#Printing vocabulary size
vocabulary = set(words)
print('The vocabulary size is: ',len(vocabulary))


# In[ ]:


# The number of words that has been deleted after removing stop words + percentage
print('The number of words that have been removed is {} which is {:.2f}% of total words'.format
      (len(tokens)-len(words),len(words)/len(tokens)*100))


# ## WordCloud of Harry Potter and the Philosopher's Stone

# In[ ]:


from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud,ImageColorGenerator

unigrams = list(ngrams(words, 1))
freq = Counter(unigrams)
topN= freq.most_common(100)

wordscount = {w[0]:f for w, f in topN}   
wordcloud = WordCloud(max_font_size=40,background_color="white")
wordcloud.fit_words(wordscount)
plt.figure(figsize=(7,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()


# ## Generate a word cloud based on an image

# In[ ]:


Image('/kaggle/input/private-dataset/image.jpg',width=400, height=300)


# In[ ]:


# Generate a word cloud image
mask = np.array(PIL.Image.open("/kaggle/input/private-dataset/image.jpg"))
wordcloud = WordCloud(background_color="white", max_words=400, mask=mask).generate(text)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[10,10])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:


count = Counter(words)
df = pd.DataFrame(count.most_common(10),columns=['Words','Frequency'])


# In[ ]:


import plotly.express as px
fig = px.bar(x=df['Words'],y=df['Frequency'])
fig.update_layout( title={
        'text': "Top 10 most frequent word",
        'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    xaxis_title="Words",
    yaxis_title="Frequency")

fig.show(renderer='kaggle')


# ## POS tagging

# In[ ]:


nltk.pos_tag(tokens[:10])


# In[ ]:


tagged_words= nltk.pos_tag(tokens)
#print ("Tagged Words: ", tagged_words)


# ## Design a chunker using chunk grammar

# In[ ]:


chunkGram='''CHUNK1: {<DT><NNS>}
            CHUNK2: {<CD><JJ><NN>}
            CHUNK3: {<CD><JJ><JJ><NN>}'''
#Creat chunk parser
find = nltk.RegexpParser(chunkGram)
#Test chunk parser on our example
chunkTree=find.parse(tagged_words)


print("Extracting three different chunks")
for subtree in chunkTree.subtrees():
    if subtree.label() == 'CHUNK1':
        finalChunk=""
        for (w,tag) in subtree.leaves():
            finalChunk=finalChunk + " " + w
        
    elif subtree.label() == 'CHUNK2':
        finalChunk2=""
        for (w,tag) in subtree.leaves():
            finalChunk2=finalChunk2 + " " + w
    elif subtree.label() == 'CHUNK3':
        finalChunk3=""
        for (w,tag) in subtree.leaves():
            finalChunk3=finalChunk3 + " " + w
print (finalChunk+"\n"+finalChunk2+"\n"+finalChunk3)


# ## Generating bi-grams from tokens

# In[ ]:


bigrams = ngrams(tokens,2)

# using Counter function to count the most common bigrams from words (vocabulary after removing stop words)
freq_bigrams=Counter(bigrams)
print ("Top 5 most common bigrams from tokens:\n", freq_bigrams.most_common(5))


# ## Generating tri-grams

# In[ ]:


from nltk.metrics import TrigramAssocMeasures

trigrams= nltk.TrigramCollocationFinder.from_words(tokens)
print ("Top tri-grams")
trigrams.nbest(TrigramAssocMeasures.raw_freq, 5)


# ## Finding the keywords

# In[ ]:


from gensim.summarization import keywords
print ('Keywords:')
#Get 0.01 key words
keyWords=keywords(text, ratio=0.01, lemmatize=True)
print (keyWords)

