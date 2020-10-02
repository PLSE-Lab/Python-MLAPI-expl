#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


skills=pd.read_csv('../input/job_skills.csv')


# In[ ]:


skills.columns


# In[ ]:


skills.shape


# In[ ]:


skills.dtypes


# In[ ]:


skills.head()


# In[ ]:


skills.Company.value_counts()


# So we have two values for company Google and Youtube

# Now let us look at the Title column

# In[ ]:


skills.Title.value_counts().head(20)


# As you can see the most number is for Business Intern 2018

# In[ ]:


skills.Title.value_counts().head(20).plot.bar()


# In[ ]:


skills.Category.value_counts()


# In[ ]:


skills.Category.value_counts().plot.bar()


# As you can see the most number of jobs are in Sales & Account Management.Suprisingly the number of jobs in software engineering is relatively low.Although there are some technical jos in other categories

# In[ ]:


skills.Location.value_counts().head(10)


# Let us split location to place,state and country

# In[ ]:


skills['place']=skills.Location.str.split(',').apply(lambda x:x[0] if(len(x)==3) else 'NaN')
skills['state']=skills.Location.str.split(',').apply(lambda x:x[1] if(len(x)==3) else (x[0] if (len(x)==2) else 'NaN'))
skills['country']=skills.Location.str.split(',').apply(lambda x:x[2] if(len(x)==3) else (x[1] if (len(x)==2) else (x[0] if(len(x)==1) else 'NaN')))


# In[ ]:


skills.head()


# In[ ]:


skills.place.value_counts().head(20)


# In[ ]:


skills.state.value_counts().head(20)


# In[ ]:


skills.state.value_counts().head(20).plot.bar()


# In[ ]:


skills.state.value_counts().head(20).plot.bar()


# In[ ]:


skills.country.value_counts().head(20).plot.bar()


# In[ ]:


skills['Minimum Qualifications'].value_counts().head()


# In[ ]:


import spacy
nlp = spacy.load("en")


# In[ ]:


my_stop_words = [u'say', u'\'s', u'Mr', u'be', u'said', u'says', u'saying']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


texts=skills['Minimum Qualifications'].values
text=' '.join(str(e) for e in texts)

doc=nlp(text)
#doc
    


# In[ ]:


# we add some words to the stop word list
texts=[]
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if  w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        texts.append(w.lemma_)
   
    


# In[ ]:


#texts


# In[ ]:


worddict={}
for word in texts:
    if (word not in worddict.keys()):
        worddict[word]=1
    else:
        worddict[word]=worddict[word]+1
        


# In[ ]:


count=0
topwords={}
for word in sorted(worddict,key=worddict.get,reverse=True):
    count=count+1
    if(count<40):
        topwords[word]=worddict[word]
    else:
        break
  


# In[ ]:


topwords


# In[ ]:


x=topwords.keys()
y=topwords.values()


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.bar(x,y)


# The above graph shows the top keywords in minimum qualification

# In[ ]:


from wordcloud import WordCloud

# Create the wordcloud object
wordcloud = WordCloud(width=1000, height=1000, margin=0).generate_from_frequencies(worddict)

# Display the generated image:
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[ ]:


texts=skills['Responsibilities'].values
text=' '.join(str(e) for e in texts)

doc=nlp(text)

# we add some words to the stop word list
texts=[]
for w in doc:
    # if it's not a stop word or punctuation mark, add it to our article!
    if  w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:
        # we add the lematized version of the word
        texts.append(w.lemma_)


# In[ ]:


worddict={}
for word in texts:
    if (word not in worddict.keys()):
        worddict[word]=1
    else:
        worddict[word]=worddict[word]+1


# Create the wordcloud object
wordcloud = WordCloud(width=1000, height=1000, margin=0).generate_from_frequencies(worddict)

# Display the generated image:
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# Looking at the world cloud the most frequent responsiblities are team,product,business

# In[ ]:




