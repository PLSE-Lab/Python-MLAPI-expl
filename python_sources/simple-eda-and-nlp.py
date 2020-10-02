#!/usr/bin/env python
# coding: utf-8

# **Notebook Objective:**
# The objective of this notebook is to explore how the data looks without any processing.
# 
# **Competition Objective :**
# The main objective of this competition is to predict whether the asked question on Quora is sincere or not. 

# **Import all required libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from nltk.corpus import stopwords 
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, FreqDist
from textblob import TextBlob
import collections
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)


# **Load data**

# In[ ]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
submission_data=pd.read_csv("../input/sample_submission.csv")


# **Inspect the data**
# 
# The training data set has a total of 1306122 records with three columns. testing data has a total of 56370 records with 2 columns. Submission data has a total of 56370 with 2 columns

# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# In[ ]:


submission_data.shape


# **Focus on Training data**

# In[ ]:


train_data.head()


# In[ ]:


train_data.columns


# **Check for null values**

# In[ ]:


train_data['question_text'].isnull().value_counts()


# In[ ]:


train_data['target'].isnull().value_counts()


# In[ ]:


train_data['question_text'].drop_duplicates().count()


# In[ ]:


# Unique target
train_data['target'].unique()


# In[ ]:


insicere_quiz=train_data[train_data['target']==1]
sincere_quiz=train_data[train_data['target']==0]


# **Percentage of insincere questions  in the dataset**

# In[ ]:


insicere_quiz['target'].count()/train_data['target'].count() * 100


# **Percentage of sincere questions in the dataset**

# In[ ]:


sincere_quiz['target'].count()/train_data['target'].count() * 100


# In[ ]:


target_counts = train_data['target'].value_counts()
target_counts


# **Simple visualization for target distribution**

# In[ ]:


pie_labels = (np.array(target_counts.index))
pie_sizes = (np.array((target_counts / target_counts.sum())*100))

trace = go.Pie(labels=pie_labels, values=pie_sizes)
pie_layout = go.Layout(title='Target distribution',font=dict(size=16),width=500,height=500)
fig = go.Figure(data=[trace], layout=pie_layout)
py.iplot(fig, filename="file_name")


# In[ ]:


bar_graph = go.Bar(
        x=target_counts.index,
        y=target_counts.values,
        marker=dict(
        color=target_counts.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

bar_layout = go.Layout(title='Target Distrinution',font=dict(size=20))
fig = go.Figure(data=[bar_graph], layout=bar_layout)
py.iplot(fig, filename="file_name")


# **Simple Natural Language Processing (NLP) Tasks**

# **1. Checking for stop words**

# In[ ]:


stop_words = stopwords.words('english')

tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)

filtered_text = [token for token in tokens if not token in stop_words]  

filtered_text = [] 
  
for i in tokens: 
    if i in stop_words: 
        filtered_text.append(i) 
        
print(np.array(filtered_text))
len(np.array(filtered_text))


# **2. Part Of Speech Tagging (POS)**

# In[ ]:


token=[]
for i in train_data[0:2]['question_text']:
    token.append(pos_tag(word_tokenize(i)))

print (token)


# **3. Extracting Noun Phrases**

# In[ ]:


for i in train_data[0:10]['question_text']:
    print(TextBlob(i).noun_phrases)


# **4. Detecting language**

# In[ ]:


# lng=[]
# for i in train_data[0:5]['question_text']:
#     lng.append(TextBlob(i).detect_language())
    
# set(lng)


# **5. Word frequency**

# In[ ]:


for i in train_data[0:5]['question_text']:
    print(TextBlob(i).word_counts["quebec"])


# **6. Words count**

# In[ ]:


for i in train_data[0:10]['question_text']:
    print(i," => ",len(word_tokenize(i)))


# 7. Sentence count

# In[ ]:


for i in train_data[0:10]['question_text']:
    print(i," => ",len(sent_tokenize(i)))


# 8. Most Common Words

# In[ ]:


tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)


frequency_distribution=FreqDist(tokens).most_common()
print(frequency_distribution)


# In[ ]:


tokens=[]
for i in train_data[0:10]['question_text']:
    for j in word_tokenize(i):
        tokens.append(j)


frequency_distribution=FreqDist(tokens)
word_c={}
for i in frequency_distribution:
    word_c[i]=token
    word_c[i]=frequency_distribution[i]

sorted(word_c.items(), key=lambda x: x[1], reverse=True)


# **9. WordCloud**

# In[ ]:


text=str(train_data[0:1000]['question_text'])
wordcloud = WordCloud(width=1600, height=800).generate(text)
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.figure( figsize=(40,30) )
plt.show()


# **9. Sentiment Analysis**

# In[ ]:


# Calculating Sentment Analysis with TextBlob
for i in train_data[0:5]['question_text']:
    print(i," => ",TextBlob(i).sentiment)


# In[ ]:


# Extracting the sentiment polarity of a text
for i in train_data[0:5]['question_text']:
    print(TextBlob(i).sentiment.polarity)


# In[ ]:


# Extracting the sentiment subjectivity of a text
for i in train_data[0:5]['question_text']:
    print(TextBlob(i).sentiment.subjectivity)


# **10. N-grams (tri-gram)**

# In[ ]:


for i in train_data[0:2]['question_text']:
    print(TextBlob(i).ngrams(n=3))


# **11. Term Frequency Inverse Document Frequency (tf-idf)**

# In[ ]:


corpus=[]
for i in train_data[0:5]['question_text']:
    corpus.append(i)

cvect = CountVectorizer(ngram_range=(1,1))
counts = cvect.fit_transform(corpus)
normalized_counts = normalize(counts, norm='l1', axis=1)

tfidf = TfidfVectorizer(ngram_range=(1,1), smooth_idf=False)
tfs = tfidf.fit_transform(corpus)
new_tfs = normalized_counts.multiply(tfidf.idf_)

feature_names = tfidf.get_feature_names()
corpus_index = [n for n in corpus]
df = pd.DataFrame(new_tfs.T.todense(), index=feature_names, columns=corpus_index)

print(df)


# **12. Bag of Words (BoW)**

# In[ ]:


#Bow with collection
token=[]
for i in train_data[0:5]['question_text']:
    token.append(i)

bow = [collections.Counter(words.split(" ")) for words in token]
total_bow=sum(bow,collections.Counter())
print(total_bow)


# **To Continue .....**
