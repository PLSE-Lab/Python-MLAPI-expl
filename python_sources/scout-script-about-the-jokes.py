#!/usr/bin/env python
# coding: utf-8

# # Scout Script
# 
# We scout out this dataset. Namely performing the following on it:
# 
# -  WordCloud generation
# -  Topic Modeling
# -  Joke length plots
# -  Joke structure estimation
# 
# First off we import some common tools

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
from textblob import TextBlob
from wordcloud import WordCloud
import sklearn
# assert sklearn.__version__ == '0.18' # Make sure we are in the modern age
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/jokes.csv')
df.info()


# In[ ]:


df.head()


# ## WordCloud!
# 
# A word cloud of all the text present.

# In[ ]:


text = ' '.join(df.Question)
cloud = WordCloud(background_color='white', width=1920, height=1080).generate(text)
plt.figure(figsize=(32, 18))
plt.axis('off')
plt.imshow(cloud)
plt.savefig('questions_wordcloud.png')


# In[ ]:


text = ' '.join(df.Answer)
cloud = WordCloud(background_color='white', width=1920, height=1080).generate(text)
plt.figure(figsize=(32, 18))
plt.axis('off')
plt.imshow(cloud)
plt.savefig('answer_wordcloud.png')


# ## Topic Models
# 
# As with any self respecting analysis on unlabeled text data, we perform some topic modeling here with Non Negative Matrix Factorization on the questions.
# 
# This lets us know about the different kinds of question joke categories.

# In[ ]:


# Some defaults
max_features=1000
max_df=0.95,  
min_df=2,
max_features=1000,
stop_words='english'

from nltk.corpus import stopwords
stop = stopwords.words('english')

# document-term matrix A
vectorized = CountVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')

a = vectorized.fit_transform(df.Question)
a.shape


# In[ ]:


from sklearn.decomposition import NMF
model = NMF(init="nndsvd",
            n_components=10,
            max_iter=200)

# Get W and H, the factors
W = model.fit_transform(a)
H = model.components_

print("W:", W.shape)
print("H:", H.shape)


# Get the list of all terms whose indices correspond to the columns of the document-term matrix.

# In[ ]:


vectorizer = vectorized

terms = [""] * len(vectorizer.vocabulary_)
for term in vectorizer.vocabulary_.keys():
    terms[vectorizer.vocabulary_[term]] = term
    
# Have a look that some of the terms
terms[-5:]


# In[ ]:


for topic_index in range(H.shape[0]):  # H.shape[0] is k
    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]
    term_ranking = [terms[i] for i in top_indices]
    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))


# We can see some popular types of question jokes in there. To name a few that I have heard:
# 
# - Cross the road
# - Change a lightbulb
# - What's the difference b/w A and B
# - My favourite joke

# ## Sentiment Analysis
# 
# We assign sentiment scores to the questions and answers.

# In[ ]:


get_polarity = lambda x: TextBlob(x).sentiment.polarity
get_subjectivity = lambda x: TextBlob(x).sentiment.subjectivity

df['q_polarity'] = df.Question.apply(get_polarity)
df['a_polarity'] = df.Answer.apply(get_polarity)
df['q_subjectivity'] = df.Question.apply(get_subjectivity)
df['a_subjectivity'] = df.Answer.apply(get_subjectivity)


# In[ ]:


plt.figure(figsize=(7, 4))
sns.distplot(df.q_polarity , label='Question Polarity')
sns.distplot(df.q_subjectivity , label='Question Subjectivity')
sns.distplot(df.a_polarity , label='Answer Polarity')
sns.distplot(df.a_subjectivity , label='Answer Subjectivity')
sns.plt.legend()


# Perhaps it's a joke is good if the sentiment changes while answering the question? Sadly there's no way to answer that because of the lack of a joke score/ upvote feature in this version of the dataset.
# 
# # About the jokes themselves
# 
# What can we say about the jokes themselves? Let's take a look at length first.

# In[ ]:


daf = df.loc[df.Answer.str.len() < 150]  # There appear to be some outliers in the dataset
sns.distplot(daf.Question.str.len(), label='Question Length')
sns.distplot(daf.Answer.str.len(), label='Answer Length')
sns.plt.legend()


# In[ ]:


# What are the outliers though?
# The threshold has been chosen to keep in the spirit of the dataset
df.loc[df.Answer.str.len() > 400].shape[0]


# We know that the answers are usually shorter than questions. Are there questions whose answers are shorter than them? What about the reverse?

# Similar results hold. A better comparison of question length vs answer length would be a scatter plot. So far we have plotted the difference but what that loses is the exact lengths of the Q and A. 500 - 550 is the same as 10 - 60

# In[ ]:


ql, al = 'Question Length', 'Answer Length'
df[ql] = df.Question.str.len()
df[al] = df.Answer.str.len()
daf = df.loc[df[al] < 250]
sns.jointplot(x=ql, y=al, data=daf, kind='kde', space=0, color='g')


# ## Can we find structure in the jokes?
# 
# [A paper](http://maroo.cs.umass.edu/getpdf.php?id=835) caught my eye. The next iteration of this script will be trying to follow that.
