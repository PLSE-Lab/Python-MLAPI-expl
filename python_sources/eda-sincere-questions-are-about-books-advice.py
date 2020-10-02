#!/usr/bin/env python
# coding: utf-8

# ...while insincere questions tend to be politicially, racially, religiously, etc. charged. 
# 
# Other notebooks (like [this one](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc)) looked at term frequencies of sincere and insincere questions and found that terms like 'will', 'many people', 'united states' appear frequently in both classes, so these terms are most likely not very predictive. 
# 
# The goal of this notebook is to go one step futher and look at class-specific term frequencies. The class-specific frequency of a term is 
# 
# $tf_{c1} = \frac{n_{c1}}{n}$,
# 
# where $n$ is the number of times the term appears in the corpus, and $n_{c1}$ is the number of times the term appears in class 1 documents (insincere questions in our case). The notebook cited above looks at $n_{c0}$ and $n_{c1}$, while we normalize $n_{c1}$ with $n$. 
# 
# It is useful to place the unique terms of the corpus on the $n$ - $tf_{c1}$ plane for two reasons:
# 
# - it helps feature selection: predictive terms are common in the corpus (high $n$) and they predominantly appear in insincere questions (high $tf_{c1}$), so these terms are in the upper right corner of the $n$ - $tf_{c1}$ plane,
# - we can get a feeling of what the sincere and insincere questions tend to be about.
# 
# We see that sincere questions are about books, computer science, and asking for advice. The following lemmatized terms have $n > 1000$ and $tf_{c1} < 0.001$.
# ```python
# ['be mean by' 'be the scope' 'book for' 'computer science' 'do i prepare'
#  'ece' 'fresher' 'in computer' 'java' 'major accomplishment'
#  'mechanical engineering' 'the best book' 'the scope' 'us of'
#  'what inspire']
# ```
# 
# And the terms below have $n > 100$ and $tf_{c1} > 0.7$.
# ``` python
# ['all muslim' 'american so' 'asian woman' 'be american so'
#  'be black people' 'be castrate' 'be democrat' 'be feminist' 'be hindus'
#  'be indians' 'be liberal' 'be liberal so' 'be muslims' 'be white people'
#  'bhakts' 'black american' 'black man' 'black men' 'black people'
#  'black woman' 'bullshit' 'castrate' 'castration' 'democrat be'
#  'democrats' 'do black people' 'do democrat' 'do feminist' 'do liberal'
#  'do white people' 'fuck' 'hindu be' 'indian on' 'indian so' 'liberal so'
#  'liberal think' 'liberals' 'men so' 'moron' 'muslim be' 'muslims'
#  'quora moderator' 'sex with my' 'shithole' 'so stupid' 'than white'
#  'that muslim' 'the fuck' 'to fuck' 'to rape' 'white american'
#  'white girl' 'white men' 'white people' 'white woman' 'why be black'
#  'why be democrat' 'why be indians' 'why be liberal' 'why be muslim'
#  'why be white' 'why do black' 'why do democrat' 'why do feminist'
#  'why do liberal' 'why do muslim' 'why do white' 'why muslim' 'woman so']
# ```
# 
# Let's walk through the notebook. 
# 

# In[ ]:


# load packages
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# In[ ]:


# tokenizer for stemming and lemmatization
def tokenize(text):
    def convert_tag(tag):
        part = {'ADJ' : 'a',
                'ADV' : 'r',
                'VERB' : 'v',
                'NOUN' : 'n'}
        if tag in part.keys():
            return part[tag]
        else:
            # other parts of speech will be tagged as nouns
            return 'n'

    tokens = nltk.word_tokenize(text)
    tokens_pos = pos_tag(tokens,tagset='universal')
    stems = []
    for item in tokens_pos:
        term = item[0]
        pos = item[1]
        # stem/lemmatize tokens consisting of alphabetic characters only
        if term.isalpha():
            stems.append(WordNetLemmatizer().lemmatize(term, pos=convert_tag(pos)))
            #stems.append(PorterStemmer().stem(item))
    return stems


# In[ ]:


# vectorize corpus
train_df = pd.read_csv("../input/train.csv")

print("Train shape : ",train_df.shape)
print(train_df.columns)

# target variable and some basic stats
y_all = train_df['target']
print('fraction of datapoints in class 1: ',1e0*np.sum(y_all == 1)/len(y_all)) # fraction of datapoints in class 1
print('number of datapoints in class 1: ',np.sum(y_all == 1)) # number of datapoints in class 1

# n-grams with n = 1 - 3, no stopwords, use words that appear in at least min_df documents
vectorizer = TfidfVectorizer(ngram_range=(1,3),tokenizer=tokenize,min_df=100,                             sublinear_tf=True) 
X_all = vectorizer.fit_transform(train_df['question_text'])#.sample(100,random_state=seed))
print(np.shape(X_all))


# In[ ]:


# a quick look at the unique terms in the corpus
terms = np.array(vectorizer.get_feature_names())
print(terms[:100]) # the first 100 terms
print(terms[-100:]) # the last 100 terms
print(np.random.choice(terms,size=100,replace=False)) # 100 randomly selected terms


# In[ ]:


# n -- number of times the terms appear in docs 
term_count = X_all.getnnz(axis=0)

indices = np.where(y_all == 1)[0]
# tf_c1 -- the fraction of times the terms appear in class 1
frac_in_class1 = 1e0*X_all.tocsc()[indices].getnnz(axis=0)/term_count

indcs = np.where((frac_in_class1 <= 0.001) & (term_count >= 1000))[0]
print(terms[indcs]) # terms in sincere questions

indcs = np.where((frac_in_class1 >= 0.7) & (term_count >= 100))[0]
print(terms[indcs]) # terms in insincere questions


# In[ ]:


# plotting: there aren't so many terms in the upper right quadrant of the plot

plt.scatter(term_count,frac_in_class1)
plt.semilogx()
plt.xlabel('n - # times term in corpus')
plt.ylabel('tf_c1 - class-specific frequency')
plt.title('scatter plot')
plt.show()

xbins = 10**np.linspace(2,5,31)
ybins = np.linspace(0,1,41)
counts, _, _ = np.histogram2d(term_count,frac_in_class1,bins=(xbins,ybins))
counts[counts == 0] = 0.5 # so log10(0) is not nan
plt.pcolormesh(xbins, ybins, np.log10(counts.T))
plt.semilogx()
plt.xlabel('n - # times term in corpus')
plt.ylabel('tf_c1 - class-specific frequency')
cbar = plt.colorbar(label='count',ticks=[0,1,2,3])
cbar.ax.set_yticklabels([1,10,100,1000])
plt.title('heatmap')
plt.show()


# Any comments or suggestions are greatly appreaciated especially on how I can improve the preprocessing and lemmatization.
