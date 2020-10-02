#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import re
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.lang.en import English
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/all.csv")


# In[ ]:


df.head()


# In[ ]:


author_list = df['author'].tolist()


# In[ ]:


author_list[0:5]


# In[ ]:


sentences1 = df['content'].tolist()


# In[ ]:


df['content']


# In[ ]:


sentences2 = list()
for element in sentences1:
    element = element.replace("\n"," ")
    element = element.replace("\r","")
    element = element.replace(",","")
    element = element.replace(".","")
    element = element.replace(";","")
    element = element.replace(":","")
    element = element.replace("?","")
    element = element.replace("!","")
    sentences2.append(element)


# In[ ]:


sent_list = []
for i in range (len(sentences2)):
    str1 = sentences2[i].split(" ")
    sent_list.append(str1)
print(sent_list[0])


# In[ ]:


print(len(sent_list))
print(len(author_list))


# In[ ]:


#Removing the blank spaces
b = list()
for mylist in sent_list:
    temp = list()
    for word in mylist:
        if word is not "":
            temp.append(word)
    b.append(temp)
sent_list = b


# In[ ]:


Nouns = 0
Verbs = 0
Adjectives = 0
noun_count = []
verb_count = []
adjective_count = []

for i in range(len(sent_list)):
    for j in range(len(sent_list[i])):
        print(sent_list[i][j])
        word = nlp(sent_list[i][j])
        for token in word:
            if token.pos_ == 'VERB':
                Verbs= Verbs + 1
            elif token.pos_ == 'NOUN':
                Nouns= Nouns + 1
            elif token.pos_ == 'ADJ':
                Adjectives= Adjectives + 1
    noun_count.append(Nouns)
    verb_count.append(Verbs)
    adjective_count.append(Adjectives)
Nouns = 0
Verbs = 0
Adjective = 0
            


# In[ ]:


print(len(noun_count), len(verb_count), len(adjective_count))


# In[ ]:


df.loc[:, "noun_count"] = noun_count
df.loc[:, "verb_count"] = verb_count
df.loc[:, "adjective_count"] = adjective_count


# In[ ]:


df.head()


# In[ ]:


#splitting the text in to words for analysis purpose
parsed = list()
for i in range(len(sent_list)):
    for j in range(len(sent_list[i])):
        if sent_list[i][j] not in parsed:
            p1 = sent_list[i][j].split(' ')
            parsed.append(p1)


# In[ ]:


len(parsed)


# In[ ]:


print("Total Nouns in the data set",df['noun_count'].sum())
print("Total Verbs in the data set",df['verb_count'].sum())
print("Total Adjectives in the data set", df['adjective_count'].sum())


# In[ ]:


"""
    Number Remover
"""
number_remove = []
for sublist in sent_list:
    mysublist = []
    for word in sublist:
        myword = ""
        for alphaneumeric in list(word):
            if re.search("[0-9]", alphaneumeric):
                pass
            else:
                myword+=alphaneumeric
        mysublist.append(myword)
    number_remove.append(mysublist)
number_remove[0]


# In[ ]:


stopwords = STOP_WORDS


# In[ ]:


s = []

for i in range(len(sent_list)):
    for str in sent_list[i]:
        s_list = [word for word in str.split(" ") if word not in stopwords]
        #print(s_list)
        str_ = ' '.join(s_list)   
    s.append(str_) 
        #print(s)
   


# In[ ]:


#creating a list of unique words
parsed_stop = []
for i in range(len(s)):
    for word in s[i].split(' '):
        if word not in parsed_stop:
            parsed_stop.append(word)
print(parsed_stop)


# In[ ]:


import seaborn as sns
import itertools
import collections
import warnings
warnings.filterwarnings("ignore")

sns.set(font_scale=1.5)
sns.set_style("whitegrid")


# In[ ]:


filtered_sentence =[] 
for i in range(len(sent_list)):
    for j in range(len(sent_list[i])):
        #print(sent_list[i][j])
        word = nlp(sent_list[i][j])
        for token in word:
            token_list = []
            #toke.lower
            token_list.append(token.lower_)
            for word in token_list:
                lexeme = nlp.vocab[word]
                if (lexeme.is_stop == False) and (lexeme.is_punct == False) and (lexeme.is_oov == True):
                     filtered_sentence.append(word)


# In[ ]:


# Create counter
counts_no_stopwords = collections.Counter(filtered_sentence)

counts_no_stopwords.most_common(15)


# In[ ]:


import matplotlib.pyplot as plt

clean_sentences = pd.DataFrame(counts_no_stopwords.most_common(50),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_sentences.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in the Poems (Without Stop Words)")

plt.show()


# In[ ]:


comment_words = ' '
for i in range(len(sent_list)):
    for j in range(len(sent_list[i])):
        comment_words = comment_words + sent_list[i][j] + ' '         


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


wordcloud = WordCloud(width = 800, height = 800, 
background_color ='white', 
stopwords = stopwords, 
min_font_size = 10).generate(comment_words) 

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 


# In[ ]:


cnt_vectorizer = CountVectorizer()
features = cnt_vectorizer.fit_transform(s)
features_nd = features.toarray()


# In[ ]:


df_subset = pd.DataFrame({"age": df['age'], "type": df['type'], "Noun_count":df['noun_count'], 
                                  "Verb_count":df['verb_count'], "Adjective_count":df['adjective_count']})


# In[ ]:


df_subset = pd.get_dummies(df_subset)


# In[ ]:


Final = np.column_stack((features_nd,df_subset))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test  = train_test_split(Final, author_list, train_size=0.75,random_state=1234)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[ ]:


clf1 = MultinomialNB()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.svm import SVC
clf2 =SVC()
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


df['type'].unique()


# In[ ]:


df['author'].unique()


# In[ ]:


df['age'].unique()


# In[ ]:


df_modern = df[df['age'] == 'Modern']


# In[ ]:


df_modern.head(10)


# In[ ]:


df_modern['author'].unique()


# In[ ]:


modern_poetry = df_modern[['author', 'poem name']].drop_duplicates()


# In[ ]:


modern_poetry.head(10)


# In[ ]:


authors = df['author'].unique()


# In[ ]:


authors


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt


# In[ ]:


age = df['age'].unique()


# In[ ]:


genre = df['type'].unique()


# In[ ]:


df['age'].value_counts()


# In[ ]:


df['type'].value_counts()


# In[ ]:


G =nx.Graph()


# In[ ]:


G.add_nodes_from(age)


# In[ ]:


G.add_nodes_from(genre)


# In[ ]:


G.add_nodes_from(authors)


# In[ ]:


G.nodes()


# In[ ]:


myEdges = [tuple(element) for element in df[['author','type']].values]
myEdges2 = [tuple(element) for element in df[['author', 'age']].values]
myEdges3 = [tuple(element) for element in df[['age', 'type']].values]


# In[ ]:


G.add_edges_from(myEdges)
G.add_edges_from(myEdges2)
G.add_edges_from(myEdges3)


# In[ ]:


myEdges


# In[ ]:


nx.draw_spring(G, with_labels = True)
plt.show()


# In[ ]:


nx.density(G)


# In[ ]:


nx.number_of_nodes(G)


# In[ ]:


nx.number_of_edges(G)


# In[ ]:


nx.closeness_centrality(G)


# In[ ]:


nx.betweenness_centrality(G)


# In[ ]:


nx.degree_centrality(G)


# In[ ]:


cross_tab = pd.crosstab(df['age'],df['type'],margins=True)
print(cross_tab)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
cross_tab.iloc[:-1,:-1].plot(kind='bar',stacked=True, color=['red','blue', 'yellow'], grid=False)


# In[ ]:


cross_tab2 = pd.crosstab(df['author'],df['type'],margins=True)
print(cross_tab2)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
cross_tab2.iloc[:-1,:-1].plot(kind='bar',stacked=True, color=['red','turquoise', 'green'], grid=False)


# In[ ]:




