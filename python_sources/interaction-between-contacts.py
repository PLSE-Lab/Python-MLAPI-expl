#!/usr/bin/env python
# coding: utf-8

# # Analysing the interaction between Hillary Clinton's contacts
# 
# Working on Hillary Clinton's emails is a good occasion to analyse if there is an interaction between her contacts :
# 
# To do so, an analysis will be lead over the emails she sent, and received. The fact that she is the sender or the receiver is not so important. What matters is who she interacts with. We will gather all the emails and group them by contact. The following analysis will be text mining, mainly divided into 4 steps :
# - Compute the term frequency of each term in each person's emails (using TFIDF)
# - Compute the cosinus similarity to point out the distance between each person's emails, and build groups
# - Prune if necessary
# - Visualize this new information through a graph
# 
# ## Fetching the data  
# To get the required data, sqlite3 will be used. Required fields for this analysis are as follow:
# - the sender
# - the receiver
# - the email content

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np
con = sqlite3.connect('../input/database.sqlite')

emails = pd.read_sql_query("""
SELECT p.Name Sender,
       e.SenderPersonId Id_sender, e.MetadataTo, e.ExtractedBodyText text,
       a.PersonId
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id 
LEFT OUTER JOIN Aliases a ON lower(e.MetadataTo)=a.Alias
""", con)

persons = pd.read_sql_query("""
SELECT Id, Name
FROM Persons 
""", con)
personsDict = {}
for i in persons.values:
    personsDict[i[0]] = i[1]


# ## Cleaning the data and setting things ready
# 
# What is interesting through this analysis is to focus on who Hillary Clinton is communicating with, whether she is the sender or the receiver. This is why it necessary to first compute the data in order to set a final Id_Contact.
# 
# When Hillary Clinton is the email receiver, things are quite easy because sqllite gives us the sender ID. When Hillary Clinton is the sender, things are a little more tricky. This step also allow us to clean a little the data, and remove the emails where Hillary Clinton is the sender, but there is no trace of who is the receiver.

# In[ ]:


def computeSender(item):
    # Sender is Hillary Clinton
    if item.Id_sender == 80 and item.MetadataTo != '' and np.isnan(item.PersonId):
        tab = item.MetadataTo.split(',')
        name = tab[1].strip() + ' ' + tab[0].strip() 
        tmp = pd.read_sql_query("SELECT Id, Name FROM Persons WHERE Name='"+ name +"'", con)
        # A person was found
        if not tmp.empty:
            item.PersonId = tmp['Id'][0]
    # Create the new Contact column
    if item.Id_sender == 80:
        item['Id_Contact'] = item.PersonId
    else:
        item['Id_Contact'] = item.Id_sender
    return item
print("Number of emails before cleaning : ",emails.shape[0])

data = emails.apply(computeSender, axis=1);

# Remove the not found persons
data = data[(~np.isnan(data.PersonId)) | (data.Id_sender != 80)]
data = data[data.Id_Contact != 80]
data['Id_Contact'] = data['Id_Contact'].apply(lambda i : personsDict[int(i)])

print("Number of emails after cleaning : ",data.shape[0])
print("Number of unique contacts : ", data['Id_Contact'].unique().shape[0])


# ## Grouping email content by contact
# 
# Next step is to get the emails, and group them by contact. In the end, each contact will have all the emails content he sent or received.

# In[ ]:


corpusTmp = {}
corpus = {}

for i, email in enumerate(data.values):
    corpusTmp[email[5]] = corpusTmp.get(email[5], "") + email[3]
    
occ = []
for key, val in corpusTmp.items():
    if int(len(val)) > 10:
        corpus[key] = val
contacts = list(corpus.keys())


# # Using NLTK to clean our data and compute TFIDFs
# 
# Before we calcul the TFIDFs, we need to tokenize the content grouped by contact. Tokenization will take car of pruning stop words. Stemming is also an important step to improve precision.

# In[ ]:


import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for contactId, text in corpus.items():
    lowers = text.lower()
    no_punctuation = lowers.translate(string.punctuation)
    corpus[contactId] = no_punctuation
        
# because documents are of different size, it is important to normalize !
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', norm='l2')
tfs = tfidf.fit_transform(corpus.values())


# # Computing distance between contacts
# 
# To calculate the cosinus similarities, we calculate the distance between all the combinaisons of contact pairs.
# 
# In this analysis, we will focus on the best relationships that might exist between Hillary Clinton's contacts. This is why we introduced 2 parameters :
# - threshold : minimum cosinus similarity we will take into account
# - limit : maximal number of contacts that a contact ca be linked to (we keep the "n" best similarities)

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import operator
similarity = cosine_similarity(tfs[0:len(contacts)], tfs)

links = {}
threshold = 0.5
limit = 5
x = []
contactsGraph = []

for i in range(len(contacts)):
    tmp = {}
    for j in range(i):
        if similarity[int(i),int(j)] > threshold:
            contactsGraph.append(int(i))
            contactsGraph.append(int(j))
            tmp["%s/%s" % (contacts[int(i)],contacts[int(j)])] = similarity[int(i),int(j)]
    tmp = sorted(tmp.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(limit):
        if i < len(tmp):
            links[tmp[i][0]] = tmp[i][1]


# # Vizualisation
# 
# In order to vizualize, we will create graphs because this kind of plot suits the best to see the link between Hillary Clinton's contacts.

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('matplotlib', 'inline')

import networkx as nx
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (50, 50)
G=nx.Graph()

for key, val in links.items():
    n1, n2 = key.split('/')
    G.add_edge(n1, n2, weight=round(val, 3))

nodes = dict()
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    nodes[i] = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] < i and d['weight'] >= round(i-0.1, 2)]

pos=nx.spring_layout(G)

nx.draw_networkx_nodes(G,pos,node_size=0,node_color='yellow')

for key, val in nodes.items():
    nx.draw_networkx_edges(G,pos,edgelist=val, width=1, style='dashed') 

nx.draw_networkx_labels(G,pos,font_size=18,font_family='sans-serif')

plt.axis('off')
plt.show()


# # Analysing the result graph : unexpected results
# 
# To view bette the graph, right-click on it and select "open in new window". There, you will see it full sized.
# 
# Basically, the analysis results did not go as i would have expected. I thought I would see groups of contacts, meaning that Hillary Clinton's contacts are gathered by topics. After performing the cosinus calculation for different thresholds and different limits, I realized that there is just 1 group gathering all the contacts. 
# 
# This could be explained as follow : The emails chosen for the analysis have some close content. It is then impossible to cluster them by topics. Maybe the emails chosen for the analysis were dealing with the same subject. A quick look at other Kaggle submission that focused on Hillary Clinton's topics show that actually, the emails subjects are quite similar.

# # Analysis brought other information
# 
# The vizualisation of the graphs showed something else, also very interesting. After playing around with thresholds and limits, I could see some kind of a structure. A few contacts are very much more connected than others. The number of connections they have is clearly significant, and we can easlily see hotspots on the graphs. 
# 
# I could reproduce this with various thresholds (from 0.2 to 0.9), always showing the same results.
# 
# So, what could this mean ?
# 
# At this step, I miss information to adress further conclusions, but it would appear that Hillary Clinton does not interact with her contacts as equal. Through the emails they have send and received, some of them may share topics with more contacts than others, acting a bit like relays. So, what gives them their ability to gather so many contacts ? Are thoses contacts mentoring some specific topics ?
# 
# A further analysis could be lead to get deeper into what kind of relationship Hillary Clinton shares with those 5 persons, and why they stand out from the crowd so significantly.
