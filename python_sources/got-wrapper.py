#!/usr/bin/env python
# coding: utf-8

# #GOT extraction

# Loading Libs

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import sqlite3
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import string
import itertools
from gensim.summarization import summarize, keywords # lib for summarisation

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 30)


# Connection to database (.sqlite)

# In[ ]:


con = sqlite3.connect('../input/database.sqlite') # connection to database


# Gameofthrones

# In[ ]:


subreddit = 'gameofthrones'  # topic 
query = 'SELECT * from May2015 WHERE (subreddit="{}") LIMIT 10'.format(subreddit) # SQl query
df_got = pd.read_sql_query(query, con) # data


# In[ ]:


choose = ['subreddit_id','subreddit','body','ups'] # cols to look in
df_got[choose]


# asioaf

# In[ ]:


subreddit = 'asoiaf'
query = 'SELECT * from May2015 WHERE (subreddit="{}") LIMIT 10'.format(subreddit)
df_as = pd.read_sql_query(query, con)


# In[ ]:


choose = ['subreddit_id','subreddit','body','ups',]
df_as[choose]


# <h3>Challenges</h3>
# Challenge 1: Given any two GoT characters display top 5 comments that are about both of these characters 
# 
# Challenge 2: Display a comment chain about the same two characters but do not contain their names. 
# 
# Challenge 3: Given a GoT character summarise or explain who the character is.

# Creating wrapper to extrcact according to challenge

# In[ ]:


# summarise the character
def summary_text(summary):
   jj = []
   tok = [nltk.word_tokenize(summary[i]) for i in range(len(summary))]
   merged = list(itertools.chain(*tok))
   merge = [word for word in merged if word not in string.punctuation]
   merge_stop = [w.lower() for w in merge if w.lower() not in stop]
   words = FreqDist(merge_stop).most_common(500)
   items = [words[i][0] for i in range(len(words))]
   pos_t = nltk.pos_tag(items)
   if pos_t:
       for i in range(len(items)):
           if pos_t[i][1] == 'JJ':
               jj.append(pos_t[i][0])
               text = " ".join([item for item in jj])
   return text


def collect_1char(d, name1): # collect comment containing specific name.
   c = [d[i].split('\n') for i in range(len(d))]
   merged = list(itertools.chain(*c))
   sent = []
   for i in range(len(merged)):
       if name1 in merged[i].lower():
           sent.append(merged[i].lower())
   return sent



def collect_sent(d, name1, name2): # # collect comment containing any 2 name.
   c = [d[i].split('\n') for i in range(len(d))]
   merged = list(itertools.chain(*c))
   sent = []
   for i in range(len(merged)):
       if name1 and name2 in merged[i].lower():
           sent.append(merged[i].lower())
   return sent, len(sent)


# remove character name    
def remove_name(df, charc1, charc2):
   remove = [df[i].lower().replace(charc1, '').replace(charc2, '') for i in range(len(df))]
   return remove


# print comment chain
def pretty_print(comments, leng):
   print ("Contains - {} comments".format(leng))
   print ("-"*50)
   for i in range(len(comments)):
       print ("-"*70)
       print ("Comment : {}".format(i+1))
       print ("_"*70)
       print (comments[i])


# In[ ]:


class GameOfThrone(object):
    def __init__(self, charac1="tyrion lannister", charac2="jon snow", db_path="../input/database.sqlite"):
        self.charac1 = "%"+charac1+"%"
        self.charac2 = "%"+charac2+"%"
        self.con = sqlite3.connect(db_path)
        self.subreddit1 = 'gameofthrones'
        self.subreddit2 = 'asoiaf'
        self.subreddit_id1 = 't5_2rjz2'
        self.subreddit_id2 = 't5_2r2o9'
        self.query1 = 'SELECT * from May2015                     WHERE ((subreddit="{}" AND subreddit_id="{}") OR                     (subreddit="{}" AND subreddit_id="{}")) AND                     (body LIKE "{}" AND body LIKE "{}")                     ORDER BY ups DESC LIMIT 5'.format(self.subreddit1,self.subreddit_id1,                                                                        self.subreddit2,self.subreddit_id2,                                                                        self.charac1, self.charac2)
        
        self.query2 = 'SELECT * from May2015                     WHERE ((subreddit="{}" AND subreddit_id="{}") OR                     (subreddit="{}" AND subreddit_id="{}")) AND                     (body LIKE "{}" AND body LIKE "{}")                     ORDER BY ups DESC LIMIT 10'.format(self.subreddit1,self.subreddit_id1,                                                                        self.subreddit2,self.subreddit_id2,                                                                        self.charac1, self.charac2)
            
             #AND (body LIKE "%[Answers]%")) \
        
  
    def PrintTop5Comments(self):
        df = pd.read_sql_query(self.query1, self.con)
        return df[['body','ups']]
    
    def CommentChain(self):
        charc1 = self.charac1.replace('%', '')
        charc2 = self.charac2.replace('%', '')
        df = pd.read_sql_query(self.query2, self.con)
        s, l = collect_sent(df['body'], charc1, charc2)
        return s, l
    
    def SummarizeCharacter(self, charc="jon snow"):
        charc = "%"+charc+"%"
        query = 'SELECT body,score from May2015                         WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or                         (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and                         body LIKE "{}" '.format(charc)
        df = pd.read_sql_query(query, self.con)
        summary = df['body']
        #ss = collect_1char(df['body'], charc.replace("%",""))
        #text = "".join([i for i in ss])
        #summary = summarize(text, ratio=ratio)
        #key = keywords(text)
        return summary


# <h3>TESTING</h3>
# ####different names are used
# 
#  1. Jon Snow  
#  2. Daenerys Targaryen
#  3. Tyrion Lannister 
#  4. Arya Stark 
#  5. Bran Stark
#  6. Cersei

# In[ ]:


# TEST 1 names = jon snow & arya stark
got1 = GameOfThrone(charac1='jon snow', charac2='arya stark')
comments1 = got1.PrintTop5Comments()  # top 5 comments
remove1, ups1 = got1.CommentChain()   #  comment chain
summary1 = got1.SummarizeCharacter('jon snow')  # summarise 


# ###Top 5 Comments

# In[ ]:


comments1 # top 5


# ### Comment Chain

# In[ ]:


pretty_print(remove1, ups1) # comment chain about 2 charac


# ###Summary WordCLOUD

# In[ ]:


charc1 = "jon snow"
df1 = pd.read_sql_query('SELECT body,score from May2015         WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or         (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and        (body LIKE "%{}%" AND body LIKE "%arya%") LIMIT 30'.format(charc1), con) # extracting data


# In[ ]:


d1 = df1['body']
s1 = collect_1char(d1, "jon snow") # extracting comment for specific name


# In[ ]:


text1 = "".join([i for i in s1]) # making it to one text


# In[ ]:


summarize(text1, ratio=0.1) # summarising 10% of content 


# In[ ]:


#wordcloud = WordCloud().generate(summary1)
#plt.imshow(wordcloud)


# ##TEST 2
# 
# names = Bran Stark & Cersei lannister

# In[ ]:


got2 = GameOfThrone(charac1='bran stark', charac2='cersei lannister')
comments2 = got2.PrintTop5Comments()
remove2, ups2 = got2.CommentChain()
summary2 = got2.SummarizeCharacter('lannister')


# In[ ]:


comments2


# In[ ]:


pretty_print(remove2, ups2)


# Summary2

# In[ ]:


charc2 = "cersei lannister"
df2 = pd.read_sql_query('SELECT body,score from May2015         WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or         (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and        (body LIKE "%{}%" AND body LIKE "%arya%") LIMIT 30'.format(charc2), con) # extracting data


# In[ ]:


d2 = df2['body']
s2 = collect_1char(d2, "cersei lannister") # extracting comment for specific name


# In[ ]:


text2 = "".join([i for i in s2]) # making it to one text


# In[ ]:


summarize(text2, ratio=0.1) # summarising 10% of content 


# In[ ]:


#wordcloud = WordCloud().generate(summary2)
#plt.imshow(wordcloud)


# ## TEST 3
# 
# names = ramsay bolton & doran martell

# In[ ]:


got3 = GameOfThrone(charac1='ramsay', charac2='doran')
comments3 = got3.PrintTop5Comments()
remove3, ups3 = got3.CommentChain()
summary3 = got3.SummarizeCharacter('martell')


# In[ ]:


comments3


# In[ ]:


pretty_print(remove3, ups3)


# summary 3

# In[ ]:


charc3 = "doran"
df3 = pd.read_sql_query('SELECT body,score from May2015         WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or         (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and        (body LIKE "%{}%" AND body LIKE "%arya%") LIMIT 30'.format(charc3), con) # extracting data
d3 = df3['body']
s3 = collect_1char(d3, "cersei lannister") # extracting comment for specific name
text3 = "".join([i for i in s3]) # making it to one text
summarize(text3, ratio=0.5) # summarising 10% of content


# In[ ]:


#wordcloud = WordCloud().generate(summary3)
#plt.imshow(wordcloud)


# # END

# ###TL;DR
# 
# procedure during creation of functions and wrapper

# In[ ]:


summary = got1.SummarizeCharacter(charc='arya stark')


# In[ ]:


summary


# In[ ]:


lists = ggg['body'].values


# In[ ]:


df = pd.read_sql_query('SELECT body,score from May2015         WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or         (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and        (body LIKE "%jon snow%" AND body LIKE "%arya%") LIMIT 30', con)


# In[ ]:


d = df['body']
s = collect_1char(d, "jon snow")


# In[ ]:


s


# In[ ]:


text = "".join([i for i in s])


# In[ ]:


summarize(text, ratio=0.1)


# In[ ]:


from gensim.summarization import summarize, keywords


# In[ ]:


def collect_1char(d, name1):
    c = [d[i].split('\n') for i in range(len(d))]
    merged = list(itertools.chain(*c))
    sent = []
    for i in range(len(merged)):
        if name1 in merged[i].lower():
            sent.append(merged[i].lower())
    return sent


# In[ ]:


ss = collect_1char(d, "cersei")
join = "".join([i for i in ss])


# In[ ]:


print ('Summary:')
print (summarize(join, ratio=0.5))


# In[ ]:


print ('Keywords:')
print (keywords(join))


# In[ ]:


print ('Summary:')
print (summarize(text, ratio=0.01, split=True))


# In[ ]:


print ('Keywords:')
print (keywords(text))


# In[ ]:


c = [d[i].split('\n') for i in range(len(d))]
merged = list(itertools.chain(*c))
sent = []
for i in range(len(merged)):
    if "jon snow" and "arya stark" in merged[i].lower():
        sent.append(merged[i].lower())


# In[ ]:


def collect_sent(d, name1, name2):
    c = [d[i].split('\n') for i in range(len(d))]
    merged = list(itertools.chain(*c))
    sent = []
    for i in range(len(merged)):
        if name1 and name2 in merged[i].lower():
            sent.append(merged[i].lower())
    return sent, len(sent)


# In[ ]:


def select_comment(d, name1, name2):
    c = [d[i].split('\n') for i in range(len(d))]
    merged = list(itertools.chain(*c))
    sent = []
    for i in range(len(merged)):
        if name1 and name2 in merged[i].lower():
            sent.append(merged[i].lower())
    return sent, len(sent)


# In[ ]:


f,g = select_comment(d, "jon", "arya")


# In[ ]:


f


# In[ ]:


c = [d[i].split('\n') for i in range(len(d))]
#    d[i].split('\n')
#d[9].split('\n')[22].lower()
#for i in range(len(c)):
#    print (len(c[i]))

#for i in range(len(c)):
#    name = "stark" or "snow"
#    l = int(len(c[i]))
#    print (range(int(len(c[i]))))
#    for c in range(l):
#        print (c)
        #if name in c[i][l].lower():
            #print (c[i][l])
        
#c
merged = list(itertools.chain(*c))
merged[100]


# In[ ]:


sent = []
for i in range(len(merged)):
    if "arya stark" and "cersei" in merged[i].lower():
        print ("-"*50)
        print (merged[i].lower())
        sent.append(merged[i].lower())


# In[ ]:


from gensim.summarization import summarize, keywords

text = "Thomas A. Anderson is a man living two lives. By day he is an " +     "average computer programmer and by night a hacker known as " +     "Neo. Neo has always questioned his reality, but the truth is " +     "far beyond his imagination. Neo finds himself targeted by the " +     "police when he is contacted by Morpheus, a legendary computer " +     "hacker branded a terrorist by the government. Morpheus awakens " +     "Neo to the real world, a ravaged wasteland where most of " +     "humanity have been captured by a race of machines that live " +     "off of the humans' body heat and electrochemical energy and " +     "who imprison their minds within an artificial reality known as " +     "the Matrix. As a rebel against the machines, Neo must return to " +     "the Matrix and confront the agents: super-powerful computer " +     "programs devoted to snuffing out Neo and the entire human " +     "rebellion. "

print ('Input text:')
print (text)


# In[ ]:


print ('Summary:')
print (summarize(text))


# In[ ]:


print ('Keywords:')
print (keywords(text))


# In[ ]:


def summary_text(summary):
    jj = []
    tok = [nltk.word_tokenize(summary[i]) for i in range(len(summary))]
    merged = list(itertools.chain(*tok))
    merge = [word for word in merged if word not in string.punctuation]
    merge_stop = [w.lower() for w in merge if w.lower() not in stop]
    words = FreqDist(merge_stop).most_common(500)
    items = [words[i][0] for i in range(len(words))]
    pos_t = nltk.pos_tag(items)
    if pos_t:
        for i in range(len(items)):
            if pos_t[i][1] == 'JJ':
                jj.append(pos_t[i][0])
    text = " ".join([item for item in jj])
    return text


# In[ ]:


import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import string


# In[ ]:


tok = [nltk.word_tokenize(summary[i]) for i in range(len(summary))]


# In[ ]:


import itertools
merged = list(itertools.chain(*tok))
len(merged)


# In[ ]:


merge = [word for word in merged if word not in string.punctuation]
merge_stop = [w.lower() for w in merge if w.lower() not in stop]


# In[ ]:


merge_stop[:10]


# In[ ]:


words = FreqDist(merge_stop).most_common(500)


# In[ ]:


words


# In[ ]:


items = [words[i][0] for i in range(len(words))]


# In[ ]:


pos_t = nltk.pos_tag(items)


# In[ ]:


jj = []
for i in range(len(items)):
    if pos_t[i][1] == 'JJ':
        jj.append(pos_t[i][0])


# In[ ]:


jj


# In[ ]:


text = " ".join([item for item in jj])


# In[ ]:


from wordcloud import WordCloud 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

wordcloud = WordCloud().generate(text)
#wordcloud.fit_words(tuple_words)
plt.imshow(wordcloud)


# In[ ]:


summary_text(summary)

