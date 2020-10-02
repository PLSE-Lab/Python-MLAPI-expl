# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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


pd.set_option('display.max_columns', 30)



con = sqlite3.connect('../input/database.sqlite') # connection to database
 
 
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
        
        
class GameOfThrone(object):
    def __init__(self, charac1="little finger", charac2="tywin lannister", db_path="../input/database.sqlite"):
        self.charac1 = "%"+charac1+"%"
        self.charac2 = "%"+charac2+"%"
        self.con = sqlite3.connect(db_path)
        self.subreddit1 = 'gameofthrones'
        self.subreddit2 = 'asoiaf'
        self.subreddit_id1 = 't5_2rjz2'
        self.subreddit_id2 = 't5_2r2o9'
        self.query1 = 'SELECT * from May2015 \
                    WHERE ((subreddit="{}" AND subreddit_id="{}") OR \
                    (subreddit="{}" AND subreddit_id="{}")) AND \
                    (body LIKE "{}" AND body LIKE "{}") \
                    ORDER BY ups DESC LIMIT 5'.format(self.subreddit1,self.subreddit_id1, \
                                                                       self.subreddit2,self.subreddit_id2, \
                                                                       self.charac1, self.charac2)
        
        self.query2 = 'SELECT * from May2015 \
                    WHERE ((subreddit="{}" AND subreddit_id="{}") OR \
                    (subreddit="{}" AND subreddit_id="{}")) AND \
                    (body LIKE "{}" AND body LIKE "{}") \
                    ORDER BY ups DESC LIMIT 10'.format(self.subreddit1,self.subreddit_id1, \
                                                                       self.subreddit2,self.subreddit_id2, \
                                                                       self.charac1, self.charac2)
            
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
    
    def SummarizeCharacter(self, charc="little finger"):
        charc = "%"+charc+"%"
        query = 'SELECT body,score from May2015 \
                        WHERE ((subreddit="gameofthrones" AND subreddit_id="t5_2rjz2") or \
                        (subreddit="asoiaf" and subreddit_id="t5_2r2o9")) and \
                        body LIKE "{}" '.format(charc)
        df = pd.read_sql_query(query, self.con)
        summary = df['body']
        #ss = collect_1char(df['body'], charc.replace("%",""))
        #text = "".join([i for i in ss])
        #summary = summarize(text, ratio=ratio)
        #key = keywords(text)
        return summary
        
# TEST 1 names = little finger & tywin lannister
got1 = GameOfThrone(charac1='little finger', charac2='tywin lannister')
comments1 = got1.PrintTop5Comments()  # top 5 comments
remove1, ups1 = got1.CommentChain()   #  comment chain
summary1 = got1.SummarizeCharacter('little finger')  # summarise 
print('top 5 comments: \n')
print(comments1);
print('Comment Chain displayed: \n')
pretty_print(remove1, ups1)
print('summary of little finger: \n')
print(summary1);
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory