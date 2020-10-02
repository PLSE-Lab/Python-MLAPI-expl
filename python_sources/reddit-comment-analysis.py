import sqlite3 as woo
import pandas as pig
from nltk.corpus import sentiwordnet as sooie
import numpy as go
import matplotlib.pyplot as hogs


# Identify the number of positive, negative, and neutral opinionated comments
# The following sentiwordnet criteria will be used:
# Positive - Highly Positive
# Negative - Highly Negative
# Neutral - Lowly Positive, Lowly Negative


def get_sentiscores(x):
    return list(sooie.senti_synsets(x))
    
def get_pos_score(comment):
    if len(comment) > 0:
        return comment[0].pos_score()
    return 0

def get_neg_score(comment):
    if len(comment) > 0:
        return comment[0].neg_score()
    return 0
    
def get_neut_score(comment):
    if len(comment) > 0:
        #logic to return the neutral score based on positive and negative score
        return comment[0].obj_score()
    return 0
    
    
    
sql_conn = woo.connect('../input/database.sqlite')

df0 = pig.read_sql("SELECT author, score, body as Comment FROM May2015 WHERE LENGTH(body) > 30 AND LENGTH(body) < 250 LIMIT 100", sql_conn)
df = pig.read_sql("SELECT body as Comment, score FROM May2015 WHERE LENGTH(body) > 30 AND LENGTH(body) < 250 LIMIT 100", sql_conn)
df2 = pig.read_sql("Select count(*) as cnt From May2015 Group By author Order By cnt desc LIMIT 100",sql_conn)
keywords = ['Positive', 'Negative', 'Neutral']

content_summary = pig.DataFrame()


pos_content = []
neg_content = []
neut_content = []
author_pos = dict()
author_pos_content = []
author_neg = dict()
author_neg_content = []
author_neut = dict()
auth_neut_content = []
cnt = 0
# get average score for all words in the comments
for string in df['Comment'].values:
    strings = string.split(" ")
    string_scores = list(map(lambda x: get_sentiscores(x), strings))
    pos_scores = list(map(lambda x: get_pos_score(x), string_scores))
    neg_scores = list(map(lambda x: get_neg_score(x), string_scores))
    neut_scores = list(map(lambda x: get_neut_score(x), string_scores))
    curr_author = df0.iloc[cnt]['author']
    cnt += 1   
    if curr_author in author_pos:
        # sum the new value to the existing one 
        author_pos[curr_author] = round(((author_pos[curr_author] + go.mean(pos_scores))*100)/cnt,2)
    else:
        author_pos[curr_author] = round(go.mean(pos_scores)*100,2)
        
    if curr_author in author_neg:
        # sum the new value to the existing one 
        author_neg[curr_author] = round(((author_neg[curr_author] + go.mean(neg_scores))*100)/cnt,2)
    else:
        author_neg[curr_author] = round(go.mean(neg_scores)*100,2)
        
    if curr_author in author_neut:
        # sum the new value to the existing one 
        author_neut[curr_author] = round(((author_neut[curr_author] + go.mean(neut_scores))*100)/cnt,2)
    else:
        author_neut[curr_author] = round(go.mean(neut_scores)*100,2)
                        
    pos_content.append(go.mean(pos_scores))
    neg_content.append(go.mean(neg_scores))
    neut_content.append(go.mean(neut_scores))

author_name_content = author_pos.keys()
author_pos_content = author_pos.values()
author_neg_content = author_neg.values()
author_neut_content = author_neut.values()
df['Positive'] = pos_content
df['Negative'] = neg_content
df['Neutral'] = neut_content
df2['Author'] = author_name_content
df2['Positive%'] = author_pos_content
df2['Negative%'] = author_neg_content
df2['Neutral%'] = author_neut_content

print('*****Mean values of Positive, Negative and Neutral scores Per comment******')
print(df)
print('*****Mean score Per author(owner of multiple comments) in positive, Negative and Neutral categories******')
print(df2)

print(pos_scores)    


keys = keywords


content_summary = content_summary.transpose()




#pos = [pos_content, neut_content, neg_content]

#pos = list(range(len(content_summary['count'])))

h = [len(neut_scores), len(pos_scores), len(neg_scores)]

N = len(h)

x = range(N)

width = .5

fig, ax = hogs.subplots(figsize=(10,5))

#bar_colors = []

hogs.bar(x, h, width, alpha = .05, color = 'r', label = keys)

	  
ax.set_ylabel('Number of comments')
ax.set_title('Reddit Comment Analysis')
#ax.set_xticklabels(keys)


hogs.grid()

hogs.savefig("CommunicationStyles.png")