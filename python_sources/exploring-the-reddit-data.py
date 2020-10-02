import sqlite3
import pandas as pd
import ggplot
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')
#The Column Headings:
#created_utc
#ups
#subreddit_id
#link_id
#name
#score_hidden
#author_flair_css_class
#author_flair_text
#subreddit
#id 
#downs
#archived
#author
#score retrieved_on
#body
#distinguished edited

df = pd.read_sql("SELECT created_utc, subreddit, subreddit_id FROM May2015 LIMIT 1000", sql_conn)
datadict = dict()
subredditidtoname = dict()
for created, subreddit, subreddit_id in zip(df['created_utc'], df['subreddit'] ,df['subreddit_id']):
    if subreddit_id in datadict:
        if created in datadict[subreddit_id]:
            datadict[subreddit_id][created] = datadict[subreddit_id][created] + 1
        else:
            datadict[subreddit_id][created] = 1
    else:
        datadict[subreddit_id] = {}
        datadict[subreddit_id][created] = 1
        subredditidtoname[subreddit_id] = subreddit
print (datadict)
print (subredditidtoname)

for subredditid in subredditidtoname.keys():
    print (subredditid + ":" + subredditidtoname[subredditid])
