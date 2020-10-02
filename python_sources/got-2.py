import numpy as np 
import pandas as pd
import sqlite3
sql_conn = sqlite3.connect('../input/database.sqlite')
#searcha = input("Enter the first GOT Character:")
#searchb = input("Enter the second GOT Character:")
searcha = "jon snow"
searchb = "arya stark"


search_split = []
i=searcha.find(" ",0,len(searcha))
j=searchb.find(" ",0,len(searchb))
search_split.append(searcha[0:i])
search_split.append(searchb[0:j])
search = ""


for k in search_split:
    search = search + 'AND body LIKE "%' + k +'%" '

  
sql_whole = 'SELECT body from May2015 WHERE ((subreddit="subreddit" AND subreddit_id="t5_2r2o9") or (subreddit="asoiaf" AND subreddit_id="t5_2rjz2"))'
sql= 'SELECT body from May2015 WHERE ((subreddit="subreddit" AND subreddit_id="t5_2r2o9") or (subreddit="asoiaf" AND subreddit_id="t5_2rjz2")) and (body LIKE "%' + searcha +'%" AND body LIKE "%' + searchb +'%") or (body LIKE' + search[13:] + ') and (author_flair_css_class IS NOT NULL)'
cw = sql_conn.execute(sql_whole)
c = sql_conn.execute(sql)

print("Executing SQL Code::::\n" + sql + "\n")
takes = []
used_takes = []
for r in cw:
    takes.append(r)
for q in c:
    used_takes.append(q)
index = 0
index_no = []
for i in takes:
    for j in used_takes:
        if(i==j):
            index_no.append(str(index))
    index = index + 1
main = []
track = 0

for k in index_no:
    track = int(k) + 1
    
    while(1):
        t = str(takes[track][0])
        if((t.find('he') or t.find('she') or t.find('it') or t.find('him') or t.find('her') or t.find('they')) and ( not(t.find(searcha)) and not(t.find(searchb)) and not(t.find(searcha[0:i])) and not(t.find(searchb[0:j])) )  ):
            print(t)
            track = track + 1
        else:
            break        
   
    









