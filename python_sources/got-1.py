import sqlite3
import pandas as pd

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

  

sql= 'SELECT body,ups from May2015 WHERE ((subreddit="subreddit" AND subreddit_id="t5_2r2o9") or (subreddit="asoiaf" AND subreddit_id="t5_2rjz2")) and (body LIKE "%' + searcha +'%" AND body LIKE "%' + searchb +'%") or (body LIKE' + search[13:] + ') and (author_flair_css_class IS NOT NULL) order by ups desc LIMIT 5'

print("Executing sql code ::::::\n"+sql)
takes = []

c = sql_conn.execute(sql)
print("Executing sql code ::::::\n"+sql)


for row in c:
    takes.append(row)
   


df = pd.DataFrame(data=takes,columns=['body','ups'])
df.to_csv('got_1.csv')



