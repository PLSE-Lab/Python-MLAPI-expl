import sqlite3
conn = sqlite3.connect('../input/database.sqlite')

results = conn.execute("SELECT COUNT(*) from May2015 WHERE  subreddit LIKE 'soccer' and LENGTH(body)<50 and Length(body) > 15  and body LIKE '%youtube.com%' ")

print (results)