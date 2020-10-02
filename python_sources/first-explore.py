import sqlite3

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
for row in c.execute('select subreddit, count(*) from May2015 group by subreddit order by count(*) desc limit 100'):
    print(row)

conn.close()