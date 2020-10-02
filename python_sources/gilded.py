import sqlite3
import pandas as ps

conn = sqlite3.connect('../input/database.sqlite')

gild = conn.execute("SELECT body, author, gilded,score, subreddit from May2015 WHERE  gilded > 10 or score > '3000'")

for comment in gild:
    print (comment[1])
    print (comment[2])
    print (comment[3])
    print (comment[0])
    
    print ('\n')