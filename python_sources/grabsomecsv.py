import sqlite3
import pandas as pd
import numpy as np
import re
import csv

sql_conn = sqlite3.connect('../input/database.sqlite')

header = 'name, parent_id, link_id, subreddit, score, gilded, controversiality, distinguished, edited, created_utc, body'
subreddits = [
    "AdviceAnimals", "AskReddit", "askscience", "atheism", "aww", "bestof", "books", "creepy", "dataisbeautiful", "DIY", "Documentaries", "explainlikeimfive", "Fitness", "food", "funny", "Futurology", "gadgets", "gaming", "gifs", "history", "IAmA", "InternetIsBeautiful", "Jokes", "LifeProTips", "mildlyinteresting", "Music", "news", "nosleep", "nottheonion", "OldSchoolCool", "personalfinance", "photoshopbattles", "pics", "politics", "science", "Showerthoughts", "space", "sports", "tifu", "todayilearned", "TwoXChromosomes", "UpliftingNews", "videos", "worldnews", "WritingPrompts", "WTF"
]

# We need this extra SELECT * wrapper because LIMIT and UNION aren't best friends.
# Get 30000 from each subreddit, ignoring comments whose score is in [-1, 1].
query1 = (
    "SELECT * "
    "FROM   ( SELECT " + header + ' '
    "         FROM May2015 "
    "         WHERE subreddit='%s' AND body<>'' AND body<>'[deleted]' AND (score > 50) "
    "         LIMIT 10 "
    "       )"
)

query2 = (
    "SELECT * "
    "FROM   ( SELECT " + header + ' '
    "         FROM May2015 "
    "         WHERE subreddit='%s' AND body<>'' AND body<>'[deleted]' AND (score <= 50) "
    "         LIMIT 10 "
    "       )"
)

queries1 = " UNION ".join(query1 % s for s in subreddits)
queries2 = " UNION ".join(query2 % s for s in subreddits)
queries = " UNION ".join([queries1, queries2])

print('Running the query and loading into a list of lists...')
res = pd.read_sql(queries, sql_conn)
df = pd.DataFrame(res)
rows = [header.split(', ')]
for i in range(len(res)):
    rows.append(df.iloc[i])

print('Writing to a CSV file...')
with open('reddit_data.csv', 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerows(rows)
    
print('Great success! =]')