import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# This script identifies which communication styles receive highest ranks
# For illustration purposes I defined 3 styles such as Passive, Assertive and Aggressive
# The list of key words must of course be extended

sql_conn = sqlite3.connect('../input/database.sqlite')

"""
df = pd.read_sql("SELECT subreddit, count(*) as 'Number of Comments'\
FROM May2015 \
GROUP BY subreddit \
HAVING count(*) > 250000 \
ORDER BY 2 DESC", sql_conn)
"""

df = pd.read_sql("SELECT created_utc,author,subreddit, body,ups,downs \
FROM May2015 \
WHERE author = 'Stonewater' \
AND subreddit = 'baseball' \
AND body like '%GIFs%'",sql_conn)

df['time'] = df['created_utc'].map(lambda x: datetime.datetime.fromtimestamp(x.item()).strftime('%Y-%m-%d %H:%M:%S'))
df.drop(['created_utc'],axis=1)

print(df)

#check out comments with common tv show / movie lines:
# that's what she said
# that's how you get ants
# phrasing
# did i do that
# i'm in a glass cage of emotions
