import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

#CASE WHEN body in('That''s what she said','that''s what she said', 'thats what she said','Thats what she said') THEN count(*)
#that's what she said
df = pd.read_sql("SELECT subreddit, count(*) as 'Number of Comments'\
FROM May2015 \
WHERE body in('That''s what she said','Thats what she said','that''s what she said','thats what she said') \
GROUP BY subreddit \
ORDER BY 2 DESC", sql_conn)

print(df)

##HAVING count(*) > 250000 \


#check out comments with common tv show / movie lines:
# that's what she said
# that's how you get ants
# phrasing
# did i do that
# i'm in a glass cage of emotions
