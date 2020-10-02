# Which subbredits inspire the most controversial posts?

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

# There is so much data here I hit the 8gb memory limit if I try and grab it all
sql_cmd = "with t as(select subreddit, count(*) as the_count from May2015 group by 1) select * from t \
where the_count > 400000 and subreddit <> 'AskReddit' order by the_count desc" #

data = pd.read_sql(sql_cmd, sql_conn)
print(data)

plt.style.use('ggplot')
data.plot(kind='bar', x='subreddit', title='Comments per subreddit')
plt.xlabel('# of comments')
plt.ylabel('subreddit')
plt.tight_layout()
plt.show()
plt.savefig('mostCommentedSubreddits.png')