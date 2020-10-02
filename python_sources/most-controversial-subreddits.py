# Which subbredits inspire the most controversial posts?

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')
sql_cmd = "Select subreddit, controversiality From May2015 ORDER BY Random() LIMIT 1000000" #

data = pd.read_sql(sql_cmd, sql_conn)
#print(data.describe())

#Which subreddits have (relatively) the most controversial posts?
allgroups = pd.pivot_table(data, index=['subreddit'], values=['controversiality'], aggfunc=[np.mean, len])
withManyPosts = allgroups[allgroups[("len","controversiality")] > 50] # Subreddits with more than 50 posts
mostControversial = withManyPosts.sort([("mean","controversiality")], ascending=False)
top20 = mostControversial[('mean','controversiality')].head(20)
top20 = top20 * 100

print(top20)
top20.sort(ascending=True)
plt.style.use('ggplot')

y_pos = np.arange(len(top20))

plt.barh(y_pos, top20, align='center')
plt.yticks(y_pos, top20.index)
plt.title("Top 20 most controversial Subreddits, May 2015")
plt.xlabel("Percentage of controversial posts")
plt.tight_layout()
plt.margins(0.00,0.01)
plt.savefig('mostControversialSubreddits.png')

