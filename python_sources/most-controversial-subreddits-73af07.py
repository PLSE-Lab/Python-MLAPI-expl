# Which subbredits inspire the most controversial posts?

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

# There is so much data here I hit the 8gb memory limit if I try and grab it all
# sql_cmd = "Select subreddit, controversiality From May2015 ORDER BY Random() LIMIT 5000000" #
sql_cmd = "Select subreddit, controversiality From May2015"

data = pd.read_sql(sql_cmd, sql_conn)

#Which subreddits have (relatively) the most controversial posts?
allgroups = pd.pivot_table(data, index=['subreddit'], values=['controversiality'], aggfunc=[np.mean, len])
withManyPosts = allgroups[allgroups[("len","controversiality")] > 50] # Subreddits with more than 50 posts
mostControversial = withManyPosts.sort_values([("mean","controversiality")], ascending=False)
top20 = mostControversial[('mean','controversiality')].head(20)
top20 = top20 * 100

print(top20)
top20.sort_index(ascending=True)
plt.style.use('ggplot')

y_pos = np.arange(len(top20))

plt.barh(y_pos, top20, align='center')
plt.yticks(y_pos, top20.index)
plt.title("Top 20 most controversial Subreddits, May 2015")
plt.xlabel("Percentage of controversial posts")
plt.tight_layout()
plt.margins(0.00,0.01)
plt.savefig('mostControversialSubreddits.png')

