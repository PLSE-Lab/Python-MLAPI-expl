# Which subbredits have the most gilded posts?

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

# There is so much data here I hit the 8gb memory limit if I try and grab it all
sql_cmd = "Select subreddit, gilded From May2015 ORDER BY Random() LIMIT 5000000" #

data = pd.read_sql(sql_cmd, sql_conn)

#Which subreddits have (relatively) the most gilded posts?
allgroups = pd.pivot_table(data, index=['subreddit'], values=['gilded'], aggfunc=[np.mean, len])
withManyPosts = allgroups[allgroups[("len","gilded")] > 10000] # Subreddits with more than 50 posts
mostGilded = withManyPosts.sort([("mean","gilded")], ascending=False)
top20 = mostGilded[('mean','gilded')].head(20)
top20 = top20 * 100

print(top20)
top20.sort(ascending=True)
plt.style.use('ggplot')

y_pos = np.arange(len(top20))

plt.barh(y_pos, top20, align='center')
plt.yticks(y_pos, top20.index)
plt.title("Top 20 most gilded Subreddits, May 2015")
plt.xlabel("Percentage of gilded posts")
plt.tight_layout()
plt.margins(0.00,0.01)
plt.savefig('mostGildedSubreddits.png')

