import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plot limit
plot_limit = 20

#connect to database
sql_conn = sqlite3.connect('../input/database.sqlite')

#group comments by subreddit and then count authors in every subreddit
subreddits = sql_conn.execute("Select subreddit, count(distinct(author)) as user_sum From May2015 group by subreddit")
rezz = subreddits.fetchall()
#sort result by number of unique authors
rezz = sorted(rezz, key=lambda tup: tup[1], reverse=True)[:plot_limit]

#unzip list of tuples
subreddits, users_num = zip(*(rezz))
users_num = list(users_num)
subreddits = list(subreddits)
indexes = np.arange(len(subreddits))[::-1]

#plot
plt.subplot(111, axisbg='#555658')
plt.barh(indexes, users_num, color='#20beff', alpha=0.92)
plt.yticks(indexes, subreddits)
plt.tight_layout()
plt.show()
plt.savefig('NumberOfSubredditUsers.png')