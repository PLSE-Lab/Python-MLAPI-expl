import sqlite3
from collections import defaultdict

sql_conn = sqlite3.connect('../input/database.sqlite')

test = sql_conn.execute("SELECT author, subreddit FROM May2015")

# Create a dictionary with the users for keys, and list all subreddits they posted in
user_reddits = defaultdict(list)

for i, k in enumerate(test):
    if k[0] != "[deleted]": # Skip deleted user accounts
        user_reddits[k[0]].append(k[1])
    
# Now discard users that posted only in one subreddit
user_reddits = {k:list(set(v)) for k,v in user_reddits.items() if len(set(v)) > 1}

print(user_reddits)