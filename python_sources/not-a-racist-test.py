import sqlite3

print('Querying...\n')
LIMIT = 2500

sql_conn = sqlite3.connect('../input/database.sqlite')
data = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE body LIKE '% /s'\
                                LIMIT " + str(LIMIT))

for post in data:
    print(post[1])
    print("\n")