import sqlite3, random, json

srs_lmt = 10
sar_lmt = 10

sql_conn = sqlite3.connect('../input/database.sqlite')

sarcasmData = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE body LIKE '% /s' \
                                LIMIT " + str(sar_lmt))

seriousData = sql_conn.execute("SELECT subreddit, body, score FROM May2015\
                                WHERE body NOT LIKE '%/s%'\
                                LIMIT " + str(srs_lmt))


def get_as_dict(sql_cursor):
  return [{"subreddit": post[0], "body": post[1], "score": post[2]}
          for post in sql_cursor]

all_data = get_as_dict(sarcasmData) + get_as_dict(seriousData)
random.shuffle(all_data)

for post in all_data:
    print(json.dumps(post))