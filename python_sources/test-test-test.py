import sqlite3
import pandas as pd
import random
import sys
import numpy as np

connect = sqlite3.connect('../input/database.sqlite')

all_red = pd.read_sql("select subreddit, count(*) as total_comments from May2015 group by subreddit;", connect)

# Mean Comments = 1087, Median Comments = 7 (per subreddit)

all_red = all_red[all_red['total_comments'] > 10000]

# 779 subreddits with at least 10,000 comments

subs = all_red['subreddit'].tolist() #all the subs we want to loop over
random.shuffle(subs)
print(subs)
#sys.exit("Error Message")

# for each subreddit on this list we are going to extract the top 1% and 
# bottom 1% of comments

final_data_best=pd.DataFrame()
final_data_worst=pd.DataFrame()
for counter in range(0,76):
    sql_text = "select subreddit, body, score, gilded from May2015 where subreddit = '%s';" % subs[counter]
    # print(sql_text)
    wood = pd.read_sql(sql_text, connect)
    run = wood.sort(['score'])
    x = len(run)
    x = int(x * .01)

    worst=run.head(x)
    best=run.tail(x)
    final_data_best=final_data_best.append(best,ignore_index=True)
    final_data_worst=final_data_worst.append(worst,ignore_index=True)

# final_data_best = final_data_best[pd.notnull(final_data_best['score'])]

final_data_worst.to_csv("worst_comments.csv",index=False)
final_data_best.to_csv("best_comments.csv",index=False)









