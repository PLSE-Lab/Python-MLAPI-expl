import sqlite3
import matplotlib.pyplot as plt
import numpy as np

#connect to the db
con = sqlite3.connect('../input/database.sqlite')

#get the most commented subreddits
top_subreddits = {}
query = "select subreddit, count(subreddit) from May2015 group by subreddit order by count(subreddit) desc limit 20;" 
with con:    
    cur = con.cursor()    
    cur.execute(query)
    rows = cur.fetchall()
    for row in rows:
        top_subreddits[row[0]] = row[1]
        #print(row[0] + " --- " + str(row[1]) )
    #print(top_subreddits)

#get the score and controversiality of the top subreddits
score_query = "select subreddit, sum(score), sum(controversiality) from May2015 where subreddit IN " + str(top_subreddits.keys()).replace("[", "").replace("]","").replace("'","\'").replace("dict_keys","") + " group by subreddit order by sum(score) desc;"
controversial = {}
scoring = {}
#print(score_query)
with con:    
    cur = con.cursor()    
    cur.execute(score_query)
    rows = cur.fetchall()
    for row in rows:
        print(row[0]+ " --- " + str(top_subreddits[row[0]]) + " --- " + str(row[1]/top_subreddits[row[0]]) + " --- " + str(row[2]/top_subreddits[row[0]]))
        controversial[row[0]] = row[2]/top_subreddits[row[0]]
        scoring[row[0]] = row[1]/top_subreddits[row[0]]

subreddit_list = []
num_comments_list = []
controversial_list = []
scoring_list = []

for key in top_subreddits:
    subreddit_list.append(key)
    num_comments_list.append(top_subreddits[key])
    controversial_list.append(controversial[key])
    scoring_list.append(scoring[key])
# create the plots
ind = np.arange(len(subreddit_list)) + 0.76

fig2 = plt.figure()
plt.bar(range(len(subreddit_list)), scoring_list)
plt.xticks(ind, subreddit_list)
fig2.autofmt_xdate()
plt.savefig("score_per_comment.png")

fig3 = plt.figure()
plt.bar(range(len(subreddit_list)), controversial_list)
plt.xticks(ind, subreddit_list)
fig3.autofmt_xdate()
plt.savefig("controversiality_per_comment.png")

fig1 = plt.figure()
plt.bar(range(len(subreddit_list)), num_comments_list)
plt.xticks(ind, subreddit_list)
fig1.autofmt_xdate()
plt.savefig("top_commended.png")
