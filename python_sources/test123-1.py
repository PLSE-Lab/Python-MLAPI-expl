"""
Author: Yash Patel
Name: Reddit_Time_Optimization.py
Description: Look into the correlations in Reddit database, primarily looking to resolve the 
following questions of interest, all dealing with optimizing a reddit post's score:
1) score vs created_utc (Is there best time of day to release a reddit for positive attention?)
2) controversiality vs score (Do more controversial reddit posts lead to more positive attention?)

UPDATE: Read UPDATE under create_score_analysis with regards to (2)
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

NUM_HOURS = 24

# Plot directly: gather all scores (per hour), average, plot. 
# UPDATE: Found results to not be particularly interesting (
# little/no correlation: left here only for reference)
def create_score_analysis(data):
    create_score_data = data.drop(['controversiality', 'subreddit'], axis=1)
    hour_score_means = [create_score_data.loc[create_score_data['hour'] == \
        hour]['score'].mean() for hour in range(NUM_HOURS)]

    xArray = range(NUM_HOURS)

    N = len(xArray)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.5    # the width of the bars
    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, hour_score_means, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Reddit Score")
    ax.set_title("Reddit Score vs. Time Created")
    ax.set_xticks(ind + width/2)

    ax.set_xticklabels(xArray)
    plt.savefig("Score_Create.png")
    plt.close()

# Similar, except just directly plot
def controv_score_analysis(data):
    controv_score_data = data.drop(['hour', 'subreddit'], axis=1)
    score = controv_score_data['score'].values
    controv_score_data = controv_score_data.sort(['controversiality'], ascending=True)
    controversiality = controv_score_data['controversiality'].values
    plt.plot(controversiality, score)
    plt.xlabel("Controversy")
    plt.ylabel("Score")
    plt.title("Controversy vs. Score")

    plt.savefig("Controv_Score.png")
    plt.close()

def main():
    sql_conn = sqlite3.connect('../input/database.sqlite')
    sql_cmd = "Select created_utc, score, controversiality, subreddit "\
        "From May2015 ORDER BY Random() LIMIT 100000"

    data = pd.read_sql(sql_cmd, sql_conn)

    START_TIME = -8
    END_TIME = -6
    data['hour'] = data['created_utc'].map(lambda x: int(datetime.datetime.fromtimestamp(
        x.item()).strftime('%Y-%m-%d %H:%M:%S')[START_TIME:END_TIME]))
    data.drop(['created_utc'], axis=1)

    create_score_analysis(data)

if __name__ == "__main__":
    main()
    print("Finished analyzing")