"""
Author: Yash Patel
Name: Reddit_Correlations.py
Description: Look into the correlations in Reddit database, primarily looking to resolve the 
following question of interest:
score vs created_utc (Is there best time of day to release a reddit for positive attention?)
controversiality vs score (Do more controversial reddit posts lead to more positive attention?)
created_utc vs. subreddit (Which subreddits most active when?)
"""

import sqlite3
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

NUM_HOURS = 24
CUTOFF_OCCURRENCES = 5

# Gather to be in the form {subreddit: [1, 2,...]} with array as # in that hour. Graph
def create_subreddit_analysis(data):
    create_subreddit_data = data.drop(['score', 'controversiality'], axis=1)
    subreddits = data['subreddit'].unique()
    cutoffSubreddits = []
    reddit_hours = {}

    for reddit in subreddits:
        reddit_hours_lookup = create_subreddit_data.loc[
            create_subreddit_data['subreddit'] == reddit]
        hour_counts = [len(reddit_hours_lookup.loc[reddit_hours_lookup['hour'] == hour].values) \
            for hour in range(NUM_HOURS)]
        if sum(hour_counts) > CUTOFF_OCCURRENCES:
            reddit_hours[reddit] = hour_counts
            cutoffSubreddits.append(reddit)

    NUM_PLOTS = 3
    numPlots = 0
    toPlotCounter = 0

    print("Graphing output...")
    xArray = range(NUM_HOURS)
    for reddit in cutoffSubreddits:
        yArray = reddit_hours[reddit]
        plt.plot(yArray, label=reddit)
        toPlotCounter += 1
        if toPlotCounter % NUM_PLOTS == 0:
            numPlots += 1
            plt.legend(bbox_to_anchor=(0., 1.0, 1., .10), loc=2,
               ncol=2, mode="expand", borderaxespad=0.)

            plt.xlabel("Hour")
            plt.ylabel("Occurrences")

            plt.savefig('Subreddit_Time{}.png'.format(numPlots))
            plt.close()

def main():
    sql_conn = sqlite3.connect('../input/database.sqlite')
    sql_cmd = "Select created_utc, score, controversiality, subreddit "\
        "From May2015 ORDER BY Random() LIMIT 5000"

    data = pd.read_sql(sql_cmd, sql_conn)

    START_TIME = -8
    END_TIME = -6
    data['hour'] = data['created_utc'].map(lambda x: int(datetime.datetime.fromtimestamp(
        x.item()).strftime('%Y-%m-%d %H:%M:%S')[START_TIME:END_TIME]))
    data.drop(['created_utc'], axis=1)

    create_subreddit_analysis(data)

if __name__ == "__main__":
    main()
    print("Finished analyzing")