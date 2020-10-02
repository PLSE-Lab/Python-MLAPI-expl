import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context("poster")
sns.set_style("white")
conn = sql.connect('../input/database.sqlite')

query = "SELECT * FROM May2015 WHERE subreddit = 'AskReddit' AND controversiality > 0;"

df = pd.read_sql(query, conn)

df["created_utc"] = pd.to_datetime(df["created_utc"], unit='s')
df["retrieved_on"] = pd.to_datetime(df["retrieved_on"], unit='s')

print(df.columns)

print(df[['created_utc', 'ups', 'subreddit_id', 'link_id', 'name', 'score_hidden',
       'author_flair_css_class', 'subreddit', 'id',
       'removal_reason', 'gilded', 'downs', 'archived', 'author', 'score',
       'retrieved_on', 'body', 'distinguished', 'edited', 'controversiality',
       'parent_id']].head())
       
df[['ups', 'downs', 'score', 'score_hidden', 'controversiality']].hist(log=True)
plt.savefig("score_histograms.svg", bbox_inches='tight')
plt.savefig("score_histograms.png", bbox_inches='tight')