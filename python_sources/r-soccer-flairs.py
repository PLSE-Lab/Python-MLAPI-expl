import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

sql_cmd = "SELECT author_flair_text FROM May2015 WHERE subreddit = 'soccer' AND author_flair_text IS NOT NULL " #

data = pd.read_sql(sql_cmd, sql_conn)

def decoder(flair):
    return flair.encode('ascii', 'ignore')

data['author_flair_text'] = data.author_flair_text.apply(decoder)

print(data.author_flair_text.value_counts().head(20))