import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



sql_conn = sqlite3.connect('../input/database.sqlite')

subreddits = sql_conn.execute("Select subreddit, body, max(score) From May2015 group by subreddit order by score DESC")
lista = subreddits.fetchmany(100)
for x in lista:
    try:
        print(">>Subreddit: " + x[0] + " -> score:" + str(x[2]) + "<<\n" + x[1] + "\n_____________________________________________________________\n")
    except:
        continue