import sqlite3 as sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def highestScore(data):
    # print the name and subreddit of the highest score post
    highest = data.loc[data['score'].idxmax()]
    print(highest)
    print(data)
    
def main():
    theConn = sql.connect('../input/database.sqlite')
    sql_cmd = "Select name,subreddit, score From May2015 ORDER BY Random() LIMIT 50"

    # read data frame
    data = pd.read_sql(sql_cmd, theConn)
  
    highestScore(data)


if __name__ == "__main__":
    main()
    




