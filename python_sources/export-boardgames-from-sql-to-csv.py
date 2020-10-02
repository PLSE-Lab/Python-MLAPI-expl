# Export data as a CSV for easy analysis externally

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


#Load in the data and tell me something about it
import sqlite3
conn = sqlite3.connect('../input/database.sqlite')
query = "SELECT * FROM BoardGames"
df_boardgame_full = pd.read_sql_query(query,conn)

df_boardgame_full.shape


df_boardgame_full = df_boardgame_full.loc[(df_boardgame_full["stats.usersrated"] > 30 ) & (df_boardgame_full["stats.owned"] >= 25)]
print(df_boardgame_full.shape)
print(df_boardgame_full.head())
print()

print(list(df_boardgame_full.columns))

df_boardgame_full.to_csv("boardGamesRatingFullBig.csv.gz",index=False,compression="gzip")