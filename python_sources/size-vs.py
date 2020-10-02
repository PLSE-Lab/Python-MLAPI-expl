# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT c.Id, c.DataSizeBytes / count(t.Id), count(t.Id)
FROM Competitions as c
INNER JOIN
Teams as t
ON c.Id = t.CompetitionId
GROUP BY c.Id""", con)

print(sample)


# It's yours to take from here!