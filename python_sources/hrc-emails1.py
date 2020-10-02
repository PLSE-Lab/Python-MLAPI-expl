# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3
import textwrap
# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT e.RawText Text
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id
Where p.Name like 'Jake Sullivan'
LIMIT 1""", con)
readout = str(sample)
# You can read a CSV file like this
persons = pd.read_csv("../input/Persons.csv")

print(textwrap.fill(readout, 30))


# It's yours to take from here!

