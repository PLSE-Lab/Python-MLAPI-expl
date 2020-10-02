# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT MetadataDocumentClass, count(DISTINCT MetadataDocumentClass) as Count
FROM Emails""", con)
print(sample)
