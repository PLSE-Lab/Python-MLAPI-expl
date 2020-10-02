
import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql_query("""
SELECT INSTNM,
       COSTT4_A AverageCostOfAttendance,
       mn_earn_wne_p6,
       Year
FROM Scorecard
WHERE INSTNM='Yale University'""", con)

#score = df[(df.FEMALE_DEBT_MDN > 5000)]
print(df)


