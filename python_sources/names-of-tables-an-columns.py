# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')


# You can read a CSV file like this
scorecard = pd.read_csv("../input/Scorecard.csv")

# It's yours to take from here!

scorecard.head()