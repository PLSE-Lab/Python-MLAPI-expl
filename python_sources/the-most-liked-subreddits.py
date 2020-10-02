import sqlite3
cursor = sqlite3.connect ('../input/database.sqlite').cursor()
for data in cursor:
    print (cursor[data])