# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3
import string

# You can read in the SQLite datbase like this
# import sqlite3
# con = sqlite3.connect('../input/database.sqlite')
# sample = pd.read_sql_query("""
# SELECT *
# FROM Papers
# LIMIT 10""", con)
# print(sample)

# You can read a CSV file like this
# papers = pd.read_csv("../input/Papers.csv")
# print(papers)

# It's yours to take from here!

def process_file(filename):
    h = dict()
    fp = open(filename)
    for line in fp:
        process_line(line, h)
    return h

def process_line(line, h):
    line = line.replace('-', ' ')
    
    for word in line.split():
        word = word.strip(string.punctuation + string.whitespace)
        word = word.lower()

        h[word] = h.get(word, 0) + 1

hist = process_file("../input/Papers.csv")
print(hist)