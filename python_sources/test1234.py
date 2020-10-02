import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

allGolds = sql_conn.execute("SELECT body, gilded, subreddit FROM May2015 WHERE gilded > 0")

print("type is")
type(allGolds)