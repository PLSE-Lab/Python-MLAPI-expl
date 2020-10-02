#!/usr/bin/env python
# coding: utf-8

# # Just checking this out
# This is a really cool site.  Super easy to get up and running.  I'm impressed.

# In[ ]:


import sqlite3
import matplotlib.pyplot as plt

conn = sqlite3.connect('../input/database.sqlite')

rows = conn.execute("select weight from player")    .fetchall()
weights = list(map(lambda row: row[0], rows))

plt.figure(figsize=(11,8.5))
values, bins, patches = plt.hist(weights, bins=10, rwidth=0.7)

plt.xticks(bins)
plt.title("Player Weight histogram")
plt.xlabel("Weight (lbs)")
plt.ylabel("Player count")

plt.show()

