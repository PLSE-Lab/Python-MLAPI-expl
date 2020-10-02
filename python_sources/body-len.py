import sqlite3
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

sql_conn = sqlite3.connect('../input/database.sqlite')

data = sql_conn.execute('SELECT MetadataTo, MetadataFrom, ExtractedBodyText FROM Emails WHERE MetadataTo LIKE "H"')

recievedBodyLen = []

for email in data:
    recievedBodyLen.append(len(email[2]))
    
fig1 = plt.figure()
n, bins, patches = plt.hist(recievedBodyLen, np.logspace(0, 5, 50), facecolor='green', alpha=0.75)
plt.gca().set_xscale("log")
plt.xlabel('Characters in body')
plt.ylabel('Count')
plt.title('Length of Messages Clinton received')
plt.savefig("1.RecievedMessages.png")

#####################

data = sql_conn.execute('SELECT MetadataTo, MetadataFrom, ExtractedBodyText FROM Emails WHERE MetadataFrom LIKE "H"')

sentBodyLen = []

for email in data:
    sentBodyLen.append(len(email[2]))

fig2 = plt.figure()    
n, bins, patches = plt.hist(sentBodyLen, np.logspace(0, 5, 50), facecolor='green', alpha=0.75)
plt.gca().set_xscale("log")
plt.xlabel('Characters in body')
plt.ylabel('Count')
plt.title('Length of Messages Clinton sent')
plt.savefig("2.SentMessages.png")


