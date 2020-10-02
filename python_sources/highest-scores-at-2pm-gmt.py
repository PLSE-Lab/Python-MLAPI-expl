#!/usr/bin/python

#This script analyzes the mean score over each hour of the day. Controversial and 
#non controversial comments are analyzed seperately. All times are UTC/GMT

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import datetime

print ("Connecting database ...\n")
sql_conn = sqlite3.connect('../input/database.sqlite')

print ("Loading non-controversial comments ...\n")
df = pd.read_sql("SELECT created_utc, score FROM May2015 WHERE controversiality == 0 LIMIT 8600000", sql_conn) 

# Limit is chosen to include non-controversial posts over 5 days

print ("Analyzing comments\n")

df['date'] = pd.to_datetime(df['created_utc'], unit='s')
df['day'] = pd.DatetimeIndex(df['date']).day 
df['hour'] = pd.DatetimeIndex(df['date']).hour 

df1 = df[df.day < 6]

df2 = df1.groupby('hour')

df3 = df2.mean()['score']
df3err = df3.std()

print ("Loading controversial comments ...\n")
df = pd.read_sql("SELECT created_utc, score FROM May2015 WHERE controversiality == 1 LIMIT 215000", sql_conn)

# Limit is chosen to include controversial posts over 5 days

print ("Analyzing comments\n")
df['date'] = pd.to_datetime(df['created_utc'], unit='s')
df['day'] = pd.DatetimeIndex(df['date']).day 
df['hour'] = pd.DatetimeIndex(df['date']).hour

df1 = df[df.day < 6]

df2 = df1.groupby('hour')

df4 = df2.mean()['score']
df4err = df4.std()

print ("Generating plots ... \n")

#Generating plot for non controversial scores

plt.figure(figsize=(15,5))
ax=plt.gca(); ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') ; ax.yaxis.set_ticks_position('left')
plt.title('Non Controversial Score', fontsize=18, fontweight='bold')
df3.plot(kind='bar', colormap='jet', yerr=df3err, ecolor ='k') 
plt.xlabel('Hour [UTC/GMT]', fontsize=15, labelpad = 5);  plt.ylabel('Average score', fontsize=15, labelpad = 25) 
plt.grid('off')

plt.savefig("scorebyhour1.png")

# Generating plot for controversial scores

plt.figure(figsize=(15,5))
ax=plt.gca(); ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom') ; ax.yaxis.set_ticks_position('left')
plt.title('Controversial Score', fontsize=18, fontweight='bold')
df4.plot(kind='bar', color='r', yerr=df4err, ecolor = 'k')
plt.xlabel('Hour [UTC/GMT]', fontsize=15, labelpad = 5) ; plt.ylabel('Average score', fontsize=15, labelpad = 25) 
plt.grid('off')

plt.savefig("scorebyhour2.png")

print ("Done ! \n")
