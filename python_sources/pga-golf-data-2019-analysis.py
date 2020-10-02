#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


f = '/kaggle/input/pga-tour-20102018-data/2019_data.csv'
data = pd.read_csv(f)
data.head()


# In[ ]:


#Stats we are interested in

s1 = 'All-Around Ranking - (TOTAL)'
s2 = 'Average Approach Shot Distance - (AVG)'
s3 = 'Club Head Speed - (AVG.)'
s4 = 'Driving Distance - (AVG.)'
s5 = 'FedExCup Season Points - (POINTS)'
s6 = 'GIR Percentage from Fairway - (%)'
s7 = 'GIR Percentage from Other than Fairway - (%)'
s8 = 'Good Drive Percentage - (%)'
s9 = 'Greens in Regulation Percentage - (%)'
s10 = 'Hit Fairway Percentage - (%)'
s11 = 'Overall Putting Average - (AVG)'
s12 = 'Par 3 Scoring Average - (AVG)'
s13 = 'Par 4 Scoring Average - (AVG)'
s14 = 'Par 5 Scoring Average - (AVG)'
s15 = "Putting from - > 10' - (% MADE)"
s16 = "Putting from 4-8' - (% MADE)"
s17 = 'Putts Per Round - (AVG)'
s18 = 'Sand Save Percentage - (%)'
s19 = 'Scrambling - (%)'
s20 = 'Spin Rate - (AVG.)'

stats = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,
        s15,s16,s17,s18,s19,s20]
print(stats)


# Clean up and prep the data:

# In[ ]:


#filter the dataset down to the 20 stats
df = data.loc[data['Variable'].isin(stats)]
df.head()


# In[ ]:


#check for nulls
df.isna().sum()


# In[ ]:


#data.dropna(inplace=True)
#data.isna().sum()
df.info()


# In[ ]:


#convert the 'Value' data to numeric values
df["Value"] = pd.to_numeric(df["Value"],errors='coerce')


# In[ ]:


pivot = pd.pivot_table(data = df, index = 'Player Name', columns = 'Variable', values = 'Value')
pivot.head()


# **Overall Correlation**
# 
# Using FedEx Cup points won as a measure of success.

# In[ ]:


pivot.corr()


# This shows the same story I highlighted with my analysis of 2010 vs 2018: the game is becoming more dominated by the long game. In order of correlation:
# 
# 1. Greens in Regulation: 0.44
# 2. Driving Distance: 0.43
# 3. Scrambling: 0.33
# 4. Putting from outside 10': 0.15
# 5. Sand Saves: 0.10
# 
# And Par 4s and 5s dominate the game:
# 
# 1. Par 4: 0.46
# 2. Par 5: 0.43
# 3. Par 3: 0.21
# 
# **Data Plots**

# In[ ]:


sns.relplot(x=s3, y=s4,size=s5, data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s4, y=s10,size=s5, data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s4, y=s5,hue=s9, data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s4, y=s13,size=s5, data=pivot)
sns.relplot(x=s4, y=s14,size=s5, data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s12, y=s5,size=s4, data=pivot)
sns.relplot(x=s13, y=s5, size=s4,data=pivot)
sns.relplot(x=s14, y=s5, size=s4,data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s9, y=s5,size=s4, data=pivot)
sns.relplot(x=s6, y=s5,size=s4, data=pivot)
sns.relplot(x=s7, y=s5,size=s4, data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s19, y=s5,hue=s16,data=pivot)
sns.relplot(x=s18, y=s5,hue=s16,data=pivot)
plt.show()


# In[ ]:


sns.relplot(x=s15, y=s5,hue=s9,size=s4,data=pivot)
sns.relplot(x=s16, y=s5,hue=s9,size=s4,data=pivot)
sns.relplot(x=s17, y=s5,hue=s9,size=s4,data=pivot)
plt.show()


# **Conclusion**
# 
# Golf is very much becoming dominated by the long game i.e. driving distances. The overall average driving distance this year is around 295 (a yard less than last year).
# 
# Yes, the short game is important but you can't get to PGA Pro level without a good short game. As a result, the key differentiators are the more technical parts of the game: driving the ball far and accurately, and iron play. A good, long drive or a great iron shot to get to the green in regulation more often than your competitors is the key. Because once on the green there is not much to differentiate between the players.
# 
# "Drive for show, putt for dough"? More like "Get to the green in regulation for dough."
