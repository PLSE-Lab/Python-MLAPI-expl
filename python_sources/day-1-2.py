#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


import pandas as pd
deliveries = pd.read_csv("../input/ipldata/deliveries.csv")
matches = pd.read_csv("../input/ipldata/matches.csv")
print(matches)


# In[ ]:


matches.rows


# In[ ]:



matches.columns


# In[ ]:


deliveries.rows


# In[ ]:


deliveries.columns


# In[ ]:


matches['city']


# In[ ]:


matches['city'][0:4]


# In[ ]:


DATA1.coloumns


# In[ ]:


DATA1.rows


# In[ ]:


DATA1.columns


# In[ ]:


DATASET.columns


# In[ ]:


import pandas as pd
file = pd.read_csv("../input/file20/file.csv")
print(file)


# In[ ]:


file.columns


# In[ ]:


file,sort_values(by=['roll'])


# In[ ]:


file.sort_values(by=['roll'])


# In[ ]:


file,sort_values(by=['Roll'])|


# In[ ]:


file,sort_values(by=['Roll'])


# In[ ]:


file.sort_values(by=['Roll'])


# In[ ]:


print(file[file['MARKS'].isnull()])


# In[ ]:


file.MARKS.plot(kind='hist');


# In[ ]:


import seaborn as sns
sns.scatterplot(x=file['Roll'], y=file['MARKS'])


# In[ ]:


import seaborn as sns
sns.boxplot(x=file['Roll'], y=file['MARKS'])


# In[ ]:


import pandas as pd
DOC = pd.read_csv("../input/employee/DOC.csv")


# In[ ]:


print(DOC)


# In[ ]:


DOC.age.plot(kind='hist');


# In[ ]:


DOC.sales.plot(kind='hist');


# In[ ]:


DOC.profit.plot(kind='hist');


# In[ ]:


import matplotlib.pyplot as plt
x=DOC['age']
y=DOC['sales']
z=DOC['profit']
plt.hist(x,bins=10,color='green')
plt.hist(y,bins=10,color='red')
plt.hist(z,bins=10,color='yellow')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
 
# Create legend & Show graphic
plt.legend()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = DOC['age']
bars2 = DOC['sales']
bars3 = DOC['profit']
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], [i for i in DOC['employ id']])
 
# Create legend & Show graphic
plt.legend()
plt.show()


# In[ ]:


labels = DOC['employ id']
sizes = DOC['age']
explode = (0, 0.5, 0, 0,0,0,0,0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


DOC['age'][5]=532


# In[ ]:


DOC


# In[ ]:


import seaborn as sns
sns.boxplot(x=DOC['profit'], y=DOC['age'])


# In[ ]:




