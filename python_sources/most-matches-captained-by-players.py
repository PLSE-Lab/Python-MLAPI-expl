#!/usr/bin/env python
# coding: utf-8

# Most matches captained by players in Indian Premier League ( 2008 -2016 )
# ---------------------------------------------------------------

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt

from subprocess import check_output
conn = sqlite3.connect('../input/database.sqlite')
exectue = conn.cursor()
matches_captained = pd.read_sql(
                    """
                    SELECT C.Player_Name , COUNT(*) As 'Matches_captained'
	                        FROM Player_Match A 
	                                INNER JOIN Rolee B
	                                    ON A.Role_Id = B.Role_Id
	                                INNER JOIN Player C
	                                    ON A.Player_Id = C.Player_Id
	                        WHERE A.Role_Id = 1 OR A.Role_Id = 4
	                GROUP BY C.Player_Name
	                ORDER BY Matches_captained DESC
                    LIMIT 5;
                  """, con = conn
       )

labels = matches_captained["Player_Name"]
y = matches_captained["Matches_captained"]

x = [1,2,3,4,5]


plt.bar(x,y,align='center')

plt.xticks(x,labels)


plt.show()

