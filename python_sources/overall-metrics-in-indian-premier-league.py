#!/usr/bin/env python
# coding: utf-8

# **Data Analysis With Indian Premier League **
# ---------------------------------------------
# 
# **Following is the SQL Query to retrieve some of the metrics in Indian Premier League ( 2008 - 2016).**
# 
#  - Matches
#  - Extra Runs
#  - Batsman Runs ( Excluding Extras )
#  - Wickets
#  - Sixes
#  - Fours
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
from subprocess import check_output

conn = sqlite3.connect('../input/database.sqlite')
exectue = conn.cursor()
metric = pd.read_sql(
                    """
                   SELECT 'Matches' As Dimension , COUNT(*) As 'Measure'
	                    FROM Match
                   UNION ALL
                   
                   SELECT 'Extra_Runs' As Dimension , SUM(Extra_Runs.Extra_Runs) As 'Measure'
	                    FROM Extra_Runs
                   UNION ALL
                   
                   SELECT 'Batsman_runs' As Dimension , SUM(B.Runs_Scored) As 'Value'
                    	FROM Batsman_Scored B
                   UNION ALL
                   
                   SELECT 'Wickets' As Dimension , COUNT(*) As 'Measure'
		               FROM Wicket_Taken
                   UNION ALL

                    SELECT 'Sixes' As Dimension , COUNT(*) As 'Measure'
		                FROM Batsman_Scored
		                    WHERE Batsman_Scored.Runs_Scored = 6
                    UNION ALL
                    SELECT 'Fours' As Dimension , COUNT(*) As 'Measure'
		                FROM Batsman_Scored
		                    WHERE Batsman_Scored.Runs_Scored = 4                                 
                    """, con = conn
       )
metric


# **Players with Most Sixes's in IPL ( 2008 - 2016 )**

# In[ ]:


sixes = pd.read_sql("""
                      SELECT C.Player_Name ,COUNT(*) As 'Sixes'
	                        FROM Ball_by_Ball A
		                        INNER JOIN Batsman_Scored B
			                        ON A.Match_Id || A.Over_Id || A.Ball_Id || A.Innings_No
						                    = B.Match_Id || B.Over_Id || B.Ball_Id || B.Innings_No					
		                        INNER JOIN Player C
			                         ON A.Striker = C.Player_Id
		                        INNER JOIN Match D
			                         ON A.Match_Id = D.Match_Id
		                        INNER JOIN Venue E
			                          ON D.Venue_Id = E.Venue_Id
		                    WHERE B.Runs_Scored = 6
			            GROUP BY C.Player_Name
			            ORDER BY Sixes DESC
                       """, con = conn
                   )
sixes

