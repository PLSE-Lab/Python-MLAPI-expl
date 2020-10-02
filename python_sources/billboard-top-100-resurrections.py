#!/usr/bin/env python
# coding: utf-8

# ### Background
# 
# The inspiration for this project came from Travis Scott. 
# 
# In May 2020, Scott's Kendrick Lamar collaboration "goosebumps" was back on the US Top 50 [four years after its release.](https://twitter.com/wluna01/status/1258776888647938049)
# 
# This had me curious
# 
# I queried [a dataset](https://data.world/kcmillersean/billboard-hot-100-1958-2017/workspace/file?filename=Hot+Stuff.csv) at data.world (credit kcmillersean) to find which songs had returned to the Billboard Top 100 after at least a year off the charts:
# 
# >     SELECT *, DATE_DIFF(debut_date, last_return_date, "year") AS gap_length
# >     FROM (SELECT song, performer, MIN(weekid) AS debut_date, MAX(weekid) AS last_return_date, COUNT(*) AS num_resurrections
# >         FROM hot_stuff_2
# >         WHERE previous_week_position IS NULL
# >         GROUP BY song, performer
# >         HAVING num_resurrections > 1) AS eternal_hits
# >     WHERE DATE_DIFF(debut_date, last_return_date, "year") > 0
# >     ORDER BY gap_length DESC;
#  
# *Note* that covers aren't accounted for. For example, Glee's cover of "Don't Stop Believin'" does not count as a resurrection because the original song by Journey didn't make the Top 100, only their new version. Getting around this problem is tricky, since grouping track names without regard for the artist lumps generic track names together. Lil Wayne's 2013 track "Love Me" wasn't a cover of Justin Bieber's 2009 song. It wasn't a new take on Bobby Hebb's 1966 version either.
# 
# ### Ten Longest-Running Billboard Chart Legacies
# 
# | Track | Artist| First Debut on Top 100 | Last Debut on Top 100 |
# | --- | --- | --- | --- | --- |
# | White Christmas	| Bing Crosby	| 1958-12-20	| 2018-12-29	|
# | Run Rudolph Run |	Chuck Berry	| 1958-12-13 |	2019-01-05 |
# | Jingle Bell Rock	| Bobby Helms	| 1958-12-20	| 2019-12-07	| 
# | The Christmas Song (Merry Christmas To You)	| Nat King Cole	| 1960-12-10	| 2019-12-14	| 
# | Rockin' Around The Christmas Tree	| Brenda Lee	| 1960-12-10	| 2019-12-07	|
# | Space Oddity	| David Bowie	| 1973-01-27	| 2016-01-30	|
# | Bohemian Rhapsody	| Queen |	1976-01-03 | 2018-11-17	|
# | Thriller	| Michael Jackson |	1984-02-11	| 2019-11-09	|
# | Under Pressure | Queen & David Bowie	| 1981-11-07	| 2016-01-30 |
# | Little Red Corvette	| Prince	| 1983-02-26 |	2016-05-07 |

# However, not every song resurgence is the result of Christmas or a coffin. **"Stand by Me"** came back twenty-five years later because of the eponynous film. [A celtics fan](https://www.youtube.com/watch?v=mOHkRk00iI8) brought Bon Jovi's **"Livin' on a Prayer"** back twenty-six years after its initial debut. A [dance-off between patient and surgeons](https://www.youtube.com/watch?v=uPdheFjRm4E) took a Beyonce song higher on the charts than its original release.
# 
# | Song | Artist| Film or Video |
# | --- | --- | --- |
# | Stand By Me (1961) | Ben E. King | Stand By Me (1986) |
# | Do You Love Me (1962) | The Contours | Dirty Dancing (1988) |
# | Twist And Shout (1964) | The Beatles | Ferris Beuler's Day Off (1986) |
# | Unchained Melody (1965) | Righteous Brother | Ghost (1990) |
# | Livin' On A Prayer (1986) | Bon Jovi | Celtics Dance (2013)
# | My Boo (1996) | Ghost Town DJ's | Running Man Challenge (2016) |
# | Only Time (2001) | Enya | Volvo Trucks Ad (2013) |
# | Get Me Bodied (2003) | Beyonce | Deb Conan Masectomy (2013)| 

# However, all these observations are from limited information, since the query only pulls the first and last time a song debuts on the Billboard 100. I wrote a second query to return *all* the data for   the resurrected songs:
# 
#     SELECT hs.songid, hs.weekid, hs.week_position
#     FROM hot_stuff_2 hs
#     INNER JOIN 
#         (SELECT song, performer
#         FROM (SELECT song, performer, MIN(weekid) AS debut_date, MAX(weekid) AS last_return_date, 
#             COUNT(*) AS num_debuts
#             FROM hot_stuff_2
#             WHERE previous_week_position IS NULL
#             GROUP BY song, performer
#             HAVING num_debuts > 1) AS eternal_hits
#         WHERE DATE_DIFF(debut_date, last_return_date, "year") > 0) AS t
#     ON hs.song = t.song AND hs.performer = t.performer
#     ORDER BY hs.song, hs.performer, hs.weekid DESC;

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/resurrected-billboar-resurrected-billboard-top-100-hits-QueryResult.csv')

