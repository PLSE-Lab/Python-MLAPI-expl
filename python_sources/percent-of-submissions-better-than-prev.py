import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('bmh')
c = sqlite3.connect('../input/database.sqlite')

p = pd.read_sql("select t.id, t.ranking, count(distinct  s.id) as co \
 from competitions c, teams t, teammemberships m, submissions s \
 where t.competitionid = c.id and t.id = m.teamid and c.id = 2496 and s.teamid = t.id and s.privatescore <> '' \
 group by t.id order by 2 ",c)

N = len(p) # N = 1543

x = list(range(N))
for i in range(N):
	q = pd.read_sql("select distinct t.id, t.ranking, s.datesubmitted, s.publicscore \
	 from competitions c, teams t, teammemberships m, submissions s \
	 where t.competitionid = c.id and t.id = m.teamid and c.id = 2496 and s.teamid = t.id and s.publicscore <> '' \
	 and t.ranking = " + str(i + 1) + " order by 3",c)
	t = np.diff(q.PublicScore)
	x[i] = sum(j > 0 for j in t) * 1.0


z = [x[i] / p.co[i] for i in range(N)]
print(sum(0.501 > j and j > 0.499 for j in z) * 1.0 / N)
print(sum(0.334 > j and j > 0.332 for j in z) * 1.0 / N)


width = 1
plt.figure()
plt.title('Heritage Health Prize: share of submissions that have better\npublic score than the previous ones for every participant', fontsize=16)
ind = [i + 1 for i in range(N)]
plt.bar(ind, z, width)
a = list(range(0, N + 1, 123));
a = [1] + a[1:]
plt.xticks([x + width/2. for x in a], a)
plt.xlabel("About 10% of all participants have this share approximately 0.5 or 0.333. That's really strange.")
plt.plot([1, 1353], [0.5, 0.5], 'r', hold=True)
plt.plot([1, 1353], [0.333, 0.333], 'b', hold=True)
plt.savefig('good_subm_to_all.png')
plt.show()
