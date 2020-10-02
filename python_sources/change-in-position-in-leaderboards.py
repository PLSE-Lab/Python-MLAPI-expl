import sqlite3
import math
import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
#plt.style.use('bmh')
c = sqlite3.connect('../input/database.sqlite')

p = pd.read_sql("select distinct t.id, t.ranking, count(s.id) as co\
 from competitions c, teams t, submissions s \
 where t.competitionid = c.id and c.id = 2496 and s.teamid = t.id \
 group by t.id order by t.ranking ",c)

N = len(p) # N = 1543
 #2496
x = list(range(N))
y = list(range(N))
for i in range(N):
	q = pd.read_sql("select distinct t.id, t.ranking, min(s.publicscore) as publicscore, min(s.privatescore) as privatescore \
	 from competitions c, teams t, submissions s \
	 where t.competitionid = c.id and c.id = 2496 and s.teamid = t.id and s.publicscore <> '' \
	 and t.ranking = " + str(p.Ranking[i]) + "  group by t.id",c)
	x[i] = q.publicscore[0]
	y[i] = q.privatescore[0]

a0 = sorted(list(range(len(x))), key=lambda k: x[k])
a = [x + 1 for x in a0]
b0 = sorted(list(range(len(y))), key=lambda k: y[k])
b = [x + 1 for x in b0]

plt.figure(figsize=(10, 10))
plt.xlim(0, 40)
plt.ylim(0, 25)
plt.scatter(a[0:20], b[0:20], c=p.co[0:20])
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('First 20 positions in private leaderboard')
plt.savefig('places_0.png')
plt.show()

plt.figure(figsize=(10, 10))
plt.xlim(0, 300)
plt.ylim(0, 125)
plt.scatter(a[0:100], b[0:100], c=p.co[0:100])
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('First 100 positions in private leaderboard')
plt.savefig('places_1.png')
plt.show()

plt.figure(figsize=(10, 10))
# plt.xlim(0, 300)
# plt.ylim(0, 125)
plt.scatter(a[100:400], b[100:400], c=p.co[100:400])
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('Positions 101-400 in private leaderboard')
plt.savefig('places_2.png')
plt.show()

plt.figure(figsize=(10, 10))
# plt.xlim(0, 300)
# plt.ylim(0, 125)
plt.scatter(a[400:800], b[400:800], c=p.co[400:800])
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('Positions 401-800 in private leaderboard')
plt.savefig('places_3.png')
plt.show()

plt.figure(figsize=(10, 10))
# plt.xlim(0, 300)
# plt.ylim(0, 125)
plt.scatter(a[800:1544], b[800:1544], c=p.co[800:1544])
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('Positions 801-1053 in private leaderboard')
plt.savefig('places_4.png')
plt.show()

plt.figure(figsize=(10, 10))
plt.xlim(0, 1454)
plt.ylim(0, 1454)
plt.scatter(a, b, c=p.co)
cbar = plt.colorbar()
cbar.ax.set_xlabel('Number of submissions')
plt.xlabel('Public leaderboard position')
plt.ylabel('Priate leaderboard position')
plt.title('All participants')
plt.savefig('places_all.png')
plt.show()