#!/usr/bin/env python
# coding: utf-8

# Hi everybody,
# 
# There was a discussion on how the release of one kernel after the merger deadline had jeopardized the positions of teams in the leaderboard, and that they lost their positions because of this kernel, etc. Here I'm going to do some analysis on how this kernel had affected the rankings in the public leaderboard. The analysis is very primitive and I would be happy to hear more from you on how we can improve the analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pylab as pl # linear algebra + plots
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.colors as colors
import matplotlib.cm as cmx
from collections import defaultdict

T = pd.read_csv('../input/santader-2018-public-leaderboard/santander-value-prediction-challenge-publicleaderboard.csv')
T['date'] = pd.to_datetime(T.SubmissionDate)
MergerDate = '2018-08-13'


# Let's start by plotting the scores of teams over time:

# In[ ]:


pl.figure(figsize=(20,5))
pl.plot(T.date, T.Score, '.')
pl.plot([pd.to_datetime(MergerDate)+pd.DateOffset(4)]*2, [0.4, 1.6], 'k')
pl.ylim([.4, 1.6])


# There is a visible drop in scores after the release of that kernel as expected. It usually happens after the release of good public kernels. Let's continue by comparing the top 500 teams at merger deadline and at the end:

# In[ ]:


T_merger = T[T.date <= pd.to_datetime(MergerDate)].groupby('TeamName').agg({'Score':'min'})
T_merger['Rank'] = pl.argsort(pl.argsort(T_merger.Score)) + 1
top500_merger = sorted(list(T_merger[T_merger.Rank <= 500].index), key=lambda x:T_merger.loc[x, 'Rank'])

T_End = T.groupby('TeamName').agg({'Score':'min'})
T_End['Rank'] = pl.argsort(pl.argsort(T_End.Score)) + 1
top500_end = sorted(list(T_End[T_End.Rank <= 500].index), key=lambda x:T_End.loc[x, 'Rank'])

print(len(set(top500_merger).intersection(set(top500_end))))
SteadyTeams = []
for n in range(10, 500, 10):
    SteadyTeams.append(len(set(top500_merger[:n]).intersection(set(top500_end[:n]))) / n * 100)
pl.plot(range(10, 500, 10), SteadyTeams)
pl.xlabel('top n teams')
pl.ylabel('percent of steady teams');


# Only 294 (60%) of 500 top teams in merger deadline have been among final top 500 teams. 80% of the top 10 teams stayed at top 10 and around 70% of the top 100 teams were not affcted in the last week. But, the question is how much of these changes in the rankings were due to public kernels. In the first attempt, I plot the distributions of ranking changes:

# In[ ]:


def plotCI(Arr, ci=[5, 95], color='b', alpha=.4):
    X = pl.arange(Arr.shape[1])
    Y0 = list(map(lambda col: pl.percentile(col, ci[0]), Arr.T))
    Y1 = list(map(lambda col: pl.percentile(col, ci[1]), Arr.T))
    M = Arr.mean(0)
    pl.fill_between(X, Y0, Y1, color=color, alpha=alpha)
    pl.plot(X, M, color=color, lw=2)


pl.figure(figsize=(7,10))
for top500, color in zip([top500_merger, top500_end], ['b', 'r']):
    Ranks = defaultdict(lambda :[])
    for day in range(0,9):
        T_temp = T[T.date <= pd.to_datetime(MergerDate) + pd.DateOffset(day)]
        T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
        T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
        for team in top500:
            Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)
    RankArr = pl.array(list(Ranks.values()))
    # replace nans with 500, they are only a few
    RankArr[pl.where(pl.isnan(RankArr))] = 500
    plotCI(RankArr, color=color)
pl.xlabel('Days after merger deadline')
pl.ylabel('Ranks')
Ylim = pl.gca().get_ylim()
pl.plot([4, 4], Ylim, 'k')
pl.plot([2, 2], Ylim, 'k:')
pl.plot([6, 6], Ylim, 'k:')

pl.legend(['top at end', 'top at merger', 'kernel release', 'release of extra groups'], loc=1);


# The graph shows the changes in the rankings are inevitable. Even before the release of the controversial public kernel, the people in the top 500 at the merger deadline were losing more rankings and other people were merging into better rankings. But the change in the slope of these processes is noticeable after the release of the kernel. Release of extra groups did not change the slope dramatically. (It seems many people are lazy, including myself ;) )
# 
# The main point of the discussion was that **many** people who had endured a lot to be among the high rankings had lost their position because of the public kernel. A more detailed view on the divergence of the rankings in the last week can show how much it had actually affected people who are among top500 at merger deadline.

# In[ ]:


top500 = top500_merger
Ranks = defaultdict(lambda :[])
for day in range(0,9):
    T_temp = T[T.date <= pd.to_datetime(MergerDate) + pd.DateOffset(day)]
    T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
    T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
    for team in top500:
        Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)

pl.figure(figsize=(10, 12))
cNorm = colors.Normalize(vmin=0, vmax=500)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='plasma')
for i, team in enumerate(top500):
    pl.plot(Ranks[team], color=scalarMap.to_rgba(i), alpha=.6)
pl.plot([4, 4], pl.gca().get_ylim(), 'k')
cb = pl.colorbar(pl.scatter(0,0,s=.1,c=0,vmin=0,vmax=500, cmap='plasma'))
cb.ax.invert_yaxis()
pl.xlabel('Days after merger deadline')
pl.ylabel('Ranks')
pl.gca().invert_yaxis()
pl.tight_layout()


# It seems many people whose ranks were 100-500 had lost **faster** their rankings after the public kernel. Many of them could move back to the top 500 after a few days (probably after employing the kernel insights). Also interestingly, many teams could get better results right after the kernel was released. So, while many lost their positions **faster** others got better results **faster** after the kernel was released.
# 
# In the next section, I'm going to do some analysis on how the rankings were expected to move before that kernel, and compare it to final results. Of course, the analysis is a weak estimate based on the trend and not fully matching the reality, but it can shed some light on the discourse.
# 
# Here is how I do it: I make linear interpolation of rankings of each team based on the data from merger deadline to the kernel release date and extrapolate it to the final day.

# In[ ]:


ExpectedRank = {}
for team in Ranks:
    p = pl.polyfit(pl.arange(5), Ranks[team][:5], 1)
    r = pl.polyval(p, pl.arange(5, 9)).round()
    r[r<0] = 0
    ExpectedRank[team] = r

FinalRank = pl.array([Ranks[x][-1] for x in Ranks])
ExpectedRankFinal = pl.array([ExpectedRank[x][-1] for x in ExpectedRank])

pl.plot(FinalRank, ExpectedRankFinal, '.')
pl.xlabel('Final Rank')
pl.ylabel('Expected Rank')

pl.figure()
pl.hist(ExpectedRankFinal - FinalRank, 20)
pl.xlabel('jumps in the leaderboard compared to expected ranking')
pl.ylabel('#')

from scipy.stats import skew
print('skew of the distribution:', skew(ExpectedRankFinal - FinalRank))
print('number of people with ranking worse than expected:', sum((FinalRank - ExpectedRankFinal) > 0))
print('number of people with ranking better than expected:', sum((FinalRank - ExpectedRankFinal) < 0))


# These results show the number of people who got worse rankings as a result of the kernel is larger than the number of teams who could get better results. As I said it is a very primitive study and it does not take many other parameters into account. Also, the results might differ if I had calculated the interpolation based on more data points before the kernel release date. The difference in public and private leaderboard is also an important note to consider. The most important point to consider here, however, is that the effect of public kernels are not linear in time and they dampen after a few days. So the main question remains open: should Kaggle allow release of public kernels after the merger deadline? or, is the 1 week time span enough to dampen the effect of public kernels?

# Let's take a look on the effects of the previous most favorated and effective public kernels in this competition:

# In[ ]:


def getRanksOnDate(Date, offset):
    T_temp = T[T.date <= pd.to_datetime(Date) + pd.DateOffset(offset)]
    T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
    T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
    return T_temp


def plotRankingChanges(kernelDate, top=500, interval=(-5,10), kernelName=''):
    T_temp = getRanksOnDate(kernelDate, interval[0])
    topTeams = sorted(list(T_temp[T_temp.Rank <= top].index), key=lambda x:T_temp.loc[x, 'Rank'])
    Ranks = defaultdict(lambda :[])
    for day in range(interval[0], interval[1]):
        T_temp = getRanksOnDate(kernelDate, day)
        for team in topTeams:
            Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)
    
    cNorm = colors.Normalize(vmin=0, vmax=top)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='plasma')
    pl.figure(figsize=(15, 12))
    for i, team in enumerate(topTeams):
        pl.plot(range(interval[0], interval[1]), Ranks[team], color=scalarMap.to_rgba(i), alpha=.5)
    pl.plot([0, 0], pl.gca().get_ylim(), 'k')
    cb = pl.colorbar(pl.scatter(0,0,s=.1,c=0,vmin=0,vmax=top, cmap='plasma'))
    cb.ax.invert_yaxis()
    pl.xlabel('Days after kernel release')
    pl.ylabel('Ranks')
    pl.title(kernelName)
    pl.gca().invert_yaxis()
    pl.tight_layout()
    return Ranks

################################
Ranks1 = plotRankingChanges('2018-07-03', top=300, interval=(-10,20), kernelName='Pipeline Kernel, xgb + fe [LB1.39]')
Ranks2 = plotRankingChanges('2018-07-13', top=300, interval=(-10,20), kernelName='Santander_46_features')
Ranks3 = plotRankingChanges('2018-07-18', top=300, interval=(-10,20), kernelName='Leak (a collection of kernels)')
Ylim = pl.gca().get_ylim(); p=[]
p.append(pl.plot([0, 0], Ylim, 'k')[0])
p.append(pl.plot([1, 1], Ylim, 'k--')[0])
p.append(pl.plot([2, 2], Ylim, 'k:')[0])
pl.legend(p, ['Giba\'s Property', 'Breaking LB - Fresh start', 'Baseline with Lag Select Fake Rows Dropped'], loc=3)


# After the release of these kernels, the slopes of changes in the leaderboard accelerate more or less. The most effective one, of course, was the leak, and in fact, a collection of kernels contributed in it. It is interesting to note that the most effective one was Mohsin's kernel. The reason I think is it had made it easy for others to submit a leaky submission and get a good score. So, everybody took advantage of that. It's one of the reasons that many people don't like the idea of public kernels, and they prefer to share the idea not the easy to copy code.
# 
# Let's take a closer look at events after the Leak:

# In[ ]:


Ranks4 = plotRankingChanges('2018-07-19', top=500, interval=(-5,35), kernelName='Breaking LB - Fresh start')
Ylim = pl.gca().get_ylim(); p=[]
p.append(pl.plot([0, 0], Ylim, 'k')[0])
p.append(pl.plot([2, 2], Ylim, 'k--')[0])
p.append(pl.plot([6, 6], Ylim, 'k-.')[0])
p.append(pl.plot([29, 29], Ylim, 'k:')[0])
pl.legend(p, ['Mohsin\'s kernel','Best Ensemble [67]','Best Ensemble [63] + Love is the Answer II','Jiazhen to Armamut via gurchetan1000 - 0.56'], loc=3)


# Usually, publication of an effective kernel or finding the data follows by a few effective kernels that usually employ the idea in other ways or combine it with other ideas. It also happened around the time of 0.56 kernel by publishing new groups and mixing techniques. 
# 
# From the graph above it seems to me that the turmoil in the Public LB has gone back to normal around 7 days after Mohsin's kernel. Next, I try to quantify this:

# In[ ]:


RanksArr = pl.array(list(Ranks4.values()))
RankDiff = pl.diff(RanksArr, 1)
pl.plot(range(-4,35), pl.std(RankDiff, 0))
pl.plot([0,0], [2, 200], 'k')
pl.plot([29,29], [2, 200], 'k')
pl.plot([-5,35], [100, 100], 'k:')
pl.ylabel('variation (std) in Rank changes')
pl.xlabel('Days after Mohsin\'s kernel')


# So, it seems that it took 7-8 days after Mohsin's kernel release for the changes in the LB to go back to baseline. And four days before the deadline were not enough for the teams to adapt to the newly released public kernel.
# 
# I think similar results can be taken from other competitions as well, and it shows this suggestion is sound that Kaggle should not allow public release of kernels after merger deadline.
