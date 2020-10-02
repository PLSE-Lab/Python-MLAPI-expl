"""Script to examine the income distribution of welders and philosophers.

In the 4th Republican Primary debate held on 10 November 2015, Sen. Marco Rubio
[stated](https://www.youtube.com/watch?v=fGAfc40uTog) that welders make more money
than philosophers, and said that America needs more welders and fewer philosophers.

There have been a number of attempts to fact check this statement, with varying results
depending on how the terms "welder" and "philosopher" are defined, and the source of the
data. I thought it would be interesting to investigate this claim using the American
Community Survey data.

While the data does list professions, "philosopher" isn't one of the possibilites.
Instead, I'll take everyone who lists philosophy as their field of degree. Furthermore, I'll
only consider people with a Bachelor's degree, since the comparison Senator Rubio was
making was directed at contrasting vocational training with getting a 4 year degree, and
people with postgraduate degrees tend to earn more, skewing the results.

The script prints the median salary for both welders and philosophers (as defined above),
and shows that while philosophers are a little higher, the two are very close. Furthermore,
looking at the whole distributions instead of just the medians tells a more nuanced story.

First, in favor of philosophers: there's more potential upside.  That is, your odds of
attaining a large annual income (in excess of $100k/year, say) are much better if you're a
philosopher than if you're a welder. This shifts both the mean and median upward. On the
other hand, the wider, flatter distribution for philosophers includes a little
more downside, as well. So, welders seem less likely to strike it rich than philosophers, but also less likely to starve.

More interesting is the question of most likely salary, or the mode of the salary
distribution. In spite of the fact that the population of philosophers has a higher median
and mean than that of the welders, the welders have a higher mode, indicating that the most
likely salary is higher for welders than philosophers.
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_style('white')

# read the data; deal with memory limits.
# we will need four columns:
cols = ['FOD1P', 'OCCP', 'PWGTP', 'PINCP']

dfa = pd.DataFrame(columns=cols)
chunka = pd.read_csv('../input/pums//ss13pusa.csv', chunksize=1000, usecols=cols)

while True:
    try:
        sub_df = chunka.get_chunk()
        #sub_df = sub_df.dropna()
        dfa = pd.concat([sub_df, dfa])
    except:
        break
dfb = pd.DataFrame(columns=cols)
chunkb = pd.read_csv('../input/pums//ss13pusb.csv', chunksize=1000, usecols=cols)
while True:
    try:
        sub_df = chunkb.get_chunk()
        #sub_df = sub_df.dropna()
        dfb = pd.concat([sub_df, dfb])
    except:
        break
    
data = pd.concat([dfa, dfb])

# we class as philosophers anyone who majored in philosophy:
philosophers = data.groupby('FOD1P').get_group(4801)

# and welders are people who are currently working as welders:
welders = data.groupby('OCCP').get_group(8140)

# we need to exercise a little caution in dealing with these data, because
# they're weighted.  Calculating the actual median is more complicated than
# just getting the median of the data.
def weighted_median_income(group):
    subset = pd.DataFrame(group, columns=['PINCP', 'PWGTP'])
    subset.sort_values('PINCP', inplace=True)
    subset['cumweight'] = subset['PWGTP'].cumsum()
    midnum = subset['PWGTP'].sum()/2
    return subset[subset['cumweight']<=midnum]['PINCP'].max()


# look at medians:
print("Median income for welders: {0:f}\n".format(weighted_median_income(welders)))
print("Median income for philosophers: {0:f}\n".format(weighted_median_income(philosophers)))

# plot distributions
philhistdata = philosophers['PINCP'].values
philweights = philosophers['PWGTP'].values
philhist, philbin_edges = np.histogram(philhistdata, weights=philweights, bins=50, normed=True)

weldhistdata = welders['PINCP'].values
weldweights = welders['PWGTP'].values
weldhist, weldbin_edges = np.histogram(weldhistdata, weights=weldweights, bins=50, normed=True)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax = plt.bar(weldbin_edges[:-1], weldhist, width=weldbin_edges[1]-weldbin_edges[0], color='green', alpha=0.2, label="Welders")
ax = plt.bar(philbin_edges[:-1], philhist, width=philbin_edges[1]-philbin_edges[0], color='blue', alpha=0.2, label="Philosophers")

plt.xlim(min(min(weldbin_edges), min(philbin_edges)), 
         300000)
lab = plt.xlabel("Income")
title = plt.title("Income distribution for welders and philosophers (truncated to $300k/year)")
leg = plt.legend()
plt.savefig('welders_and_philosophers.png')
