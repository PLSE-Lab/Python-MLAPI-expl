from __future__ import division

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from sklearn import metrics

import os

print(os.listdir("../input"))



rootPath = "../input/"

file_names = [rootPath + i for i in check_output(["ls", "../input"]).decode("utf8").split("\n")[:-1]]

data_frames = [pd.read_csv(i) for i in file_names]

mainFrame = data_frames[1]

mainFrame["change"] = mainFrame["close"] - mainFrame["open"]

df_interest = mainFrame[["symbol", "date", "open", "close", "change", "volume"]]

df_interest["date"] = pd.to_datetime(df_interest["date"])

df_interest.head()

symbols = df_interest["symbol"].unique().tolist()

full_count = len(symbols)



oppFrame = df_interest.pivot(index = 'date', columns = 'symbol', values = 'close')

oppFrame = oppFrame.dropna(axis=1)

opp_symbols = oppFrame.columns

part_count = len(oppFrame.columns) 

'''

for u in opp_symbols[:10]:

    plt.plot(oppFrame.index.tolist(), oppFrame[u].values.tolist())

plt.legend(symbols, loc='upper left')

plt.show()

'''




ss = df_interest.groupby(by=["symbol"])["change"].std()

ss = (ss-ss.mean())/ss.std()



pcs = df_interest.groupby(by='symbol').apply(lambda grp: grp[grp['change'] > 0]['change'].count() / grp['change'].size)

pcs = (pcs-pcs.mean())/pcs.std()



avgv = df_interest.groupby(by=['symbol'])['volume'].mean()/10000000

avgv = (avgv-avgv.mean())/avgv.std()



newdf = pd.concat([ss, pcs, avgv], axis=1).reset_index()

newdf.columns = ['symbol', 'std', 'prop_pos_day_change', "avg_volume"]

newdf.head()



for i in newdf['symbol'].tolist():

    x = newdf[newdf['symbol'] == i]['std']

    y = newdf[newdf['symbol'] == i]['prop_pos_day_change']

    plt.scatter(x,y)

plt.legend(newdf['symbol'].tolist(),

           bbox_to_anchor=(1.05, 1),

           loc=2,

           borderaxespad=0.,

          ncol=10)



plt.title(r'Stock Change and Spread', fontsize=32)

plt.xlabel('Daily Positive Change (%)', fontsize=22)

plt.ylabel('Total Standard Devation', fontsize=22)

plt.figure(figsize=(10,100))

plt.show()



df1 = newdf.iloc[0:10,:]

df2 = newdf.iloc[100:110,:]

df3 = newdf.iloc[200:210,:]

df4 = newdf.iloc[300:310,:]

df5 = newdf.iloc[400:410,:]

testdf = pd.concat([df1,df2,df3,df4,df5])

for i in range(0,50):

    testdf.index.values[i] = i



from sklearn.cluster import KMeans

kmdf = testdf

met={}

# Visualize K = {3..9}

kValues = [i for i in range(3,10)]

for k in kValues:

    kmeans = KMeans(n_clusters=k, random_state=0).fit(kmdf[['std','prop_pos_day_change']].as_matrix())

    kmdf[str(k)] = kmeans.labels_



kmdf = pd.melt(kmdf, 

                id_vars=["symbol", 'std', 'prop_pos_day_change'],

                var_name="k", 

                value_name="values",

                value_vars=list(kmdf.columns[-7:]))



kmdf.head()



g = sns.FacetGrid(kmdf, col="k", hue="values", col_wrap=4, palette='Set2')

g = g.map(plt.scatter, "std", "prop_pos_day_change")

g.set(xlabel="Closing Deviation")

g.set(ylabel="'Daily Positive Change (%)")

g.fig.suptitle("Stock Cluster Analysis", size=28)

g.fig.subplots_adjust(top=.8)

plt.subplots_adjust(hspace=1.2, wspace=0.4)

g.add_legend()

g._legend.set_title("Cluster")

#handles = g._legend_data.values()

#labels = g._legend_data.keys()

#g.fig.legend(handles=handles, labels=labels, loc='lower right', ncol=3)

met={}

for i in range(3,10):

    met[str(i)] = metrics.silhouette_score(kmdf.loc[kmdf['k']==str(i)][['std','prop_pos_day_change']], kmdf.loc[kmdf['k']==str(i)]['values'], metric='euclidean')



metdf = pd.Series(met)






















