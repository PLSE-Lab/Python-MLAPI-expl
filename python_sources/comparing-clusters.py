from __future__ import division

print(__doc__)

import time

from subprocess import check_output

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

from sklearn import cluster, datasets

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler

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

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])

colors = np.hstack([colors] * 20)

clustering_names = [

    'AffinityPropagation', 'MeanShift',

    'Spectral', 'Agglomerative', 'Birch']



plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))

plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,

                    hspace=.01)



plot_num = 1


ss = df_interest.groupby(by=["symbol"])["change"].std()

ss = (ss-ss.mean())/ss.std()



pcs = df_interest.groupby(by='symbol').apply(lambda grp: grp[grp['change'] > 0]['change'].count() / grp['change'].size)

pcs = (pcs-pcs.mean())/pcs.std()



avgv = df_interest.groupby(by=['symbol'])['volume'].mean()/10000000

avgv = (avgv-avgv.mean())/avgv.std()



newdf = pd.concat([ss, pcs, avgv], axis=1).reset_index()

newdf.columns = ['symbol', 'std', 'prop_pos_day_change', "avg_volume"]





# estimate bandwidth for mean shift

bandwidth = cluster.estimate_bandwidth(newdf[['std','prop_pos_day_change']], quantile=0.3)



# connectivity matrix for structured Ward

connectivity = kneighbors_graph(newdf[['std','prop_pos_day_change']], n_neighbors=10, include_self=False)

# make connectivity symmetric

connectivity = 0.5 * (connectivity + connectivity.T)



# create clustering estimators

ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)



spectral = cluster.SpectralClustering(n_clusters=4,

                                      eigen_solver='arpack',

                                      affinity="nearest_neighbors")



affinity_propagation = cluster.AffinityPropagation(damping=.9,

                                                   preference=-200)



average_linkage = cluster.AgglomerativeClustering(

    linkage="average", affinity="cityblock", n_clusters=4,

    connectivity=connectivity)



birch = cluster.Birch(n_clusters=2)

clustering_algorithms = [

    affinity_propagation, ms, spectral, average_linkage,  birch]



for name, algorithm in zip(clustering_names, clustering_algorithms):

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(newdf[['std','prop_pos_day_change']])

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(newdf[['std','prop_pos_day_change']])



    # plot

    plt.title(name, size=18)

    plt.subplot(4, len(clustering_algorithms), plot_num)



    plt.scatter(newdf.iloc[:, 1], newdf.iloc[:, 2], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
    
    
    
    
    
    

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plot_num += 1