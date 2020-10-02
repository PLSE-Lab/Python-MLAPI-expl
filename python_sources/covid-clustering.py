# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import json
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pylab as plt
from matplotlib import cm
from sklearn.metrics import silhouette_score, silhouette_samples
from joblib import dump
from biobert_embedding import pooling
from helper_functions import upload_file
from sklearn.metrics import silhouette_score, calinski_harabasz_score
sns.set_style("ticks")


def evaluate_metric_score(df_embedding, df_clusters, metric="silhouette"):

    df_score = pd.DataFrame(index=df_clusters.columns.tolist(),
                            columns=["metric"])

    for v in df_clusters.columns.tolist():
        if metric == "silhouette":

            silhouette_avg = silhouette_score(df_embedding, df_clusters[v])

        elif metric == "calinski":
            silhouette_avg = calinski_harabasz_score(df_embedding.values,
                                                     df_clusters[v].values)

        df_score.at[v] = silhouette_avg

        print(
            " n_clusters =",
            v,
            ", average metric score =",
            silhouette_avg,
        )
    return df_score

def graph_silhouette(
    sample_silhouette_values, cluster_labels, n_clusters, silhouette_avg, ax
):
    """
    draw the silhouette graph plots

    Parameters:
    -----------
    sample_silhouette_values: np.array
        sample_silhouette_values obtained from
        sklearn.metrics.silhouette_samples

    cluster_labels: np.array
        cluster label for each sample point

    n_clusters: int
        number of clusters

    silhoeutte_avg: float
        average silhouette value

    Return:
    -------
    ax: matplotlib axis
        axis of the plot
    """
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.get_cmap("Spectral")(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return ax


def plot_silhouette_graph(
    df_pooled_embedding, df_clusters, n_clusters, fig, axes
):

    for k, ax in enumerate(axes.flatten()):

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters

        try:
            print("processing n_clusters = %d" % n_clusters[k])
            cluster_label = df_clusters["labels_cluster_" + str(n_clusters[k])].values
            sample_silhouette_values = silhouette_samples(
                df_pooled_embedding, cluster_label
            )
            silhouette_avg = silhouette_score(df_pooled_embedding, cluster_label)
            
            print(
                " n_clusters =",
                n_clusters[k],
                ", average silhouette_score =",
                silhouette_avg,
            )
            graph_silhouette(
                sample_silhouette_values,
                cluster_label,
                n_clusters[k],
                silhouette_avg,
                ax,
            )
        except IndexError:
            pass

    plt.tight_layout()

def plot_metric_vs_cluster(df):
    fig, ax = plt.subplots(figsize=(6,4))
    x = df["n_cluster"].astype(int).values
    y = df["metric"].astype(float).values
    ax.plot(x, y, color="blue")
    ax.set_xlabel("# clusters")
    sns.despine()
    plt.tight_layout()
    
    return ax

def miniBatchClustering(df_pooled_embedding, df_clusters, n_clusters, nbatch=128):
    kmeans = []
    # prepare kmeans clusters
    n_cluster_name = ["labels_cluster_" + str(i) for i in n_clusters]

    for j in n_clusters:
        kmeans.append(MiniBatchKMeans(n_clusters=j, random_state=42, batch_size=nbatch))

    X = df_pooled_embedding.values

    if df_clusters is None:
        df_clusters = pd.DataFrame(index=df_pooled_embedding.index)

    for k, clf in enumerate(kmeans):
        print("Processing cluster %d" % (n_clusters[k]))
        clf.partial_fit(X)
        df_clusters[n_cluster_name[k]] = clf.predict(X)

    return df_clusters