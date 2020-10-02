# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.import os
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import textwrap


def sphereize(x):
    """Sphereize data, shift the datapoint by the centroid on the embeddings,
    and vectorize it,  with keep_dims=True the result will broadcast
    correctly against the input array.

    Arguments:
        x {np.array} -- embeddings to be normalized

    Returns:
        [np.array] -- normalized embeddings
    """
    centroid_vector = np.mean(x, axis=0)
    N_vectors = x - centroid_vector
    N_vectors = N_vectors / np.linalg.norm(N_vectors, axis=-1, keepdims=True)

    return N_vectors


def compute_pca(X, index, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(
        X_pca, columns=["X_" + str(i) for i in np.arange(n_components)], index=index
    )

    print(
        "We explained %s%% of the variance with %d principal components."
        % (
            str(100 * np.sum(pca.explained_variance_ratio_[:n_components])),
            n_components,
        )
    )

    return df_pca


def tsne_cuda(X, index, df_clusters, n_cluster=10):
    import pycuda.autoinit
    from tsnecuda import TSNE

    tsne = TSNE(n_components=2, perplexity=50, n_iter=3000, learning_rate=10)
    X_tsne = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne, columns=["x", "y"], index=index)
    df_tsne["label"] = df_clusters.loc[index]["labels_cluster_%d" % n_cluster]

    return df_tsne


def resample(df, df_meta, df_clusters, colors, nsamples=5000, n_cluster=10):

    index_sample = df.sample(nsamples).index
    df_loc = df.loc[index_sample].copy()
    df_loc["title"] = df_meta.loc[index_sample].title.values
    df_loc["cluster"] = [
        str(colors[i].lower())
        for i in df_clusters.loc[index_sample]["labels_cluster_%d" % n_cluster].values
    ]
    df_loc["abstract"] = df_meta.loc[index_sample]["abstract"]

    return df_loc


def prepare_info(df_loc, max_len=40):

    title = df_loc["title"].apply(
        lambda x: "<br>".join(textwrap.wrap(x, width=max_len))
        if not pd.isnull(x)
        else x
    )
    abstract = df_loc["abstract"].apply(
        lambda x: "<br>".join(textwrap.wrap(x, width=max_len + 10))
        if not pd.isnull(x)
        else x
    )
    title_abstract = pd.concat([title, abstract], axis=1).apply(
        lambda x: "<br><br><b>Abstract</b>:<br>".join(
            [str(x["title"]), str(x["abstract"])]
        ),
        axis=1,
    )

    return title_abstract


def plot_tsne(df_loc, var_x, var_y, title_abstract, var_z=None):

    if var_z is not None:
        fig = go.Figure(
            go.Scatter3d(
                x=df_loc[var_x].values,
                y=df_loc[var_y].values,
                z=df_loc[var_z].values,
                mode="markers",
                marker=dict(color=df_loc["cluster"].values.tolist(), size=3),
                hovertemplate="<b>Title</b>: %{text}<br>",
                text=title_abstract,
                showlegend=False,
                hoverlabel=dict(bgcolor=df_loc["cluster"].values.tolist()),
                name="",
            )
        )
    else:
        fig = go.Figure(
            go.Scatter(
                x=df_loc[var_x].values,
                y=df_loc[var_y].values,
                # marker = df_tsne["cluster"].values.tolist(),
                marker=dict(color=df_loc["cluster"].values.tolist(), size=5),
                hovertemplate="<b>Title</b>: %{text}<br>",
                text=title_abstract,
                showlegend=False,
                name="",
            )
        )

    fig.update_traces(mode="markers")

    layout = go.Layout(
        autosize=True,
        width=800,
        height=800,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=go.layout.XAxis(title="", showticklabels=False),
        yaxis=go.layout.YAxis(title="", showticklabels=False),
        # hoverlabel=dict(bgcolor=df_tsne["cluster"].values.tolist(), align="auto"),
        font=dict(family="Courier New, monospace", size=12, color="#7f7f7f"),
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    fig.update_layout(layout)

    fig.show()