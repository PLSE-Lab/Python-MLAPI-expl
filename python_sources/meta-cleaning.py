# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pylab as plt
import seaborn as sns
import json
import requests
from time import sleep
from helper_functions import load_files, generate_clean_df

sns.set_style("ticks")


def annotate_values(ax, offset_x=0.1, offset_y=650, rotation=0):
    """
    annotate values on figures, especially for barchart
    Arguments:
        ax {matplotlib axes} -- [description]

    Keyword Arguments:
        offset_x {float} -- [description] (default: {0.1})
        offset_y {int} -- [description] (default: {650})
        rotation {int} -- [description] (default: {0})
    """

    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(
            i.get_x() + offset_x,
            i.get_height() + offset_y,
            str(i.get_height()),
            fontsize=12,
            rotation=rotation,
        )


def plot_missing_value_barchart(df, save=True):
    # Generate missing value figures
    fig, ax = plt.subplots(figsize=(10, 5))
    df.count().sort_values(ascending=False).plot.bar(ax=ax)
    annotate_values(ax, offset_x=-0.1, offset_y=500)
    sns.despine()
    ax.set_ylabel("Count")
    ax.set_xlabel("Variables")
    plt.tight_layout()

def author_feats(x):
    if not pd.isnull(x):
        x = x.replace(",", ";")
        splitted_feats = x.split("; ") if not pd.isnull(x) else x
        num_authors = len(splitted_feats)
        f_author = (splitted_feats[0],)
        l_author = (splitted_feats[-1],)
        author_list = splitted_feats

    else:
        num_authors = 0
        f_author = (np.NaN,)
        l_author = (np.NaN,)
        author_list = []

    return pd.Series(
        {
            "num_authors": num_authors,
            "first_author": f_author,
            "last_author": l_author,
            "authors_list": author_list,
        }
    )


def plot_num_author_distrib(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    df.value_counts().plot.bar(ax=ax)
    ax.set_xlabel("# authors")
    ax.set_ylabel("Count")
    plt.tight_layout()
    sns.despine()


def plot_article_sources_distrib(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    df.source_x.value_counts().plot.bar(ax=ax)
    annotate_values(ax, offset_x=0, offset_y=500)
    ax.set_xlabel("Sources")
    ax.set_ylabel("Count")
    sns.despine()

    plt.tight_layout()


def groupby_publish_date(df):
    df_publish_date = (
        df[["publish_date", "sha"]]
        .groupby(["publish_date"])
        .apply(lambda x: x.count())
        .rename(columns={"sha": "count"})
        .drop(columns=["publish_date"])
        .reset_index()
    )

    return df_publish_date


def gropuby_date_source(df):
    df_date_source = (
        df[["publish_date", "source_x", "cord_uid"]]
        .groupby(["publish_date", "source_x"])
        .agg("count")
        .reset_index()
    )

    return df_date_source


def plot_publish_date_wrt_sources(df):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    start_date = datetime.date(2002, 1, 1)
    end_date = df.publish_date.dt.date.max()
    unique_df_date_source = df.source_x.unique()

    for i, ax in enumerate(axes.flatten()):
        df_selected_temp = df[
            np.logical_and.reduce(
                [
                    df.publish_date.dt.date > start_date,
                    df.publish_date.dt.date < end_date,
                    df.source_x == unique_df_date_source[i],
                ]
            )
        ]

        ax = sns.lineplot(
            x="publish_date", y="cord_uid", data=df_selected_temp, ax=ax, markers=True
        )
        ax.set_title(unique_df_date_source[i])

    sns.despine()
    plt.tight_layout()




def plot_publish_date_distrib(df):
    fig, ax = plt.subplots(figsize=(16, 5))
    start_date = df["publish_date"].dt.date.min()
    end_date = df["publish_date"].dt.date.max()

    df_selected_date = df[
        np.logical_and(
            df["publish_date"].dt.date > start_date,
            df["publish_date"].dt.date < end_date,
        )
    ]

    ax = sns.lineplot(x="publish_date", y="count", data=df_selected_date, ax=ax)

    sns.despine()
    plt.tight_layout()

def parse_biorxiv(biorxiv_path):

    biorxiv_filenames = load_files(biorxiv_path)
    df_biorxiv = generate_clean_df(biorxiv_filenames)

    return df_biorxiv


def parse_comm(comm_subset_path):

    comm_subset_filenames = load_files(comm_subset_path)
    df_comm = generate_clean_df(comm_subset_filenames)

    return df_comm


def parse_noncomm(noncomm_subset_path):

    noncomm_subset_filenames = load_files(noncomm_subset_path)
    df_noncomm = generate_clean_df(noncomm_subset_filenames)

    return df_noncomm


def merge_datasets(df_meta, df_biorxiv, df_comm, df_noncomm):
    keep_col_articles = [
        "paper_id",
        "affiliations",
        "text",
        "bibliography",
        "raw_bibliography",
        "title",
        "abstract",
        "authors",
    ]

    keep_col_meta_author = [
        "cord_uid",
        "sha",
        "source_x",
        "title",
        "abstract",
        "authors",
        "journal",
        "publish_date",
        "doi",
        "url",
    ]
    df_concat = pd.concat(
        [df_biorxiv[keep_col_articles], df_comm[keep_col_articles]], axis=0
    )
    df_concat = pd.concat([df_concat, df_noncomm[keep_col_articles]], axis=0)

    df_merge = df_meta[keep_col_meta_author].merge(
        df_concat, left_on=["sha"], right_on=["paper_id"], how="left"
    )

    return df_merge

def drop_duplicates(df_merge_impute):
    keep_cols = [
        "cord_uid",
        "sha",
        "source_x",
        "title",
        "abstract",
        "authors",
        "journal",
        "publish_date",
        "affiliations",
        "text",
        "bibliography",
        "raw_bibliography",
        "doi",
        "url",
    ]

    df_authors = df_merge_impute["authors"].apply(author_feats)
    df_temp = df_merge_impute[keep_cols].merge(
        df_authors, left_index=True, right_index=True
    )
    df_clean = df_temp.drop_duplicates(subset=["cord_uid","sha",
        "source_x",
        "title",
        "journal",
        "publish_date"])
    df_clean_row = df_clean.reset_index().drop(columns=["index"])
    df_clean_row.index.name ="row_id" 

    return df_clean_row


def impute_columns(df):
    col1 = df["title_x"].copy()
    col2 = df["title_y"].copy()
    col1[pd.isna(col1)] = col2[pd.isna(col1)]

    col3 = df["abstract_x"].copy()
    col4 = df["abstract_y"].copy()
    col3[pd.isna(col3)] = col4[pd.isna(col3)]

    col5 = df["authors_x"].copy()
    col6 = df["authors_y"].copy()
    col5[pd.isna(col5)] = col6[pd.isna(col5)]

    col7 = df["sha"].copy()
    col8 = df["paper_id"].copy()
    col7[pd.isna(col7)] = col8[pd.isna(col7)]

    df = df.assign(**{"title": col1, "abstract": col3, "authors": col5, "sha": col7})

    return df


def get_referenced_by_count(x):
    if not pd.isnull(x):
        try:

            headers = {"User-Agent": "mailto:lee.patrickmunseng@gmail.com"}
            r = requests.get("https://api.crossref.org/works/%s" % x, headers=headers)
            res = pd.Series(
                {
                    "is-referenced-by-count": r.json()["message"][
                        "is-referenced-by-count"
                    ]
                }
            )
            sleep(0.05)
        except json.decoder.JSONDecodeError:
            res = pd.Series({"is-referenced-by-count": np.NaN})
        return res

    return pd.Series({"is-referenced-by-count": np.NaN})
        
# Any results you write to the current directory are saved as output.