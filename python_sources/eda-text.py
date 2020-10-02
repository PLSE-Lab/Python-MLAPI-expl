# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.

# %% [code]
import re
import matplotlib.pylab as plt
from collections import defaultdict
from config import STOPWORDS
import seaborn as sns
import string

sns.set_style("ticks")


def multiple_replace(text, dict):

    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def remove_http(text):
    return re.sub(r"http?://\S+", "", text, flags=re.MULTILINE)


def remove_alone_digit(text):
    return re.sub(r"\s\d+\s", " ", text, flags=re.MULTILINE)


def nlp_preprocess(x, punct_dict):
    http_x = remove_http(str(x))
    sub_x = multiple_replace(http_x, punct_dict)
    return " ".join([i for i in sub_x.lower().split() if i not in STOPWORDS])


def process_author(x):

    prep_text = str(x).lower().split("; ")
    splitted_feats = [x for x in prep_text if not x in ["nan"]]

    return splitted_feats


def process_affiliations(x):
    """
    retrieve the last occurence of affiliation because the last author is
    normally the head of the group
    """
    match = re.search(r"\(([^\)(]+)\)$", str(x), flags=re.MULTILINE)

    if bool(match):
        return match.groups()

    return "nan"


def corpora_freq(df, authors=False, affiliation=False):
    wordcount = defaultdict(int)

    for k, v in df.iteritems():
        if affiliation:
            wordcount[v] += 1

        else:
            if authors:
                v_list = v
            else:
                v_list = v.split()

            for j in v_list:
                wordcount[j] += 1

    sorted_wordcount = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)

    df_wordcount = pd.DataFrame(sorted_wordcount, columns=["word", "count"])

    return df_wordcount


def plot_distrib(df_wc, name, ntop=40):
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x="word", y="count", data=df_wc[:ntop], ax=ax)
    sns.despine()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    
    return ax


def nlp_preprocess_text(x, punct_dict):
    extra_stopwords = list(STOPWORDS) + [
        "et",
        "al",
        "fig",
        "figure",
        "can",
        "may",
        "also",
        "doi",
        "biorxiv",
        "preprint",
        "copyright",
        "peer-reviewed",
        "authorfunder",
        "table",
        "license",
    ]
    http_x = remove_http(str(x))
    digit_x = remove_alone_digit(http_x)
    sub_x = multiple_replace(digit_x, punct_dict)

    return " ".join([i for i in sub_x.lower().split() if i not in extra_stopwords])