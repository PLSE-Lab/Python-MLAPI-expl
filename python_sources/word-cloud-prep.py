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

# Any results you write to the current directory are saved as output.
import random
import matplotlib.pylab as plt
from collections import defaultdict
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from config import PUNCT_DICT
from biobert_embedding import join_title_abstract
from eda_text import nlp_preprocess
from wordcloud import WordCloud

# nltk.download("punkt")


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def prepare_word_cloud(df_data, stopwords=[]):
    count = CountVectorizer(
        stop_words=stopwords, tokenizer=LemmaTokenizer(), max_df=1.0, min_df=2
    )

    X = count.fit_transform(df_data.astype(str))

    df_wc = (
        pd.DataFrame(
            X.todense(), columns=count.get_feature_names(), index=df_data.index
        )
        .sum(axis=0)
        .sort_values(ascending=False)
    )

    return df_wc


# Custom Color Function
def grey_color_func(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def plot_word_cloud(word_cloud, ax, n_top=50):
    print("Plotting word cloud...")
    n_top = 50
    i = 0

    for k in word_cloud.keys():
        # take relative word frequencies into account, lower max_font_size
        try:
            df_top_words_sorted = word_cloud[k]
            wordcloud = WordCloud(
                background_color="black",
                max_words=n_top,
                max_font_size=50,
                relative_scaling=0.5,
            ).generate(" ".join(df_top_words_sorted.index))

            ax[i].imshow(
                wordcloud.recolor(color_func=grey_color_func, random_state=42),
                interpolation="bilinear",
            )
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
            title = ", ".join(df_top_words_sorted.head(3).index.tolist())
            ax[i].set_title(title, fontsize=14, fontweight="bold")

            i += 1
        except ValueError:
            pass

    plt.tight_layout(w_pad=-4.0)
