
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tabulate
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tqdm import tqdm_notebook
from collections import defaultdict
import matplotlib.pylab as plt
# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud
import eda_text as et
import config
import textwrap
import word_cloud_prep as wcp
import biobert_embedding as be
import pdb

class qCOVID19():
    def __init__(self,meta_data, embs_title, embs_title_abstract, modelname = "use"):
        if modelname =="use":
            module_url ="https://tfhub.dev/google/universal-sentence-encoder/4"
            self.embed = hub.load(module_url)
            
        elif modelname =="biobert":
            self.model = load_biobert()
            
        self.embs_title = embs_title
        self.embs_title_abstract = embs_title_abstract
        self.meta_data = meta_data
    
    def load_biobert():
        pass
        
    def query(self, text, with_title_only = True, abstract_width=50, display_info = ["title","abstract","authors","journal","publish_date"]):
        if with_title_only:
            embs = self.embs_title
            
        else:
            embs = self.embs_title_abstract
            
        embedding_text = self.embed([text])
        embedding_text = embedding_text.numpy()
        similarities = np.inner(embedding_text, embs[:, 1:])
        
        indices = np.argsort(similarities)[0]
        indices = indices[::-1]
        
        row_ids = embs[indices, :][:, 0]
        row_ids = list(map(int, row_ids))
         
        df = self.prepare_data(row_ids, similarities.squeeze()[indices], display_info)
        
        if "abstract" in display_info:
            df["shorten_abstract"] = df["abstract"].apply(lambda x: textwrap.shorten(x, width = abstract_width, placeholder="...") if not pd.isnull(x) else "")
                
        del embs
        
        return df
        
    def prepare_data(self, row_ids, similarities, display_info):
        
        df = self.meta_data.loc[row_ids][display_info]
        
        df["similarities"] = similarities

        return df
    
    def word_cloud(self, df, ax, ntop=50, ndoc = 50):
        df["title_abstract"] = df[["title", "abstract"]].apply(
                be.join_title_abstract, axis=1
            )
        df_wc = df["title_abstract"].apply(lambda x: et.nlp_preprocess(str(x), config.PUNCT_DICT) if not pd.isnull(x) else "")
        
        df_wc_r = wcp.prepare_word_cloud(df_wc.iloc[:ndoc])
  
        wordcloud = WordCloud(
            background_color="black",
            max_words=ntop,
            max_font_size=50,
            relative_scaling=0.5,
        ).generate(" ".join(df_wc_r.index))

        ax.imshow(
            wordcloud,
            interpolation="bilinear",
        )
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        title = ", ".join(df_wc_r.head(3).index.tolist())
        ax.set_title(title, fontsize=14, fontweight="bold")
            
        plt.tight_layout()
        del wordcloud, df_wc_r, df_wc,
        return ax
    
        
if __name__ == "__main__":
    FILEDIR = os.path.join("../input/","use-clustering")
    ndisplay = 10
    df_meta = pd.read_csv(os.path.join(FILEDIR, "df_meta_comp.csv" )).set_index("row_id")
    embs_title = np.load(os.path.join(FILEDIR, "embeddings_titles.npy" ))
    embs_title_abstract = np.load(os.path.join(FILEDIR, "embeddings_title_abstract.npy" ))
    qcovid = qCOVID19(df_meta, embs_title, embs_title_abstract)
    df = qcovid.query("Range of incubation periods for the disease in humans and how long individuals are contagious, even after recovery.")
    #print(df.head(ndisplay).sort_values(by="publish_date", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = qcovid.word_cloud(df, ax)
    