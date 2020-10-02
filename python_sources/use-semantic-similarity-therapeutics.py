#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pylab as plt
from IPython.display import display, HTML
from tabulate import tabulate
import query_covid19 as qc
FILEDIR = os.path.join("../input/","use-covid19-search-engine")

# Any results you write to the current directory are saved as output.


# We have previously written a kernel on building a search engine based on semantic similarity using embeddings from Universal Sentence Encoder (USE) in which we have done an EDA and provided clustering among all articles so that users can get some insights by navigating between simiar articles. (Please refer to [Covid 19 search engine based on semantic similarity using Universal Sentence Encoder](https://www.kaggle.com/pmlee2017/use-covid19-search-engine/) for more info). 
# 
# In this kernel we will use the embeddings in our previous work to answer questions for the task **"What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?"**
# 
# We show the top 10 most similar results and a word cloud generated based on the title and abstract of the top 50 articles. The displayed results are sorted by date.

# # Load files

# In[ ]:


df_meta = pd.read_csv(os.path.join(FILEDIR, "df_meta_comp.csv" )).set_index("row_id")
embs_title = np.load(os.path.join(FILEDIR, "embeddings_titles.npy" ))
embs_title_abstract = np.load(os.path.join(FILEDIR, "embeddings_title_abstract.npy" ))

# Initialize
qcovid = qc.qCOVID19(df_meta, embs_title, embs_title_abstract)


# In[ ]:


# define function for better display
def display_q(text, df_top, display_cols = ["publish_date", "title", "shorten_abstract", "authors", "journal", "similarities"]):
    display(HTML("Search term : <b>%s<b>" %text))
    display(HTML(tabulate(df_top[display_cols], headers = display_cols,tablefmt='html')))


# # Process queries

# In[ ]:


questions_temp = [
    "Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.",
    "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.",
    "Exploration of use of best animal models and their predictive value for a human vaccine.",
    "Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.",
    "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.",
    "Efforts targeted at a universal coronavirus vaccine.",
    "Efforts to develop animal models and standardize challenge studies",
    "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers",
    "Approaches to evaluate risk for enhanced disease after vaccination",
    "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models in conjunction with therapeutics",
]

#prepend these words, to emphasize on covid19
prepend_words = ["Covid-19, coronavirus"]
questions = []

for question in questions_temp :
    questions.append(", ".join(prepend_words + [question] ))


# In[ ]:


def generate_resp(question, ntop=10):
    df = qcovid.query(question, abstract_width=200)
    display_q(question, df.head(ntop).sort_values(["publish_date"], ascending=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    qcovid.word_cloud(df, ax)
    
    
    


# In[ ]:


generate_resp(questions[0])
    


# In[ ]:


generate_resp(questions[1])


# In[ ]:


generate_resp(questions[2])


# In[ ]:


generate_resp(questions[3])


# In[ ]:


generate_resp(questions[4])


# In[ ]:


generate_resp(questions[5])


# In[ ]:


generate_resp(questions[6])


# In[ ]:


generate_resp(questions[7])


# In[ ]:


generate_resp(questions[8])


# In[ ]:


generate_resp(questions[9])

