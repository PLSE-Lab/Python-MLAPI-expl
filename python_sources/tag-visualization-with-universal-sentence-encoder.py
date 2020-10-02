#!/usr/bin/env python
# coding: utf-8

# # Tag Visualization with Universal Sentence Encoder
# 
# This kernel is based on [An Attemplt to Visualize Topic Model (LDA)](https://www.kaggle.com/ceshine/an-attemplt-to-visualize-topic-model-lda). This kernel removes the topic model and instead focus on the sentence embeddings space from the universal sentence encoder model and the (hash)tags.
# 
# As with the previous kernel, hashtags are removed from the question texts. So the universal sentence encoder does **not** have any direct information whatsoever regarding to the questions associated hastags.
# 
# > Hashtags are removed. I want to separate natural language understanding from (implicit) tag grouping in this task, that is, only focus on the questions, not tags.
# 
# Dimension reduction is done via [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/). And [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-large/3) comes from [TensorFlow Hub](https://www.tensorflow.org/hub).

# ## Contents
# 
# 1. [Imports](#Imports)
# 1. [Preprocessing](#Preprocessing)
#   * [Checking](#Checking)
#   * [The Real Deal](#The-Real-Deal)
# 1. [Tags](#Tags)
# 1. [Sentence Embeddings](#Sentence-Embeddings)
# 1. [Visualization (Global)](#Global-Visualization)
# 1. [Visualization (Tags)](#Tag-Visualization)
#   * [#medicine & #engineering](##medicine-&-#engineering)
#   * [#medicine & #business](##medicine-&-#business)
#   * [#engineering & #business & #medicine](##engineering-&-#business-&-#medicine)
# 1. [Summary](#Summary)

# ## Imports

# In[ ]:


import os
import re
import html as ihtml
import warnings
import random
warnings.filterwarnings('ignore')

os.environ["TFHUB_CACHE_DIR"] = "/tmp/"

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import scipy
import umap

import tensorflow as tf
import tensorflow_hub as hub

import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', -1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


umap.__version__


# ## Preprocessing

# In[ ]:


input_dir = '../input'

questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))
tags = pd.read_csv(os.path.join(input_dir, 'tags.csv'))
tag_questions = pd.read_csv(os.path.join(input_dir, 'tag_questions.csv'))


# In[ ]:


def clean_text(text, remove_hashtags=True):
    text = BeautifulSoup(ihtml.unescape(text), "lxml").text
    text = re.sub(r"http[s]?://\S+", "", text)
    if remove_hashtags:
        text = re.sub(r"#[a-zA-Z\-]+", "", text)
    text = re.sub(r"\s+", " ", text)        
    return text


# In[ ]:


questions['questions_full_text'] = questions['questions_title'] + ' '+ questions['questions_body']


# ### Checking

# In[ ]:


sample_text = questions[questions['questions_full_text'].str.contains("&a")]["questions_full_text"].iloc[0]
sample_text


# In[ ]:


sample_text = clean_text(sample_text)
sample_text


# ### The Real Deal

# In[ ]:


get_ipython().run_cell_magic('time', '', "questions['questions_full_text'] = questions['questions_full_text'].apply(clean_text)")


# In[ ]:


questions['questions_full_text'].sample(2)


# ## Tags

# Top tags:

# In[ ]:


tag_questions.groupby(
    "tag_questions_tag_id"
).size().sort_values(ascending=False).to_frame("count").merge(
    tags, left_on="tag_questions_tag_id", right_on="tags_tag_id"
).head(10)


# There are 24 questions tagged both #medicine and #engineering:

# In[ ]:


questions_id_medicine = set(tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id)
questions_id_engineering = set(tag_questions[tag_questions.tag_questions_tag_id == 54].tag_questions_question_id)
len(questions_id_medicine), len(questions_id_engineering), len(questions_id_medicine.intersection(questions_id_engineering))


# ## Sentence Embeddings
# 
# The model used is the universal sentence encoder (large/transformer) version 3. The extracted sentence embeddings will have a dimension of 512. Here we also use cosine similarity.

# In[ ]:


embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")


# In[ ]:


import logging
from tqdm import tqdm_notebook
tf.logging.set_verbosity(logging.WARNING)
BATCH_SIZE = 128

sentence_input = tf.placeholder(tf.string, shape=(None))
# For evaluation we use exactly normalized rather than
# approximately normalized.
sentence_emb = tf.nn.l2_normalize(embed(sentence_input), axis=1)

sentence_embeddings = []       
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm_notebook(range(0, len(questions), BATCH_SIZE)):
        sentence_embeddings.append(
            session.run(
                sentence_emb, 
                feed_dict={
                    sentence_input: questions["questions_full_text"].iloc[i:(i+BATCH_SIZE)].values
                }
            )
        )


# In[ ]:


sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
sentence_embeddings.shape


# ## Global Visualization
# 
# Here we plot 10,000 samples to give readers a sense of what does the embedding space looks like.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'embedding = umap.UMAP(metric="cosine", n_components=2, random_state=42).fit_transform(sentence_embeddings)')


# In[ ]:


df_se_emb = pd.DataFrame(embedding, columns=["x", "y"])


# In[ ]:


df_emb_sample = df_se_emb.sample(10000)
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    df_emb_sample["x"].values, df_emb_sample["y"].values, s=1
)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Sentence embeddings embedded into two dimensions by UMAP", fontsize=18)
plt.show()


# ### Examine the outliers
# 
# 

# The far right cluster contains questions related to "stream". Many of them starts with "In which stream". Arguably this is not the best way to map these questions, as it's not useful in recommending questions to the professionals.

# In[ ]:


print(questions[df_se_emb.x > 10].shape[0])
questions[df_se_emb.x > 10].questions_full_text.sample(5)


# The top cluster is related to textbooks. This one looks rather reasonable.

# In[ ]:


print(questions[df_se_emb.y > 8].shape[0])
questions[df_se_emb.y > 8].questions_full_text.sample(5)


# ## Tag Visualization
# 
# The number of questions is relative small here so we could use the new [Plotly Express](https://medium.com/@plotlygraphs/introducing-plotly-express-808df010143d) library to provide interactive visualizations. The legends in the right side are clickable. Use them to help you better distinguish the questions from different tags.

# ### #medicine & #engineering

# In[ ]:


questions_id_medicine = tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id
questions_id_engineering = tag_questions[tag_questions.tag_questions_tag_id == 54].tag_questions_question_id
df_se_emb["tag"] = "none"
df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"
df_se_emb.loc[questions.questions_id.isin(questions_id_engineering), "tag"] = "engineering"
df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(
    set(questions_id_engineering)))), "tag"] = "both"


# In[ ]:


df_se_emb.tag.value_counts()


# In[ ]:


px.colors.qualitative.D3


# In[ ]:


df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()
df_emb_sample["tag"] = df_emb_sample.tag.astype("category")
df_emb_sample["size"] = 20
px.scatter(
    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",
    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,
    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid
    
)


# ### #medicine & #business

# In[ ]:


questions_id_medicine = tag_questions[tag_questions.tag_questions_tag_id == 89].tag_questions_question_id
questions_id_biz = tag_questions[tag_questions.tag_questions_tag_id == 27292].tag_questions_question_id
df_se_emb["tag"] = "none"
df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"
df_se_emb.loc[questions.questions_id.isin(questions_id_biz), "tag"] = "business"
df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(
    set(questions_id_biz)))), "tag"] = "both"


# In[ ]:


df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()
df_emb_sample["tag"] = df_emb_sample.tag.astype("category")
df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()
df_emb_sample["tag"] = df_emb_sample.tag.astype("category")
df_emb_sample["size"] = 20
px.scatter(
    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",
    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,
    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid
    
)


# ### #engineering & #business & #medicine
# 
# In this plot we remove all the intersections to make the plot cleaner:

# In[ ]:


df_se_emb["tag"] = "none"
df_se_emb.loc[questions.questions_id.isin(questions_id_engineering), "tag"] = "engineering"
df_se_emb.loc[questions.questions_id.isin(questions_id_biz), "tag"] = "business"
df_se_emb.loc[questions.questions_id.isin(questions_id_medicine), "tag"] = "medicine"
df_se_emb.loc[questions.questions_id.isin((set(questions_id_engineering).intersection(
    set(questions_id_biz)))), "tag"] = "none"
df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(
    set(questions_id_biz)))), "tag"] = "none"
df_se_emb.loc[questions.questions_id.isin((set(questions_id_medicine).intersection(
    set(questions_id_engineering)))), "tag"] = "none"


# In[ ]:


df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()
df_emb_sample["tag"] = df_emb_sample.tag.astype("category")
df_emb_sample = df_se_emb.loc[df_se_emb.tag != "none"].copy()
df_emb_sample["tag"] = df_emb_sample.tag.astype("category")
df_emb_sample["size"] = 20
px.scatter(
    df_emb_sample, x="x", y="y", color="tag", template="plotly_white", size="size",
    range_x=((-7, 12)), range_y=((-9, 10)), opacity=0.3, size_max=5,
    width=800, height=600, color_discrete_sequence=px.colors.qualitative.Vivid
    
)


# ## Summary
# 
# The questions with #business, #engineering, and #medicine tags each formed one bigger cluster that are clearly separated from each other. Those questions are the "easy" ones. We can easily create a tag recommendation model for those questions. The others, however, can be a bit problematic. Many of them are still distinguishable, but might require higher degree of non-linearity to model them.
# 
# The results are better than I expected. At least we did not get something that are fully intertwined. The universal sentence encoder does show some potential for this problem.
