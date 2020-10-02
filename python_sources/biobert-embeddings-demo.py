#!/usr/bin/env python
# coding: utf-8

# We will use the BioBERT Vocab uncased model as that is what is recommended on the official GitHub Page.

# # BioBERT Embeddings Analysis
# This is a basic tutorial of how to download and use the BioBERT model to create naive embeddings, which can be used for exploring concepts in the literature corpus. Of course long term we would probably want to fine-tune this model in a unsupervised fashion on the document corpus. Additionally, many of the demonstrated techniques are naive (for instance simply averaging the word embeddings to form a sentence embedding), however this demonstrates how embeddings could be used for this challenge

# In[ ]:


get_ipython().system('pip install transformers')
get_ipython().system('wget -O scibert_uncased.tar https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz')
get_ipython().system('tar -xvf scibert_uncased.tar')

import torch
from transformers import BertTokenizer, BertModel
import argparse
import logging

import torch

from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert


logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
convert_tf_checkpoint_to_pytorch("biobert_v1.1_pubmed/model.ckpt-1000000", "biobert_v1.1_pubmed/bert_config.json", "biobert_v1.1_pubmed/pytorch_model.bin")


# In[ ]:


get_ipython().system('ls biobert_v1.1_pubmed')
get_ipython().system('mv biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json')
get_ipython().system('ls biobert_v1.1_pubmed')
model_version = 'biobert_v1.1_pubmed'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)


# In[ ]:





# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states

def get_similarity(em, em2):
    return cosine_similarity(em.detach().numpy(), em2.detach().numpy())


# In[ ]:


coronavirus_em = embed_text("Coronavirus", model).mean(1)
# We will use a mean of all word embeddings.
mers_em = embed_text("Middle East Respiratory Virus", model).mean(1)
flu_em = embed_text("Flu", model).mean(1)
dog_em = embed_text("Bog", model).mean(1)
print("Similarity for Coronavirus and Flu:" + str(get_similarity(coronavirus_em, flu_em)))
print("Similarity for Coronavirus and MERs:" + str(get_similarity(coronavirus_em, mers_em)))
print("Similarity for Coronavirus and Bog:" + str(get_similarity(coronavirus_em, dog_em)))


# So we can see anecdotally even in the raw embeddings there seems to be at least some correlation between concepts. Note that our embedding method 
# 
# Let's now look at visualizing some of these vectors with U-Map. I'm choosing U-Map here due to the high-dimensionality of the data (768-D) and its ability to scale. However, I will also add some T-SNE visualizations below if I have time

# In[ ]:


get_ipython().system('pip install umap-learn')
import umap
reducer = umap.UMAP()


# In[ ]:


import os
import json 
def make_the_embeds(number_files, start_range=0, 
                    the_path="/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset", data_key=["metadata", "title"]):
    the_list = os.listdir(the_path)
    title_embedding_list = [] 
    title_list = []
    for i in range(start_range, number_files):
        file_name = the_list[i]
        final_path = os.path.join(the_path, file_name)
        with open(final_path) as f:
            data = json.load(f)
        tensor, title = make_data_embedding(data, data_key)
        title_embedding_list.append(tensor)
        title_list.append(title)
    return torch.cat(title_embedding_list, dim=0), title_list
        
def make_data_embedding(article_data, data_keys, method="mean", dim=1):
    data = article_data
    for key in data_keys:
        data = data[key]
    text = embed_text(data, model)
    if method == "mean":
        return text.mean(dim), data
    
#embed_list, title_list = make_the_embeds(200)
#red = reducer.fit_transform(embed_list.detach().numpy())


# I found 200 to be a good chunk size for running quick analysis as doing a full plot can get kind of crowded and is slow to compute.

# In[ ]:


from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10, Category20c
from bokeh.palettes import magma
import pandas as pd
output_notebook()


# In[ ]:


def make_plot(red, title_list, number, color = True):
    digits_df = pd.DataFrame(red, columns=('x', 'y'))
    digits_df['digit'] = title_list
    datasource = ColumnDataSource(digits_df)
    plot_figure = figure(
    title='UMAP projection of the article title embeddings',
    plot_width=890,
    plot_height=600,
    tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 10px; color: #224499'></span>
        <span style='font-size: 10px'>@digit</span>
    </div>
    </div>
    """))
    if color:
        color_mapping = CategoricalColorMapper(factors=title_list, palette=magma(number))
        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='digit', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=7
        )
        show(plot_figure)
    else:
        
        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='digit'),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=7
        )
        show(plot_figure)
    
#make_plot(red, title_list, 200)


# There do seem to be a few interesing patterns when analyizng with U-Map. However, I believe fine-tuning methods could definitely improve the clustering of groups. Let's examine another chunk:

# 

# In[ ]:


embed_list2, title_list2 = make_the_embeds(401, 201)
#red2 = reducer.fit_transform(embed_list.detach().numpy())
#print(len(title_list2))
#make_plot(red2, title_list2, 200)


# We'll attempt to make a plot of all ~~9000~~ ~~1000~~ (that did make it run out of RAM)  articles in that directory (warning this might crash your notebook). For fun we'll make these a different 1000 then what we already viewed.

# In[ ]:


#max_len = len(os.listdir("/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset"))
#embed_list, title_list_full = make_the_embeds(2000,1200)
#red_full = reducer.fit_transform(embed_list.detach().numpy())
#make_plot(red_full, title_list_full, 256, color=False)


# Visualizing with T-SNE 

# ## Part 2 Search Attempts on Titles

# In[ ]:


import collections
search_terms = embed_text("coronavirus infection origin", model).mean(1)


# In[ ]:


def top_n_closest(search_term_embedding, title_embeddings, original_titles, n=10):
    proximity_dict = {}
    i = 0 
    for title_embedding in title_embeddings:
        proximity_dict[original_titles[i]] = {"score": get_similarity(title_embedding.unsqueeze(0),search_term_embedding), 
                                              "title_embedding":title_embedding}
        i+=1
    order_dict = collections.OrderedDict({k: v for k, v in sorted(proximity_dict.items(), key=lambda item: item[1]["score"])})
    proper_list = list(order_dict.keys())[-n:]
    return proper_list, order_dict
        


# In[ ]:


top_titles, order_dict = top_n_closest(search_terms, embed_list2, title_list2)


# In[ ]:


top_titles


# The results actually don't seem that bad given the model doesn't have any specific training.

# DEMO

# In[ ]:





# In[ ]:


get_ipython().system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')
import en_core_sci_sm as en


nlp = en.load()
#while True:
#text = input("question ")
text = "What is known about covid-19 incubation period?"
doc = nlp(text)

print(list(doc.ents))
txt = ""
for ent in list(doc.ents):
    txt += str(ent)
    txt += " "

search_terms2 = embed_text(txt, model).mean(1)
top_titles2, order_dict1 = top_n_closest(search_terms, embed_list2, title_list2)
print(top_titles2)


# ## Embedding Abstracts
# Just for fun and to enrich our knowledge later let's try embedding abstracts. 

# In[ ]:


absd_embeds, abs_orig = make_the_embeds(4, 2, data_key=['abstract', 0, "text"])


# Since the following abstracts will be hard to display in U-Map I won't plot them. Instead let's just look at these two

# In[ ]:


abs_orig[0]


# In[ ]:


abs_orig[1]


# In[ ]:


get_similarity(absd_embeds[0].unsqueeze(0), absd_embeds[1].unsqueeze(0))


# I honestly don't know enough about the subject area to tell if that is a good similarity score for those two. I'll add some more examples in a bit, but for now that should serve as good intro.

# In[ ]:




