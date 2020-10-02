#!/usr/bin/env python
# coding: utf-8

# **Coreference visualization for jupyter notebooks (Code repository: https://github.com/sattree/gpr_pub)**
# 
# AllenNLP style highlighting of mention clusters (https://demo.allennlp.org/coreference-resolution/NjA2MjY3**) extended to stanford corenlp, huggingface, and pronoun resolutions
# ***
# This notebook demonstrates two contributions made to the gpr visualization task:
# 1. Extend the visualization code logic for rendering in jupyter notebooks - allennlp functionality provides these visualizations only through a web app interface and is natively built in js.
# 1. Extend the visualization api to cover and provide a uniform interface for stanford and huggingface coref apis, and to also handle pronoun resolution labels.
# 
# Visualization renderer has a displacy (spacy) style api interface, again aimed at maintaining uniformity in interfaces.
# 
# I found allennlp entity highlighting and linking type of visualizations to be better suited for longer text snippets as opposed to the spacy dependency style visualizations offered by huggingface.
# 
# This kernel is the first in a tri-series of self-contained installments to introduce the GPR problem.
# 1. **Coref visualization**
# 1. Reproducing GAP results - achieves a logloss score of 0.84
# 1. A better baseline - without any training
# 
# By no means am I implying that this series is a comprehensive coverage of the problem. There are numerous wonderful kernels available in the competition to that effect. The aim of this series is to provide a good starting point for fellow participants to hit the ground running.
# ***

# Download and install all dependencies
# * gpr_pub - contains code for visualizations

# In[ ]:


get_ipython().system('git clone https://github.com/sattree/gpr_pub.git')
get_ipython().system('wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip')
get_ipython().system('unzip stanford-corenlp-full-2018-10-05.zip')
get_ipython().system('pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_sm-3.0.0/en_coref_sm-3.0.0.tar.gz')
get_ipython().system('pip install stanfordcorenlp')
get_ipython().system('pip install allennlp --ignore-installed greenlet')
# Huggingface neuralcoref model has issues with spacy-2.0.18
get_ipython().system('conda install -y cymem==1.31.2 spacy==2.0.12')


# In[ ]:


from IPython.core.display import display, HTML
# Add css styles and js events to DOM, so that they are available to rendered html
display(HTML(open('gpr_pub/visualization/highlight.css').read()))
display(HTML(open('gpr_pub/visualization/highlight.js').read()))


# In[ ]:


import pandas as pd

import en_coref_sm
from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.corenlp import CoreNLPParser
from allennlp.predictors.predictor import Predictor

from gpr_pub import visualization


# In[ ]:


# Instantiate stanford corenlp server
STANFORD_CORENLP_PATH = 'stanford-corenlp-full-2018-10-05/'
PORT = 9090
try:
    server = StanfordCoreNLP(STANFORD_CORENLP_PATH, port=PORT, quiet=True)
except OSError as e:
    print('The port is occupied, probably an instance is already running.')
    server = StanfordCoreNLP('http://localhost', port=PORT, quiet=True)
    
STANFORD_SERVER_URL = server.url
ALLENNLP_COREF_MODEL_PATH = 'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz'


# In[ ]:


# create model instances
stanford_model = CoreNLPParser(url=STANFORD_SERVER_URL)
allennlp_model = Predictor.from_path(ALLENNLP_COREF_MODEL_PATH)
huggingface_model = en_coref_sm.load()

# If annotators are not preloaded, stanford model can take a while for the first call and may even timeout
# make a dummy call to the server
try:
    stanford_model.api_call('This is a dummy text.', properties={'annotators': 'coref'})
except:
    pass


# In[ ]:


# Load GPR data
train = pd.read_csv('gpr_pub/data/gap-development.tsv', sep='\t')
# normalizing column names
train.columns = map(lambda x: x.lower().replace('-', '_'), train.columns)
with pd.option_context('display.max_rows', 10, 'display.max_colwidth', 15):
    display(train)


# In[ ]:


# 'proref' is a special case that handles highlighting pronoun references that may be present
# in ground truth or predictions
# the renderer expects ['text', 'pronoun', 'pronoun_offset', 'a_coref', 'a_offset', 'b_coref', 'b_offset'] 
# keys (columns) to be present in the input (row)
row = train.loc[0]
visualization.render(row, proref=True, jupyter=True)


# In[ ]:


rows = []
for idx, row in train.iterrows():
    data = stanford_model.api_call(row.text, properties={'annotators': 'coref'})
    html = visualization.render(data, stanford=True, jupyter=False)
    rows.append({'sample_idx': idx, 
                 'model': 'Stanford',
                 'annotation': html})
    
    data = allennlp_model.predict(row.text)
    html = visualization.render(data, allen=True, jupyter=False)
    rows.append({'sample_idx': idx, 
                 'model': 'AllenNlp',
                 'annotation': html})
    
    data = huggingface_model(row.text)
    html = visualization.render(data, huggingface=True, jupyter=False)
    rows.append({'sample_idx': idx, 
                 'model': 'Huggingface',
                 'annotation': html})
    
    # Special rendering for labelled pronouns, either gold or predicted
    # labels in 'a_coref', 'b_coref'
    html = visualization.render(row, proref=True, jupyter=False)
    rows.append({'sample_idx': idx, 
                 'model': 'GPR',
                 'annotation': html})
    
    break

df = pd.DataFrame(rows).groupby(['sample_idx', 'model']).agg(lambda x: x)
s = df.style.set_properties(**{'text-align': 'left'})
display(HTML(s.render(justify='left')))


# In[ ]:


get_ipython().system('rm -r stanford-corenlp-full-2018-10-05/')
get_ipython().system('rm -r gpr_pub/')
get_ipython().system('rm stanford-corenlp-full-2018-10-05.zip')


# Hope you find these visualizations useful for your projects!
# 
# Stay tuned for the 2nd and 3rd installments...

# In[ ]:




