#!/usr/bin/env python
# coding: utf-8

# # Kensho Derived Wikimedia Dataset (KDWD) - Create NER Labels for Wikidata Items Based on Subclass Paths

# In[ ]:


import os

import networkx as nx
import numpy as np
import pandas as pd


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read Wikidata Statements 

# In[ ]:


file_path = '/kaggle/input/kensho-derived-wikimedia-data/statements.csv'
qpq_df = pd.read_csv(file_path, dtype={'source_item_id': int, 'edge_property_id': int, 'target_item_id': int})


# In[ ]:


qpq_df


# # Create Subclass Directed Graph

# In[ ]:


p279g = nx.DiGraph()
p279g.add_edges_from(qpq_df[qpq_df['edge_property_id']==279][['source_item_id', 'target_item_id']].values)


# # Define NER Root Nodes

# In[ ]:


root_qids = {
    'per': 5,        # https://www.wikidata.org/wiki/Q5  human
    'loc': 2221906,  # https://www.wikidata.org/wiki/Q2221906  geographic location
    'org': 43229,    # https://www.wikidata.org/wiki/Q43229  organization
}


# # Calculate Subclasses of Each Root Node

# In[ ]:


subclass_qids = {
    lbl: set(nx.ancestors(p279g, qid)).union(set([qid]))
    for lbl, qid in root_qids.items()
}


# # Create NER Signatures for each Wikidata Item

# In[ ]:


ner_df = pd.DataFrame(index=qpq_df['source_item_id'].unique())
ner_df.index.name = 'qid'


# In[ ]:


mask1 = qpq_df['edge_property_id'] == 31
for lbl, qid in root_qids.items():
    mask2 = qpq_df['target_item_id'].isin(subclass_qids[lbl])
    tmp_df = qpq_df[mask1 & mask2][['source_item_id', 'target_item_id']]
    tmp_df.set_index('source_item_id', drop=True, inplace=True)
    tmp_df.index.name = 'qid'
    
    # de-duplicate index 
    tmp_df = tmp_df[~tmp_df.index.duplicated()]
    
    # add to dataframe
    ner_df[lbl] = tmp_df['target_item_id']


# In[ ]:


ner_df = ner_df.dropna(how='all').fillna(0).astype(np.int)


# In[ ]:


ner_df


# # Interpretation
# 
# Each cell in the `ner_df` DataFrame is 0 or a Wikidata item id.  The second row indicates that [Q16 (Canada)](https://www.wikidata.org/wiki/Q16) has NER type location b/c it is an [P31 (instance of)](https://www.wikidata.org/wiki/Property:P31) [Q3624078 (sovereign state)](https://www.wikidata.org/wiki/Q3624078) which is in the subclass path of the root item for location, [Q2221906 (geographic location)](https://www.wikidata.org/wiki/Q2221906).  Because [Q3624078 (sovereign state)](https://www.wikidata.org/wiki/Q3624078) is also in the subclass path of the root item for organization [Q43229 (organization)](https://www.wikidata.org/wiki/Q43229) it also has NER type organization.  
# 
# Note that an item can be an instance of multiple other items that would make it a certain NER type.  If this happens, we use pandas to deduplicate and only record one.  For example, [Q17 (Japan)](https://www.wikidata.org/wiki/Q17) is an [P31 (instance of)](https://www.wikidata.org/wiki/Property:P31) [Q3624078 (sovereign state)](https://www.wikidata.org/wiki/Q3624078) and [Q112099 (island nation)](https://www.wikidata.org/wiki/Q112099), both of which are in the subclass path of [Q2221906 (geographic location)](https://www.wikidata.org/wiki/Q2221906), but we only record [Q112099 (island nation)](https://www.wikidata.org/wiki/Q112099).

# In[ ]:


ner_df.to_csv('wikidata_subclass_path_ner.csv')

