#!/usr/bin/env python
# coding: utf-8

# # Event listings
# 
# This is just a fork of [Boris Filippov's notebook](https://www.kaggle.com/frenzykryger/event-specifications) with some added info and different ordering, plus hiding the code generating the summaries.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from ast import literal_eval
from IPython.display import display, Markdown


# In[ ]:


get_ipython().run_cell_magic('time', '', 'data_dir = Path("/kaggle/input/data-science-bowl-2019/")\nsample_submission = pd.read_csv(data_dir / "sample_submission.csv")\nspecs = pd.read_csv(data_dir / "specs.csv")\ntrain = pd.read_csv(data_dir / "train.csv")')


# In[ ]:


train_info = train[['event_id', 'event_code', 'type', 'title', 'world']]


# In[ ]:


event_info = train_info.groupby(['event_id', 'event_code', 'type', 'title', 'world']).first().reset_index()


# In[ ]:


specs_merged = specs.merge(event_info, on='event_id').sort_values(['event_code', 'type', 'title', 'world', 'event_id'])


# In[ ]:


def print_specs(specs_merged):
    old = pd.options.display.max_colwidth 
    pd.options.display.max_colwidth = 999
    try:
        for i, event_id, info, args, event_code, type_, title, world in specs_merged.itertuples():
            display(Markdown(f"""**{event_code}:** _{title}_ / _{world}_ ({event_id} : {type_})

{info}"""))
            display(pd.DataFrame(data=literal_eval(args)))
    finally:
        pd.options.display.max_colwidth = old
print_specs(specs_merged)

