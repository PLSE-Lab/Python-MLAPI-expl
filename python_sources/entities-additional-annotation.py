# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import json
#from befree.src.ner import BeFree_NER_covid19

from absl import app

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'


def get_data(argv):
    # Data directories
    records_pub_entity = 'records_pub_entity.json'
    records_add_ents = 'records_add_ents.json' #None # 
    loc = 'befreeout'
    if not os.path.exists(loc): os.makedirs(loc)
    BeFree_NER_covid19.entity_identification(data_path, records_pub_entity,
                                             entity_path, add_ents_records=records_add_ents, loc=loc)


def run_setup():
    app.run(get_data)

#if __name__ == "__main__":
#    run_setup()
