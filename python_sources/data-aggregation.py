# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import re
import csv
import json
from tqdm import tqdm

from absl import app

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'

# function for process  kaggle input data
def data_aggregation(records_add_ents, records_data_aggregate = None):
    records_add_ents =  json.load(open(f'{data_path}/{records_add_ents}', 'r', encoding = 'utf-8'))
    records_data_aggregate = {} if records_data_aggregate == None else json.load(open(f'{data_path}/{records_data_aggregate}', 'r', encoding = 'utf-8'))
    
    # start aggregation
    for k, v in tqdm(records_add_ents.items(), desc = 'data aggregation'):
        if k in records_data_aggregate: continue
        file = json.load(open(f'{cord_path}/{k}.json', 'r', encoding = 'utf-8'))
        entity_file = json.load(open(f'{entity_path}/{k}.json', 'r', encoding = 'utf-8'))
        file['title'] = entity_file['title']
        file['abstract'] = entity_file['abstract']
        
        json.dump(file, open(f'{json_path}/{k}.json', 'w', encoding = 'utf-8'), indent = 4)
        records_data_aggregate[k] = v

    json.dump(records_data_aggregate, open(f'{data_path}/records_data_aggregate.json', 'w', encoding = 'utf-8'), indent = 4)


def process_data(argv):
    # Data directories
    records_add_ents = 'records_add_ents.json'
    records_data_aggregate = 'records_data_aggregate.json' #None # 
    if not os.path.exists(json_path): os.makedirs(json_path)
    data_aggregation(records_add_ents, records_data_aggregate = records_data_aggregate)

def run_setup():
    app.run(process_data)

#if __name__ == "__main__":
#    run_setup()
