# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import json
from tqdm import tqdm

from absl import app

from utils import *

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'

# function for process  kaggle input data
def retrieve_posted_tiabs_from_pubtator(pubtator_section_nums, records_pubtator):
    # load ids file
    section_nums = json.load(open(f'{data_path}/{pubtator_section_nums}', 'r', encoding = 'utf-8'))
    records = json.load(open(f'{data_path}/{records_pubtator}', 'r', encoding = 'utf-8'))
    
    # retrieve pubtators
    success_nums = []
    for section_num, ids in tqdm(section_nums.items(), desc = 'retrieve pubtator annotations'):
        pubtators = retrieve_pubtators(section_num)
        if pubtators != None:
            try:
                assert len(pubtators) == len(ids)
            except:
                logging.info(f'AssertionError with section num: {section_num}')
                continue
            for id_idx, pid in enumerate(ids):
                with open(f'{pubtator_path}/{pid}.txt', 'w', encoding = 'utf-8') as f:
                    f.write(pubtators[id_idx])
                records[pid] = 'post'
            success_nums.append(section_num)
    
    for num in success_nums:
        section_nums.pop(num)

    json.dump(section_nums, open(f'{data_path}/pubtator_section_nums.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(records, open(f'{data_path}/records_pubtator.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    pubtator_section_nums = 'pubtator_section_nums.json'
    records_pubtator = 'records_pubtator.json'
    retrieve_posted_tiabs_from_pubtator(pubtator_section_nums, records_pubtator)


def run_setup():
    app.run(get_data)


#if __name__ == "__main__":
#    run_setup()
