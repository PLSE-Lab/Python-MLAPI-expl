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
def post_tiabs_for_pubtator(paper_ids, records_pubtator):
    # load ids file
    paper_ids = json.load(open(f'{data_path}/{paper_ids}', 'r', encoding = 'utf-8'))
    records = json.load(open(f'{data_path}/{records_pubtator}', 'r', encoding = 'utf-8'))
    
    # post to pubtator for annotation
    section_nums = {}
    count = 0
    input_string = ''
    cord_uids = []
    # query pubtators
    for cord_uid, ids in tqdm(paper_ids.items(), desc = 'post tiabs to pubtator'):
        if cord_uid in records: continue
        
        file = json.load(open(f'{cord_path}/{cord_uid}.json', 'r', encoding = 'utf-8'))
        input_string += f"{cord_uid}|t|{file['title']}\n{cord_uid}|a|{' '.join(file['abstract'])}\n\n"
        cord_uids.append(cord_uid)
        count += 1
        if count < 100: continue
        section_num = post_to_pubtators(input_string)
        section_nums[section_num] = cord_uids
        count = 0
        input_string = ''
        cord_uids = []
    section_num = post_to_pubtators(input_string)
    section_nums[section_num] = cord_uids

    json.dump(section_nums, open(f'{data_path}/pubtator_section_nums.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    paper_ids = 'paper_ids.json'
    records_pubtator = 'records_pubtator.json'
    post_tiabs_for_pubtator(paper_ids, records_pubtator)


def run_setup():
    app.run(get_data)


#if __name__ == "__main__":
#    run_setup()
