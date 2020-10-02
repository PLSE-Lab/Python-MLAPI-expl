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
def time_indexing_year_month(mapping_corduid2nid, records_data_aggregate,
                             index_time_year = None, index_time_month = None):
    mapping_file =  json.load(open(f'{data_path}/{mapping_corduid2nid}', 'r', encoding = 'utf-8'))
    aggregate_records =  json.load(open(f'{data_path}/{records_data_aggregate}', 'r', encoding = 'utf-8'))
    time_pattern = re.compile(r'(19[0-9]{2}|20[0-9]{2})(-(1[0-2]|0?[1-9])(?=-|$)(-(3[01]|[1-2][0-9]|0?[1-9])(?=$))?)?')
    time_indexing_year = {} if index_time_year == None else json.load(open(f'{data_path}/{index_time_year}', 'r', encoding = 'utf-8'))
    time_indexing_month = {} if index_time_month == None else json.load(open(f'{data_path}/{index_time_month}', 'r', encoding = 'utf-8'))
    # start indexing
    for k, v in tqdm(aggregate_records.items(), desc = 'processing indexing by year and month'):
        file = json.load(open(f'{json_path}/{k}.json', 'r', encoding = 'utf-8'))
        reget = re.search(time_pattern, file['publish_time'])
        if reget:
            if reget[1]:
                time_indexing_year[str(reget[1])] = time_indexing_year.get(str(reget[1]), [])
                if mapping_file[k] not in time_indexing_year[str(reget[1])]:
                    time_indexing_year[str(reget[1])].append(mapping_file[k])
            else:
                time_indexing_year['0'] = time_indexing_year.get('0', [])
                if mapping_file[k] not in time_indexing_year['0']:
                    time_indexing_year['0'].append(mapping_file[k])
            
            if reget[3]:
                month = str(reget[3][-1]) if str(reget[3]).startswith('0') else str(reget[3])
                time_indexing_month[month] = time_indexing_month.get(month, [])
                if mapping_file[k] not in time_indexing_month[month]:
                    time_indexing_month[month].append(mapping_file[k])
            else:
                time_indexing_month['0'] = time_indexing_month.get('0', [])
                if mapping_file[k] not in time_indexing_month['0']:
                    time_indexing_month['0'].append(mapping_file[k])
        else:
            time_indexing_year['0'] = time_indexing_year.get('0', [])
            if mapping_file[k] not in time_indexing_year['0']:
                time_indexing_year['0'].append(mapping_file[k])
            time_indexing_month['0'] = time_indexing_month.get('0', [])
            if mapping_file[k] not in time_indexing_month['0']:
                time_indexing_month['0'].append(mapping_file[k])

    time_indexing_year = dict(sorted(time_indexing_year.items(), key = lambda x:x[0]))
    time_indexing_month = dict(sorted(time_indexing_month.items(), key = lambda x:x[0]))
    json.dump(time_indexing_year, open(f'{data_path}/index_time_year.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(time_indexing_month, open(f'{data_path}/index_time_month.json', 'w', encoding = 'utf-8'), indent = 4)


def process_data(argv):
    # Data directories
    mapping_corduid2nid = 'mapping_corduid2nid.json'
    records_data_aggregate = 'records_data_aggregate.json'
    index_time_year = 'index_time_year.json'#None #
    index_time_month = 'index_time_month.json'#None #
    time_indexing_year_month(mapping_corduid2nid, records_data_aggregate,
                             index_time_year = index_time_year, index_time_month = index_time_month)

def run_setup():
    app.run(process_data)

#if __name__ == "__main__":
#    run_setup()
