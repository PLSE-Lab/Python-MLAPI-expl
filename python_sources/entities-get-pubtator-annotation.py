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
def get_pubtator_entities(paper_ids, records_pubtator = None):
    # load ids file
    paper_ids = json.load(open(f'{data_path}/{paper_ids}', 'r', encoding = 'utf-8'))
    records = {} if records_pubtator == None else json.load(open(f'{data_path}/{records_pubtator}', 'r', encoding = 'utf-8'))
    # query pubtators
    for cord_uid, ids in tqdm(paper_ids.items(), desc = 'get pubtator annotations'):
        if cord_uid in records:
            if records[cord_uid] == 'pmcid':
                continue
            if ids['pmcid'] != '':
                pmcid = ids['pmcid']
                pubtator = get_json_pubtators(pmcid, id_type = 'pmcids')
                if pubtator != None and pubtator != [] and pubtator[0] != '':
                    json.dump(pubtator, open(f'{pubtator_path}/{cord_uid}.json', 'w', encoding = 'utf-8'))
                    records[cord_uid] = 'pmcid'
                else:
                    continue

        if ids['pmcid'] != '' and ids['pmid'] != '':
            pmcid = ids['pmcid']
            pmid = ids['pmid']
            pubtator = get_json_pubtators(pmcid, id_type = 'pmcids')
            if pubtator != None and pubtator != [] and pubtator[0] != '':
                pubtator = {'cord_uid':cord_uid, 'pmcid':pmcid, 'pubtator':pubtator[0]}
                json.dump(pubtator, open(f'{pubtator_path}/{cord_uid}.json', 'w', encoding = 'utf-8'))
                records[cord_uid] = 'pmcid'
            else:
                pubtator = get_json_pubtators(pmid, id_type = 'pmids')
                if pubtator != None and pubtator != [] and pubtator[0] != '':
                    pubtator = {'cord_uid':cord_uid, 'pmid':pmid, 'pubtator':pubtator[0]}
                    json.dump(pubtator, open(f'{pubtator_path}/{cord_uid}.json', 'w', encoding = 'utf-8'))
                    records[cord_uid] = 'pmid'
        elif ids['pmcid'] != '' and ids['pmid'] == '':
            pmcid = ids['pmcid']
            pubtator = get_json_pubtators(pmcid, id_type = 'pmcids')
            if pubtator != None and pubtator != [] and pubtator[0] != '':
                pubtator = {'cord_uid':cord_uid, 'pmcid':pmcid, 'pubtator':pubtator[0]}
                json.dump(pubtator, open(f'{pubtator_path}/{cord_uid}.json', 'w', encoding = 'utf-8'))
                records[cord_uid] = 'pmcid'
        elif ids['pmcid'] == '' and ids['pmid'] != '':
            pmid = ids['pmid']
            pubtator = get_json_pubtators(pmid, id_type = 'pmids')
            if pubtator != None and pubtator != [] and pubtator[0] != '':
                pubtator = {'cord_uid':cord_uid, 'pmid':pmid, 'pubtator':pubtator[0]}
                json.dump(pubtator, open(f'{pubtator_path}/{cord_uid}.json', 'w', encoding = 'utf-8'))
                records[cord_uid] = 'pmid'
        else: continue
    json.dump(records, open(f'{data_path}/records_pubtator.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    paper_ids = 'paper_ids.json'
    records_pubtator = 'records_pubtator.json'
    if not os.path.exists(pubtator_path): os.makedirs(pubtator_path)
    get_pubtator_entities(paper_ids, records_pubtator = records_pubtator)


def run_setup():
    app.run(get_data)

#if __name__ == "__main__":
#    run_setup()
