# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import csv
import gzip
import json
from tqdm import tqdm
from urllib.request import urlopen

from absl import app

from utils import *

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'
url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"

# function for process  kaggle input data
def get_doi_pmcid_pmid(metadata, paper_ids = None):
    ids_mapping = {'doi': {}, 'pmcid': {}, 'pmid': {}}
    with gzip.open(urlopen(url), 'rt') as fin:
        csv_reader = csv.reader(fin)
        header = next(csv_reader)
        for num, line in tqdm(enumerate(csv_reader), desc = 'geting ids'):
            doi = line[7].strip()
            pmcid = line[8].strip()
            pmid = line[9].strip()
            if doi != '' and (doi not in ids_mapping['doi']):
                ids_mapping['doi'][doi] = {'pmcid':pmcid, 'pmid':pmid} 
            if pmcid != '' and (pmcid not in ids_mapping['pmcid']):
                ids_mapping['pmcid'][pmcid] = {'doi':doi, 'pmid':pmid} 
            if pmid != '' and (pmid not in ids_mapping['pmid']):
                ids_mapping['pmid'][pmid] = {'doi':doi, 'pmcid':pmcid} 
    
    paper_ids = {} if paper_ids == None else json.load(open(f'{data_path}/{paper_ids}', 'r', encoding = 'utf-8'))
    with open(f'{input_data_path}/{metadata}', 'r', encoding = 'utf-8') as fin:
        csv_reader = csv.reader(fin)
        next(csv_reader)
        for line in tqdm(csv_reader, desc='agrregate ids'):
            cord_uid = line[0].strip()
            if cord_uid in paper_ids: continue
            paper_ids[cord_uid] = {'doi':line[4].strip(),
                                   'pmcid':line[5].strip(),
                                   'pmid':line[6].strip()}
    
    for k, v in tqdm(paper_ids.items(), desc='check by pmcid'):
        if v['pmcid'] != '':
            if v['pmcid'] in ids_mapping['pmcid']:
                if v['doi'] == '': v['doi'] = ids_mapping['pmcid'][v['pmcid']]['doi']
                if v['pmid'] == '': v['pmid'] = ids_mapping['pmcid'][v['pmcid']]['pmid']
    
    for k, v in tqdm(paper_ids.items(), desc='check by pmid'):
        if v['pmid'] != '':
            if v['pmid'] in ids_mapping['pmid']:
                if v['doi'] == '': v['doi'] = ids_mapping['pmid'][v['pmid']]['doi']
                if v['pmcid'] == '': v['pmcid'] = ids_mapping['pmid'][v['pmid']]['pmcid']
    
    for k, v in tqdm(paper_ids.items(), desc='check by doi'):
        if v['doi'] != '':
            if v['doi'] in ids_mapping['doi']:
                if v['pmcid'] == '': v['pmcid'] = ids_mapping['doi'][v['doi']]['pmcid']
                if v['pmid'] == '': v['pmid'] = ids_mapping['doi'][v['doi']]['pmid']

    json.dump(paper_ids, open(f'{data_path}/paper_ids.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    metadata = 'metadata.csv'
    paper_ids = 'paper_ids.json'
    if not os.path.exists(data_path): os.makedirs(data_path)
    get_doi_pmcid_pmid(metadata, paper_ids = paper_ids)


def run_setup():
    app.run(get_data)

#if __name__ == "__main__":
#    run_setup()
