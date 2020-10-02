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
def process_pubtator_entities(records_pubtator, records_pub_entity = None):
    # load ids file
    pub_records = json.load(open(f'{data_path}/{records_pubtator}', 'r', encoding = 'utf-8'))
    ent_records = {} if records_pub_entity == None else json.load(open(f'{data_path}/{records_pub_entity}', 'r', encoding = 'utf-8'))
    # process entities in pubtators
    for pid, source in tqdm(pub_records.items(), desc = 'parsing pubtator entity annotations'):
        if pid in ent_records: continue
        if source == 'pmcid':
            #print(pid)
            file = json.load(open(f'{pubtator_path}/{pid}.json', 'r', encoding = 'utf-8'))
            ents = pubtator_biocjson_paser(file['pubtator'])
            entities = {}
            for ent in ents:
                if ent['section'] == 'front|title':
                    entities['title'] = {'text':ent['text'],'ents':ent['ents']}
                if ent['section'] == 'abstract|abstract':
                    entities.setdefault('abstract', []).append({'text':ent['text'],'ents':ent['ents']})

            if 'abstract' in entities:
                if len(entities['abstract']) > 1:
                    abs_ents = entities['abstract']
                    text_string = ''
                    offset = [0]
                    for i in range(1, len(abs_ents)):
                        text_string += abs_ents[i-1]['text']
                        offset.append(len(text_string) + i )
                    text = ' '.join([ent['text'] for ent in abs_ents])
                    ents_list = []
                    for sec_idx, sec in enumerate(abs_ents):
                        for ent in sec['ents']:
                            if int(ent[0]) < 0: continue
                            start, end = int(ent[0])+offset[sec_idx], int(ent[1])+offset[sec_idx]
                            assert text[start:end] == ent[2]
                            ents_list.append([start, end, ent[2], ent[3], ent[4]])
                    entities['abstract'] = {'text':text, 'ents':ents_list}
                else:
                    entities['abstract'] = entities['abstract'][0]
            else:
                entities['abstract'] = {'text':'', 'ents':[]}
            json.dump(entities, open(f'{entity_path}/{pid}.json', 'w', encoding = 'utf-8'), indent = 4)
            ent_records[pid] = source
        
        if source == 'pmid':
            file = json.load(open(f'{pubtator_path}/{pid}.json', 'r', encoding = 'utf-8'))
            ents = pubtator_biocjson_paser(file['pubtator'])
            entities = {}
            for ent in ents:
                if ent['section'] == 'title|title':
                    entities['title'] = {'text':ent['text'],'ents':ent['ents']}
                if ent['section'] == 'abstract|abstract':
                    entities['abstract'] = {'text':ent['text'],'ents':ent['ents']}
            json.dump(entities, open(f'{entity_path}/{pid}.json', 'w', encoding = 'utf-8'), indent = 4)
            ent_records[pid] = source
        
        if source == 'post':
            with open(f'{pubtator_path}/{pid}.txt', 'r', encoding = 'utf-8') as f:
                file = [line.strip() for line in f]
            entities = pubtator_pubtator_paser(file)
            json.dump(entities, open(f'{entity_path}/{pid}.json', 'w', encoding = 'utf-8'), indent = 4)
            ent_records[pid] = source

    json.dump(ent_records, open(f'{data_path}/records_pub_entity.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    records_pubtator = 'records_pubtator.json'
    records_pub_entity = 'records_pub_entity.json'
    if not os.path.exists(entity_path): os.makedirs(entity_path)
    process_pubtator_entities(records_pubtator, records_pub_entity = records_pub_entity)


def run_setup():
    app.run(get_data)

#if __name__ == "__main__":
#    run_setup()
