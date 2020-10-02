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
from itertools import combinations
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

from utils import *

from absl import app
from absl import logging

#logging.get_absl_handler().use_absl_log_file('absl_logging', 'befreeout')
#logging.set_verbosity(logging.INFO)

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'


# function for process  kaggle input data
def get_nodes_relations(mapping_file, aggregate_records, entity_lists, 
                        sentences = None, mapping_sents = None, index_word_sents = None,
                        entity_nodes = None, entity_relations = None, records_nodes_relations = None):
    # load ids file
    mapping_file = json.load(open(f'{data_path}/{mapping_file}', 'r', encoding = 'utf-8'))
    agg_records = json.load(open(f'{data_path}/{aggregate_records}', 'r', encoding = 'utf-8'))
    entity_lists = json.load(open(f'{data_path}/{entity_lists}', 'r', encoding = 'utf-8'))
    
    sentences = {} if sentences == None else json.load(open(f'{data_path}/{sentences}', 'r', encoding = 'utf-8'))
    sents_mapping = {} if mapping_sents == None else json.load(open(f'{data_path}/{mapping_sents}', 'r', encoding = 'utf-8'))
    index_sents = {} if index_word_sents == None else json.load(open(f'{data_path}/{index_word_sents}', 'r', encoding = 'utf-8'))
    
    entity_nodes = {} if entity_nodes == None else json.load(open(f'{data_path}/{entity_nodes}', 'r', encoding = 'utf-8'))
    entity_relations = {} if entity_relations == None else json.load(open(f'{data_path}/{entity_relations}', 'r', encoding = 'utf-8'))
    re_records = {} if records_nodes_relations == None else json.load(open(f'{data_path}/{records_nodes_relations}', 'r', encoding = 'utf-8'))
    # query pubtators
    for cord_uid, nid in tqdm(agg_records.items(), desc="processing nodes and relations"):
        if cord_uid in re_records: continue
        file =  json.load(open(f'{json_path}/{cord_uid}.json', 'r', encoding = 'utf-8'))
        # combine entities from different sources
        file['title']['ents'] = combine_entities(file['title'])
        file['abstract']['ents'] = combine_entities(file['abstract'])
        nodes = {}
        relations = {}
        # get nodes and relations in title
        if file['title']['ents'] != []:
            doc = get_section_doc(file['title'], entity_lists)
            sent_idx = f'{mapping_file[cord_uid]}|t|0'
            sents_mapping[sent_idx] = len(sents_mapping)
            sentences[sent_idx] = doc.text
            #word index
            for token in doc:
                if token.is_stop or token.is_punct or token.is_digit: continue
                if token.lemma_.lower() in index_sents:
                    if sents_mapping[sent_idx] not in index_sents[token.lemma_.lower()]:
                        index_sents[token.lemma_.lower()].append(sents_mapping[sent_idx])
                else:
                    index_sents[token.lemma_.lower()] = [sents_mapping[sent_idx]]
            #entity nodes and relations process
            ents = list(doc.ents)
            ents.sort(key = lambda x:f'{x.label_} {x.text}')
            # get entity nodes
            for ent in ents:
                if ent.label_ in nodes:
                    nodes[ent.label_][ent.text] = nodes[ent.label_].get(ent.text, 0) + 1
                else:
                    nodes[ent.label_] = {ent.text: 1}
            
            # get relations
            if len(ents) > 1:
                for comb in combinations(ents, 2):
                    rel_type = f'{comb[0].label_}|{comb[1].label_}'
                    pair = f'{comb[0].text}|{comb[1].text}'
                    if rel_type in relations:
                        if pair in relations[rel_type]:
                            relations[rel_type][pair]['count'] += 1
                            if sent_idx not in relations[rel_type][pair]['sents']:
                                relations[rel_type][pair]['sents'].append(sent_idx)
                        else:
                            relations[rel_type][pair] = {'count': 1, 'sents':[sent_idx]}
                    else:
                        relations[rel_type] = {pair: {'count': 1, 'sents':[sent_idx]}}

        elif file['title']['text'] != '':
            doc = nlp(file['title']['text'])
            sent_idx = f'{mapping_file[cord_uid]}|t|0'
            sents_mapping[sent_idx] = len(sents_mapping)
            sentences[sent_idx] = doc.text
            #word index
            for token in doc:
                if token.is_stop or token.is_punct or token.is_digit: continue
                if token.lemma_.lower() in index_sents:
                    if sents_mapping[sent_idx] not in index_sents[token.lemma_.lower()]:
                        index_sents[token.lemma_.lower()].append(sents_mapping[sent_idx])
                else:
                    index_sents[token.lemma_.lower()] = [sents_mapping[sent_idx]]
        
        # get nodes and relations in abstract
        if file['abstract']['ents'] != []:
            doc = get_section_doc(file['abstract'], entity_lists)
            for sent_id, sent in enumerate(doc.sents):
                sent_idx = f'{mapping_file[cord_uid]}|a|{sent_id}'
                sents_mapping[sent_idx] = len(sents_mapping)
                sentences[sent_idx] = sent.text
                #word index
                for token in sent:
                    if token.is_stop or token.is_punct or token.is_digit: continue
                    if token.lemma_.lower() in index_sents:
                        if sents_mapping[sent_idx] not in index_sents[token.lemma_.lower()]:
                            index_sents[token.lemma_.lower()].append(sents_mapping[sent_idx])
                    else:
                        index_sents[token.lemma_.lower()] = [sents_mapping[sent_idx]]
                # process nodes and relations
                if sent.ents == []: continue
                ents = sent.ents
                ents.sort(key = lambda x:f'{x.label_} {x.text}')
                # get entity nodes
                for ent in ents:
                    if ent.label_ in nodes:
                        nodes[ent.label_][ent.text] = nodes[ent.label_].get(ent.text, 0) + 1
                    else:
                        nodes[ent.label_] = {ent.text: 1}
                
                # get relations
                if len(ents) > 1:
                    for comb in combinations(ents, 2):
                        rel_type = f'{comb[0].label_}|{comb[1].label_}'
                        pair = f'{comb[0].text}|{comb[1].text}'
                        if rel_type in relations:
                            if pair in relations[rel_type]:
                                relations[rel_type][pair]['count'] += 1
                                if sent_idx not in relations[rel_type][pair]['sents']:
                                    relations[rel_type][pair]['sents'].append(sent_idx)
                            else:
                                relations[rel_type][pair] = {'count': 1, 'sents':[sent_idx]}
                        else:
                            relations[rel_type] = {pair: {'count': 1, 'sents':[sent_idx]}}

        elif file['abstract']['text'] != '':
            doc = nlp(file['abstract']['text'])
            for sent_id, sent in enumerate(doc.sents):
                sent_idx = f'{mapping_file[cord_uid]}|a|{sent_id}'
                sents_mapping[sent_idx] = len(sents_mapping)
                sentences[sent_idx] = sent.text
                #word index
                for token in sent:
                    if token.is_stop or token.is_punct or token.is_digit: continue
                    if token.lemma_.lower() in index_sents:
                        if sents_mapping[sent_idx] not in index_sents[token.lemma_.lower()]:
                            index_sents[token.lemma_.lower()].append(sents_mapping[sent_idx])
                    else:
                        index_sents[token.lemma_.lower()] = [sents_mapping[sent_idx]]
        
        entity_nodes[mapping_file[cord_uid]] = nodes
        entity_relations[mapping_file[cord_uid]] = relations
        # keep the records
        re_records[cord_uid] = nid
    
    json.dump(sentences, open(f'{data_path}/sentences.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(sents_mapping, open(f'{data_path}/mapping_sents2nid.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(index_sents, open(f'{data_path}/index_word_sents.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(entity_nodes, open(f'{data_path}/entity_nodes.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(entity_relations, open(f'{data_path}/entity_relations.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(re_records, open(f'{data_path}/records_nodes_relations.json', 'w', encoding = 'utf-8'), indent = 4)


def get_data(argv):
    # Data directories
    mapping_file = 'mapping_corduid2nid.json'
    aggregate_records = 'records_data_aggregate.json'
    entity_lists = 'entity_lists.json'
    sentences = None # 'sentences.json'
    mapping_sents = None #'mapping_sents2nid.json'
    index_word_sents = None # 'index_word_sents.json'
    entity_nodes = None # 'entity_nodes.json'
    entity_relations = None # 'entity_relations.json'
    records_nodes_relations = None # 'records_nodes_relations.json'
    get_nodes_relations(mapping_file, aggregate_records, entity_lists, 
                        sentences = sentences, mapping_sents = mapping_sents, index_word_sents = index_word_sents,
                        entity_nodes = entity_nodes, entity_relations = entity_relations,
                        records_nodes_relations = records_nodes_relations)


def run_setup():
    app.run(get_data)

#if __name__ == "__main__":
#    run_setup()
