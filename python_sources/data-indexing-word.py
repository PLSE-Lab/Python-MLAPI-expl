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
from spacy.lang.en import English

from absl import app

nlp = English()
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'

# function for process  kaggle input data
def word_indexing_tiabs(mapping_file, aggregate_records, index_word_title = None,
                        index_word_abstract = None, paper_word_counts = None,
                        index_word_table = None, paper_tables = None, records_index_word = None):
    mapping_file =  json.load(open(f'{data_path}/{mapping_file}', 'r', encoding = 'utf-8'))
    aggregate_records = json.load(open(f'{data_path}/{aggregate_records}', 'r', encoding = 'utf-8'))
    
    index_word_title = {} if index_word_title == None else json.load(open(f'{data_path}/{index_word_title}', 'r', encoding = 'utf-8'))
    index_word_abstract = {} if index_word_abstract == None else json.load(open(f'{data_path}/{index_word_abstract}', 'r', encoding = 'utf-8'))
    paper_word_counts = {} if paper_word_counts == None else json.load(open(f'{data_path}/{paper_word_counts}', 'r', encoding = 'utf-8'))
    
    index_word_table = {} if index_word_table == None else json.load(open(f'{data_path}/{index_word_table}', 'r', encoding = 'utf-8'))
    paper_tables = {} if paper_tables == None else json.load(open(f'{data_path}/{paper_tables}', 'r', encoding = 'utf-8'))
    
    records_index_word = {} if records_index_word == None else json.load(open(f'{data_path}/{records_index_word}', 'r', encoding = 'utf-8'))
    # start indexing
    for k, v in tqdm(aggregate_records.items(), desc = 'words indexing for title, abstract and table heading'):
        if k in records_index_word: continue
        file = json.load(open(f'{json_path}/{k}.json', 'r', encoding = 'utf-8'))
        title = file['title']['text']
        abstract = file['abstract']['text']
        wc = {}
        # process title
        if title != '':
            doc = nlp(title)
            for token in doc:
                if token.is_stop or token.is_punct or token.is_digit: continue
                if token.lemma_.lower() in index_word_title:
                    if mapping_file[k] not in index_word_title[token.lemma_.lower()]:
                        index_word_title[token.lemma_.lower()].append(mapping_file[k])
                else:
                    index_word_title[token.lemma_.lower()] = [mapping_file[k]]
                
                wc[token.lemma_.lower()] = wc.get(token.lemma_.lower(), 0) + 1
        # process abstract
        if abstract != '':
            doc = nlp(abstract)
            for token in doc:
                if token.is_stop or token.is_punct or token.is_digit: continue
                if token.lemma_.lower() in index_word_abstract:
                    if mapping_file[k] not in index_word_abstract[token.lemma_.lower()]:
                        index_word_abstract[token.lemma_.lower()].append(mapping_file[k])
                else:
                    index_word_abstract[token.lemma_.lower()] = [mapping_file[k]]
                
                wc[token.lemma_.lower()] = wc.get(token.lemma_.lower(), 0) + 1
        # process tables
        if file['tables'] != []:
            for tid, table in enumerate(file['tables']):
                if table['text'] != '':
                    paper_tables['heading'] = paper_tables.get('heading', {})
                    if mapping_file[k] in paper_tables['heading']:
                        if tid not in paper_tables['heading'][mapping_file[k]]:
                            paper_tables['heading'][mapping_file[k]].append(tid)
                    else:
                        paper_tables['heading'][mapping_file[k]] = [tid]
                    
                    doc = nlp(table['text'])
                    for token in doc:
                        if token.is_stop or token.is_punct or token.is_digit: continue
                        if token.lemma_.lower() in index_word_table:
                            if mapping_file[k] in index_word_table[token.lemma_.lower()]:
                                if tid not in index_word_table[token.lemma_.lower()][mapping_file[k]]:
                                    index_word_table[token.lemma_.lower()][mapping_file[k]].append(tid)
                            else:
                                index_word_table[token.lemma_.lower()][mapping_file[k]] = [tid]
                        else:
                            index_word_table[token.lemma_.lower()] = {mapping_file[k]:[tid]}
                        
                        wc[token.lemma_.lower()] = wc.get(token.lemma_.lower(), 0) + 1
                
                if table['html'] != '':
                    paper_tables['html'] = paper_tables.get('html', {})
                    if mapping_file[k] in paper_tables['html']:
                        if tid not in paper_tables['html'][mapping_file[k]]:
                            paper_tables['html'][mapping_file[k]].append(tid)
                    else:
                        paper_tables['html'][mapping_file[k]] = [tid]
        
        paper_word_counts[mapping_file[k]] = wc
        records_index_word[k] = v

    json.dump(index_word_title, open(f'{data_path}/index_word_title.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(index_word_abstract, open(f'{data_path}/index_word_abstract.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(paper_word_counts, open(f'{data_path}/paper_word_counts.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(index_word_table, open(f'{data_path}/index_word_table.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(paper_tables, open(f'{data_path}/paper_tables.json', 'w', encoding = 'utf-8'), indent = 4)
    json.dump(records_index_word, open(f'{data_path}/records_index_word.json', 'w', encoding = 'utf-8'), indent = 4)


def process_data(argv):
    # Data directories
    mapping_file = 'mapping_corduid2nid.json'
    aggregate_records = 'records_data_aggregate.json'
    index_word_title = None #'index_word_title.json' #
    index_word_abstract = None #'index_word_abstract.json' #
    paper_word_counts = None # 'paper_word_counts.json' #
    index_word_table = None # 'index_word_table.json' #
    paper_tables = None # 'paper_tables.json' # 
    records_index_word = None #'records_index_word.json' #
    word_indexing_tiabs(mapping_file, aggregate_records, index_word_title = index_word_title,
                        index_word_abstract = index_word_abstract, paper_word_counts = paper_word_counts,
                        index_word_table = index_word_table, paper_tables = paper_tables,
                        records_index_word = records_index_word)


def run_setup():
    app.run(process_data)

#if __name__ == "__main__":
#   run_setup()
