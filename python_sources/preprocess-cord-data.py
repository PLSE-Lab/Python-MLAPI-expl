# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:35:08 2020

@author: shubotian
"""

# Loading packages
import os
import csv
import json
from tqdm import tqdm

from absl import app
from absl import logging

logging.set_verbosity(logging.INFO)

input_data_path = 'kaggle/input/CORD-19-research-challenge'
data_path = 'data'
cord_path = 'cord_data'
pubtator_path = 'pubtators'
entity_path = 'entities'
json_path = 'json_files'

# function for process  kaggle input data
def process_cord_data(metadata, paper_ids, mapping_corduid2nid = None, cord_data = None, include_fulltext = True):
    paper_ids = json.load(open(f'{data_path}/{paper_ids}', 'r', encoding = 'utf-8'))
    mapping_corduid2nid = {} if mapping_corduid2nid == None else json.load(open(f'{data_path}/{mapping_corduid2nid}', 'r', encoding = 'utf-8'))
    #input_data = []
    if cord_data == None:
        fcsv = open(f'{data_path}/cord_data.csv', 'w', encoding = 'utf-8')
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(['cord_uid', 'title', 'abstract', 'authors',
                             'journal', 'publish_time', 'url'])
    else:
        fcsv = open(f'{data_path}/{cord_data}', 'a', encoding = 'utf-8')
        csv_writer = csv.writer(fcsv)

    with open(f'{input_data_path}/{metadata}', 'r', encoding = 'utf-8') as fin:
        csv_reader = csv.reader(fin)
        header = next(csv_reader)
        line_length = len(header)
        for num, line in tqdm(enumerate(csv_reader), desc = 'processing cord-19 dataset'):
            cord_uid = line[0].strip()
            if cord_uid in mapping_corduid2nid: continue
            if len(line) != line_length:
                logging.info(f'ERROR metacsv line no {num}')
            else:
                mapping_corduid2nid[cord_uid] = len(mapping_corduid2nid)
                line_data = {'cord_uid':cord_uid,
                             'source':line[2].strip(),
                             'journal':line[11].strip(),
                             'mag_id':line[12].strip(),
                             'who_covidence_id':line[13].strip(),
                             'arxiv_id':line[14].strip(),
                             'doi':paper_ids[cord_uid]['doi'],
                             'pmcid':paper_ids[cord_uid]['pmcid'],
                             'pmid':paper_ids[cord_uid]['pmid'],
                             'authors':line[10].strip(),
                             'publish_time':line[9].strip(),
                             'title':line[3].strip(),
                             'abstract':[line[8].strip()]}
                # full text
                if include_fulltext:
                    if line[15].strip() != '':
                        jsonfile = f'{input_data_path}/{line[15].strip().split(";")[0]}'
                    elif line[16].strip() != '':
                        jsonfile = f'{input_data_path}/{line[16].strip().split(";")[0]}'
                    else:
                        line_data['body_text'] = []
                        line_data['figures'] = []
                        line_data['tables'] = []
                        line_data['url'] = line[17].strip()
                        #input_data.append(line_data)
                        csv_writer.writerow([cord_uid, line_data['title'], ' '.join(line_data['abstract']),
                                             line_data['authors'], line_data['journal'],
                                             line_data['publish_time'], line_data['url']])
                        json.dump(line_data,
                                  open(f'{cord_path}/{cord_uid}.json', 'w', encoding = 'utf-8'), indent = 4)
                        continue
                    jsonfile = json.load(open(jsonfile, 'r', encoding = 'utf8'))
                    # title
                    title = jsonfile['metadata']['title']
                    # abstract
                    if 'abstract' in jsonfile:
                        abstract = [text['text'] for text in jsonfile['abstract']]
                    else: abstract = []
                    # body
                    if 'body_text' in jsonfile:
                        body_text = []
                        current_section = ''
                        current_text = []
                        for text in jsonfile['body_text']:
                            section = text['section']
                            if section != current_section:
                                if current_text != []:
                                    body_text.append({'section':current_section, 'text':current_text})
                                elif current_section != '':
                                    body_text.append({'section':current_section, 'text':current_text})
                                current_section = section
                                current_text = [text['text']] if text['text'] != '' else []
                            elif text['text'] != '':
                                current_text.append(text['text'])
                        if current_section != '' or current_text != []:
                            body_text.append({'section':current_section, 'text':current_text})
                    else: body_text = []
                    # figures and tables
                    if jsonfile['ref_entries'] != None:
                        figures = [ref['text'] for ref in jsonfile['ref_entries'].values()
                                   if ref['type'] == 'figure' and ref['text'] != '']
                        tables = [{'text': ref['text'], 'html':ref.get('html', '')}
                                  for ref in jsonfile['ref_entries'].values()
                                  if ref['type'] == 'table' and (ref['text'] != ''
                                                                 or ref.get('html', '') != '')]
                    else:
                        figures = []
                        tables = []
                    
                    if line_data['title'] == '' and title != '': line_data['title'] = title
                    if line_data['abstract'] == [] and abstract != []: line_data['abstract'] = abstract
                    line_data['body_text'] = body_text
                    line_data['figures'] = figures
                    line_data['tables'] = tables
                line_data['url'] = line[17].strip()
                #input_data.append(line_data)
                csv_writer.writerow([cord_uid, line_data['title'], ' '.join(line_data['abstract']),
                                     line_data['authors'], line_data['journal'],
                                     line_data['publish_time'], line_data['url']])
                json.dump(line_data,
                          open(f'{cord_path}/{cord_uid}.json', 'w', encoding = 'utf-8'), indent = 4)
    fcsv.close()
    
    json.dump(mapping_corduid2nid, open(f'{data_path}/mapping_corduid2nid.json', 'w', encoding = 'utf-8'), indent = 4)


def process_data(argv):
    # Data directories
    metadata = 'metadata.csv'
    paper_ids = 'paper_ids.json'
    mapping_corduid2nid = 'mapping_corduid2nid.json'
    cord_data = 'cord_data.csv'
    include_fulltext = True
    if not os.path.exists(cord_path): os.makedirs(cord_path)
    process_cord_data(metadata, paper_ids, mapping_corduid2nid = mapping_corduid2nid,
                      cord_data = cord_data, include_fulltext = include_fulltext)

def run_setup():
    app.run(process_data)

#if __name__ == "__main__":
#    run_setup()
