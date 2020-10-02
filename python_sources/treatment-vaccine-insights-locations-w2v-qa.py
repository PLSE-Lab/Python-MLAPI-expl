# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np 

# from scispacy.umls_linking import UmlsEntityLinker
!pip install pycountry
import pycountry
import glob
import json
#nlp = spacy.load("en_ner_bc5cdr_md")
from collections import Counter
# Any results you write to the current directory are saved as output.

def load_metadata(root_path):
    
    metadata_path = f'{root_path}/metadata.csv'
    meta_df = pd.read_csv(metadata_path, dtype={
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str, 
        'doi': str
    })
    return meta_df

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.body_text = []
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.body_text = ' '.join(self.body_text)
            # Extend Here
            #
            #
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def json_to_dataframe(root_path):
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    len(all_json)

    with open(all_json[1]) as file:
        first_entry = json.load(file)
        print(first_entry.keys())
        print(first_entry['metadata'].keys())


    dict_ = {'paper_id': [], 'body_text': []}
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
        content = FileReader(entry)
        dict_['paper_id'].append(content.paper_id)
        dict_['body_text'].append(content.body_text)
    df_covid = pd.DataFrame(dict_, columns=['paper_id','body_text'])
    df_covid = df_covid[df_covid['body_text'] != ""]
    return df_covid

def load_data(root_path):
    meta_df = load_metadata(root_path)
    df_covid = json_to_dataframe(root_path)
    return df_covid, meta_df

root_path = '../input/CORD-19-research-challenge'
df_covid, meta_df = load_data(root_path)

from enum import Enum

class date_cate(Enum):
  BEFORE_COVID = 1
  BEFORE_WORLD_EMERGENCY = 2
  BEFORE_ITALY = 3
  BEFORE_DECLARED_PANDEMIC = 4
  AFTER_ALL = 5

def categorize_date(date):
  if date < '2019-12-12': 
    return date_cate.BEFORE_COVID
  elif date < '2020-01-30': 
    return date_cate.BEFORE_WORLD_EMERGENCY
  elif date < '2020-02-21': 
    return date_cate.BEFORE_ITALY
  elif date < '2020-03-11':
    return date_cate.BEFORE_DECLARED_PANDEMIC
  else : 
    return date_cate.AFTER_ALL

#dates = sorted(meta_df.publish_time[~meta_df.publish_time.isna()])

df_covid['sha'] = df_covid['paper_id']
meta_df_wb = pd.merge(meta_df, df_covid, on = 'sha')
meta_df_wb = meta_df_wb[~meta_df_wb.publish_time.isna()]
meta_df_wb['date_cat'] = meta_df_wb['publish_time'].apply(categorize_date)
