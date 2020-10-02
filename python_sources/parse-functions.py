# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import json

def return_corona_df(json_filenames, df, source):

    for file_name in json_filenames:

        row = {"doc_id": None, "source": None, "title": None,
              "abstract": None, "text_body": None}

        with open(file_name) as json_data:
            data = json.load(json_data)

            row['doc_id'] = data['paper_id']
            row['title'] = data['metadata']['title']

            # Now need all of abstract. Put it all in 
            # a list then use str.join() to split it
            # into paragraphs. 

            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]
            abstract = "\n ".join(abstract_list)

            row['abstract'] = abstract

            # And lastly the body of the text. For some reason I am getting an index error
            # In one of the Json files, so rather than have it wrapped in a lovely list
            # comprehension I've had to use a for loop like a neanderthal. 
            
            # Needless to say this bug will be revisited and conquered. 
            
            body_list = []
            for _ in range(len(data['body_text'])):
                try:
                    body_list.append(data['body_text'][_]['text'])
                except:
                    pass

            body = "\n ".join(body_list)
            
            row['text_body'] = body
            
            # Now just add to the dataframe. 
            
            if source == 'b':
                row['source'] = "biorxiv_medrxiv"
            elif source == "c":
                row['source'] = "common_use_sub"
            elif source == "n":
                row['source'] = "non_common_use"
            elif source == "p":
                row['source'] = "pmc_custom_license"
            
            df = df.append(row, ignore_index=True)
    
    return df
    