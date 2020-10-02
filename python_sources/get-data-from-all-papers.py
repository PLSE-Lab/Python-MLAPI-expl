# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# List of Papers
papers = []

# Paper Fields
# id
# title
# abstract.text
# body.text
paper = {}

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print("Working on " + dirname)
    for filename in filenames:
        
        # Search the directory for files and extract `Paper Fields`
        #print(filename)
        if ".json" in filename:
            data = open(os.path.join(dirname, filename), 'r').read()
            paper_dict = json.loads(data)

            paper['paper_id'] = paper_dict['paper_id']
            paper['title'] = paper_dict['metadata']['title']

            paper['abstract_text'] = ""
            if len(paper_dict['abstract']) > 0:
                paper_abstract_text = ""
                for abstract_text in paper_dict['abstract']:
                    paper_abstract_text += abstract_text['text']
                paper['abstract_text'] = paper_abstract_text

            paper['body_text'] = ""
            if len(paper_dict['body_text']) > 0:
                paper_body_text = ""
                for body_text in paper_dict['body_text']:
                    paper_body_text += body_text['text']
                paper['body_text'] = paper_body_text

            papers.append(paper)
 

# Total number of papers we extracted data
print(len(papers))
        

# Any results you write to the current directory are saved as output.
