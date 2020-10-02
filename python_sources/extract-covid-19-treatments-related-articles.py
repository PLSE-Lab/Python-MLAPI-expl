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

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
import re
# load the meta data from the CSV file using 3 columns (abstract, title, authors),
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])
print (df.shape)
#drop duplicate abstracts
df = df.drop_duplicates(subset='abstract', keep="first")
#drop NANs
df=df.dropna()
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()
#show 5 lines of the new dataframe
print (df.shape)
print(df.head())

# Any results you write to the current directory are saved as output.

search_list = ['stem cells therapy','chloroquine','hydroxychloroquine','umifenovir','lopinavir-ritonavir','remdesivir',
               'Plasmatreatment','Favipiravir','Oseltamivir','Methylprednisolone','Baloxavir Marboxil'
               ,'Thalidomide','Darunavir/cobicistat','Thymosin','PD-1 blocking antibody',
               'Tocilizumab','Intravenous Immunoglobulin','Ozonated autohemotherapy',
               'Type 1 Interferon injection','Interferon nebulization']

df['Treatment'] = df.abstract.str.extract('({0})'.format('|'.join(search_list)), flags=re.IGNORECASE)
result = df[~pd.isna(df.Treatment)]
result.to_csv('out1.csv')
print(result)
