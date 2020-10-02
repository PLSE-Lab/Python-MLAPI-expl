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

# Any results you write to the current directory are saved as output.

import re as clear

class process():
    def master_clean_text(text):
            #clean up all the html tags
            text = clear.sub(r'<.*?>','',text)
            #remove the unwanted punctation chars

            text = clear.sub(r"\\","",text)
            text = clear.sub(r"\'","",text)
            text = clear.sub(r"\"","",text)

            # coversion to lowercase to remove complexity
            text = text.strip().lower()

            #removing unwanted expressions

            unwanted = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

            convert = dict((c," ") for c in unwanted)

            # str.maketrans() --->> creates a one to one mapping of a character to its translation/replacement.
            mapping_trans = str.maketrans(convert)

            text = text.translate(mapping_trans)

            return text
    #master_clean_text("<a> Say youre scrapping a text from you'r website !! WEll it might be swap CASE or unevened you wanna remove all the punctation's into separate WOrd !!!!</a>").split()

