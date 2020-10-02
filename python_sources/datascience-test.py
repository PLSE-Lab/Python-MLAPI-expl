# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


from datascience import *
import numpy as np
path_data = '../input/'
import matplotlib
# %matplotlib inline
matplotlib.use('Agg', warn=False)

import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from urllib.request import urlopen 
import re

def read_url(url): 
    return str(urlopen(url).read())
    #return re.sub('\\s+', ' ', urlopen(url).read().decode())
    
def read_file(path):
    fd = open(path, "r")
    contents = fd.read()
    fd.close()
    return contents

HUCK = "../input/huck_finn.txt"
huck_finn_url = 'https://www.inferentialthinking.com/data/huck_finn.txt'
#huck_finn_text = read_url(huck_finn_url)
huck_finn_text     = read_file(HUCK)
huck_finn_chapters = huck_finn_text.split('CHAPTER ')[44:]

t = Table().with_column('Chapters', huck_finn_chapters)
#print(t)

counts = Table().with_columns([
        'Jim', np.char.count(huck_finn_chapters, 'Jim'),
        'Tom', np.char.count(huck_finn_chapters, 'Tom'),
        'Huck', np.char.count(huck_finn_chapters, 'Huck')
    ])

# Plot the cumulative counts:
# how many times in Chapter 1, how many times in Chapters 1 and 2, and so on.

cum_counts = counts.cumsum().with_column('Chapter', np.arange(1, 44, 1))
cum_counts.plot(column_for_xticks=3)
plots.title('Cumulative Number of Times Each Name Appears', y=1.08)
plots.show()
plots.savefig('./graph.png')

print("done")
print("input", os.listdir("../input"))
print("files", os.listdir("./"))

# Any results you write to the current directory are saved as output.