# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

training_data = pd.read_csv("../input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
clean_train = []
bag = []
example1 = BeautifulSoup(training_data['review'][0], "lxml")

lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()  
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
                      
lower_case = letters_only.lower()            # Convert to lower case
words = lower_case.split()                   # Split into words
print (words)
                    

# Any results you write to the current directory are saved as output.