# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import random as rd # randomness
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from difflib import SequenceMatcher

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

def simple_rdany(input_text):
    data = pd.read_csv('../input/rdany_conversations_2016-01-24.csv')
    
    highest_similarity = 0.0
    highest_similarity_index = None
    for index, message in enumerate(data.as_matrix()):
        if message[3] != "human":
            continue
        if similar(message[4], input_text) > highest_similarity:
            highest_similarity = similar(message[4], input_text)
            highest_similarity_index = index
    
    similar_human = data.as_matrix()[highest_similarity_index][4]
    
    print ("Human: {0} (similar to \"{1}\")".format(input_text, similar_human))
    
    current_index = highest_similarity_index
    while 1:
        current_index = current_index + 1
        if data.as_matrix()[current_index][3] == "robot":
            break
    
    print ("rDany: {0}\n".format(data["text"].as_matrix()[current_index]))

simple_rdany("Hello!")
simple_rdany("how r you")
simple_rdany("What are you doing now?")
simple_rdany("Hola!!!")
simple_rdany("What is the color of the sky?")


