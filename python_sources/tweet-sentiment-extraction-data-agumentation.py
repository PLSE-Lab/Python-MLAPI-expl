#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import string, os
from tqdm import tqdm
#from googletrans import Translator
#from textblob import TextBlob
import time
import requests


#Function to automate translation Yandex taslate
def traslate(text, key, lang = 'en'):
    
    url_yandex ="https://translate.yandex.net/api/v1.5/tr.json/translate?key=%s&text=%s&lang=%s" % (key,text,lang)
    time.sleep(0.1)
    response = requests.get(url_yandex, timeout=None)
    response_data = eval(response.content.decode('utf-8'))
    lb = response_data['text'][0]
    return lb
 
#Test function
text = 'Hola Mundo!'
key = 'trnsl.1.1.20200514T223841Z.e8be2cb3b512ab50.6d2435315749e911978fe98e0a00417ddafbdcb3'
traslate(text, key)
#'Hello World!'

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df = df[df.sentiment != 'neutral'].reset_index(drop = True)


# In[ ]:


#df = df.iloc[2500*i:2500*(i+1), :]


# In[ ]:


df.count()


# In[ ]:


def Trans(row):
    try:
        D = row[1].to_dict()
        text = D['text']

        selected_text = D['selected_text'].lower()

        text1 = traslate(text, key , lang = 'la')
        
        #text2 = traslate(text1, key, lang = 'la')

        text3 = traslate(text1, key)

        D['translated_text'] = text3

        if(selected_text in text3.lower()):
            if(text != text3):
                return D
    except:
        pass


list_dictonaries = []

counter = 0

for row in tqdm(df.iterrows()):
    counter = counter+1
    K = Trans(row)
    if(K is not None):
        list_dictonaries.append(Trans(row))
        print(counter)
    #print(row)
    #break

    


# In[ ]:


import pickle

file = open(f"file{i}.pkl", "wb")
pickle.dump(list_dictonaries,file)
file.close()


# In[ ]:


A = [i for i in list_dictonaries if i is not None]


# In[ ]:


df_a = pd.DataFrame(A)


# In[ ]:


df_a.to_csv("translated_file.csv", index = False)

