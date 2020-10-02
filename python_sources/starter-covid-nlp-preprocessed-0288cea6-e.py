#!/usr/bin/env python
# coding: utf-8

# Check the [preprocessing kernel](https://www.kaggle.com/donkeys/preprocess-input-docs-from-apr-17-upload-dataset/edit) that produces this for the details on how this is built.
# 
# Check the [Topic Models and Transformer Summaries kernel](https://www.kaggle.com/donkeys/topics-and-summaries-lda-and-transformers) for an example of bigger use case.
# 
# The original input the has been preprocessed is from the [COVID19 NLP dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
# 
# Following is a short example on how to load the data.

# The is under "output" directory. What a name.. There are two txt files to help find words for cleanup. Then two directories for differently processed documents. Mostly you are interested in these directories unless you want to finetune the preprocessing kernel.

# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/')


# Under "paragraphs" there are the documents for the four input sources used in the COVID19 NLP dataset. Files under this directory are JSON files with each document containing paragraphs in a similar set as in the original dataset. So each document split into multiple parts.

# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/paragraphs')


# Under "whole" are all the documents in one big text file per document. Not splitting or anything.

# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/whole')


# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/whole/biorxiv_medrxiv | head -n 10')


# In[ ]:


with open("/kaggle/input/covid-nlp-preprocess/output/whole/biorxiv_medrxiv/006df1a5284369a9e2ff2dc7ab267a9f70294d8d.txt", "r") as f:
    text = f.read()
    print(text[:400])


# The first line of the "whole" documents contains the document ID as present in the COVID metadata:

# In[ ]:


get_ipython().system('ls /kaggle/input/CORD-19-research-challenge')


# In[ ]:


import pandas as pd

df_metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
df_metadata.head()


# In[ ]:


df_metadata[df_metadata["cord_uid"] == "zvrfqkol"]


# If you look at the above, the "zvrfqkol" is the ID from the example "whole" files first line as loaded above (few cells up).

# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/paragraphs')


# In[ ]:


get_ipython().system('ls /kaggle/input/covid-nlp-preprocess/output/paragraphs/biorxiv_medrxiv | head -n 10')


# An example of how to load one, and how to access the doc id similar to above for the whole doc:

# In[ ]:


import json

with open("/kaggle/input/covid-nlp-preprocess/output/paragraphs/biorxiv_medrxiv/006df1a5284369a9e2ff2dc7ab267a9f70294d8d.json") as f:
    d = json.load(f)
    print("doc_id: "+d["doc_id"])
    texts = ""
    for paragraph in d["body_text"]:
        paragraph_text = " ".join(paragraph["text"])
        texts += paragraph_text + "\n\n"
    print(texts)


# In[ ]:




