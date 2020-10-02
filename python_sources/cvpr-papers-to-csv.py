#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# **CVPR is the premier annual computer vision event comprising the main conference and several co-located workshops and short courses. With its high quality and low cost, it provides an exceptional value for students, academics and industry researchers.**
# 
# 
# <img src="http://cvpr2019.thecvf.com/images/CVPRLogo.png"
#      alt="Markdown Monster icon"
#      style="float: left; margin-right: 10px;" />
# 
# 
# CVPR is one of the best conferences in machine learning and deep learning. The **CVPR 2019 Papers** contains all the paper presented in CVPR 2019. Now, I'll process and analyse the data throughly. But first our challenge is to get the data in a desired format. Currently, the is in PDF format and hence unsuitable for through analysis. Hence, In this kernel, I'll convert this pdf data set to desirable csv format for further processing. I analyse, clean and process the data in [Data cleaning, Data Processing & Data Analysis](https://www.kaggle.com/hsankesara/data-cleaning-data-processing-data-analysis). Check it out and leave your feedback.
# 
# **Note:** I'm only intending to do text analysis and hence will not try to extract images or tables.

# In[ ]:


get_ipython().system('pip install pdfminer.six')


# In[ ]:


get_ipython().system('pip install PyPDF2')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from wand.image import Image as Img
import io
import subprocess
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from tqdm import tqdm
import PyPDF2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/cvpr2019/CVPR2019/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Img(filename="../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf", resolution=300)


# In[ ]:


get_ipython().system('pdf2txt.py -o health.txt ../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf')


# In[ ]:


pdfTxtFile = 'health.txt'
pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')
strn = ''
# loop over all the lines
for line in pdf_txt:
    strn += line


# In[ ]:


print(strn)


# In[ ]:


get_ipython().system('rm health.txt')


# In[ ]:


pdfFileObj = open("../input/cvpr2019/CVPR2019/papers/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.pdf", 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)


# In[ ]:


print(pdfReader.numPages)


# In[ ]:


pageObj = pdfReader.getPage(0)
# extracting text from page.
# this will print the text you can also save that into String
print(pageObj.extractText())


# In[ ]:


pdfReader.getDocumentInfo()


# In[ ]:


pdfTxtFile = '../input/cvpr2019/CVPR2019/abstracts/Acuna_Devil_Is_in_the_Edges_Learning_Semantic_Boundaries_From_Noisy_CVPR_2019_paper.txt'
pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')

# loop over all the lines
for line in pdf_txt:
    print(repr(line))


# ## Data Reading

# In[ ]:


papers = os.listdir('../input/cvpr2019/CVPR2019/papers/')


# In[ ]:


data_dict = {'content': [], 'abstract': [], 'authors':[], 'title':[]}


# In[ ]:


def pdf_to_text(path):
    bashCommand = "pdf2txt.py -o pap.txt " + path
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    pdfTxtFile = 'pap.txt'
    pdf_txt = open(pdfTxtFile, 'r', encoding='utf-8')
    strn = ''
    # loop over all the lines
    for line in pdf_txt:
        strn += line
    return strn


# In[ ]:


def read_txt(path):
    abs_text = open(path, 'r', encoding='utf-8')
    strn = ''
    # loop over all the lines
    for line in abs_text:
        strn += line
    return strn


# In[ ]:


def read_a_paper(name):
    if os.path.exists('../input/cvpr2019/CVPR2019/abstracts/' + name.split('.')[0] + '.txt'):
        data_dict['content'].append(pdf_to_text('../input/cvpr2019/CVPR2019/papers/' + name))
        data_dict['abstract'].append(read_txt('../input/cvpr2019/CVPR2019/abstracts/' + name.split('.')[0] + '.txt'))
        pdfFileObj = open('../input/cvpr2019/CVPR2019/papers/' + name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pdf_meta = pdfReader.getDocumentInfo()
        data_dict['title'].append(pdf_meta['/Title'])
        data_dict['authors'].append(pdf_meta['/Author'])


# In[ ]:


for paper in tqdm(papers):
    read_a_paper(paper)


# In[ ]:


len(data_dict['content'])


# In[ ]:


df = pd.DataFrame(data_dict)


# In[ ]:


df.to_csv('cvpr2019.csv')


# In[ ]:





# In[ ]:




