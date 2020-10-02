#!/usr/bin/env python
# coding: utf-8

# ## 500 Springer Books, Download using Python

# Springer.com has released about 500 ebooks for free during the #COVID19 pandemic. Learn about Python, Data Mining, Artificial Intelligence, Neural Networks and more at Springer.com.
# Share with your friends :)

# ![Source: Welcome AI](https://lh3.googleusercontent.com/MqK4RxiPpXb-ZeoYIWB6kZcGVDgoHF9rfT9GXaoOa1xAQdQgaoilvrkfgF_G9Em5sRhPDN0UO3_mBrCxEFYnU96tuItd0hsrW4BMzEIRGCzKc_jheGRzAznvI7w9fA0Qgw3lsxlYXyk0Fy4AafoC9uVA7K3OlhAmBWsKgSSIuy-rAiIGnt37m30zxHlGRgmRtSqMnEFtrG-wk71aCaU0dzD_nhia5fEDp6yRZQhFvCVo0Qbj7zjWRgGzl_WLzOKMC543Hf-KgnRcSUeRajhUzPZEX-oCQ-kMrqs0IAnMezrdI94IoKDuHHB6bexcLA389ZbHoJ8z0umZQjT1obx7nlQvHRlskN2k6BmMf88cxWG29n_FWw2iSER-_F6l_lJ_Q6bt4PQi8VmsA-l8EgpjxTKThH_Srb9J6iiyZcLlQr8yYRMQat2jTXcN0n6Bl_MKxkUMzhN7BzrD7IQUZvDkLM0CRiM-fALIz9rtOrRAsdvzWvs9fXcCBk5Qnt0S-oZhM_hyQ07HNv82erkyEmX9zbz7yelosN-T-DuEAeB-GbMPJf4Q6T_raRnvIyMv-dcW3GXLKvtNEdxYR38dfrS-nQm6jj5imXVILaHI9MrgvS_vzzIkXAdXBEg0yhb4JIQmCpqnoi2kOmKzyAAbyI_4Ez3W0tzoGHjmqE20evzHDAtWpNvWFz85ucCT0-AE1U0=w753-h932-no)
# 
# Source: Welcome AI

# 

# ### Importing required libraries, wget for fetching data from URL, and pandas to read xlsx

# In[ ]:


get_ipython().system('pip install wget')

import requests, wget
import pandas as pd


# ### Loading the list of books and URLs from xlsx file

# In[ ]:


df = pd.read_excel("/kaggle/input/FreeEnglishtextbooks.xlsx")


# ### Loop through URLs and download books one by one

# In[ ]:


for index, row in df.iterrows():
        # loop through the excel list
        file_name = f"{row.loc['Book Title']}_{row.loc['Edition']}".replace('/','-').replace(':','-')
        url = f"{row.loc['OpenURL']}"
        r = requests.get(url) 
        download_url = f"{r.url.replace('book','content/pdf')}.pdf"
        wget.download(download_url, f"{file_name}.pdf") 
        print(f"downloading {file_name}.pdf Complete ....")


# ### All books are downloaded in output section.
