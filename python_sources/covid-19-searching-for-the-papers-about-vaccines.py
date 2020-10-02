#!/usr/bin/env python
# coding: utf-8

# <font size="3">Browsing papers with highlight</font>
# <br><br>
# If you run this code, an input box will apper.By entering a word in the input box, you can search for the papers which has the word in the title, and can browse them with the word highlighted.

# In[ ]:


from IPython.core.display import display, HTML
import pandas as pd
import re
import ipywidgets

def search_papers(title: str):
    
    df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
    
    if title != '':
        df_s = df.loc[df['title'].fillna('').str.contains(title, case=False),['title','abstract','doi']]
        df_s['title'] = [re.sub(title,'<span style="background-color:lime">' + title + '</span>', i, flags=re.IGNORECASE) for i in df_s['title']]
        df_s['abstract'] = [re.sub(title,'<span style="background-color:lime">' + title + '</span>', j, flags=re.IGNORECASE) for j in df_s['abstract'].fillna('')]
        df_s['doi'] = '<a href = "https://doi.org/' + df_s['doi'] + '" target="_blank">link</a>'
        msg = str(len(df_s)) + ' papers'
        if len(df_s) > 2000:
            df_s = df_s.head(2000)
            msg = '2000 of ' + msg
        results = HTML(msg + df_s.to_html(escape=False))
        
    else:
        msg = 'Please enter a keyword'
        results = HTML(msg)
    
    return display(results)

ipywidgets.interactive(search_papers, title='')


# For example, if you enter "vaccine" in the input box, you can get the output below.

# In[ ]:


ipywidgets.interactive(search_papers, title='vaccine')

