#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import glob
import datetime
import json
import re
import numpy as np

# bokeh
from bokeh.io import output_notebook, push_notebook
from bokeh.io import show, save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, NumeralTickFormatter
from bokeh.palettes import Set1_9 as palette
from ipywidgets import interact, IntSlider
import ipywidgets as widget
output_notebook()

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# # **Data Extraction**

# Load all sources metadata

# In[ ]:


root_path = '/kaggle/input/CORD-19-research-challenge'
df_metadata = pd.read_csv('%s/metadata.csv' % root_path)


# publish_time to datetime function

# In[ ]:


def publish_time_to_datetime(publish_time):
    if(str(publish_time) == 'nan'):
        return_date = None
        
    else:
        list_publish_time = re.split('[ -]',publish_time)
        if len(list_publish_time) >2 :
            try:
                #'2020 Jan 27'
                #'2017 Apr 7 May-Jun'
                return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%b-%d')

            except :                
                try :
                    #'2020 03 16'
                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%m-%d')
                    
                except:
                    #'2015 Jul-Aug'
                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:2]), '%Y-%b')

        elif len(list_publish_time) == 2:
            #'2015 Winter' -> 1 fev            
            if(list_publish_time[1] == 'Winter'):
                return_date = datetime.datetime(int(list_publish_time[0]), 2, 1)

            #'2015 Spring' -> 1 may            
            elif(list_publish_time[1] == 'Spring'):
                return_date = datetime.datetime(int(list_publish_time[0]), 5, 1)
                
            #'2015 Autumn' -> 1 nov
            elif(list_publish_time[1] in ['Autumn','Fall']):
                return_date = datetime.datetime(int(list_publish_time[0]), 11, 1)            
            else:
                #"2015 Oct"
                return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y-%b')

        elif len(list_publish_time) == 1:
            #'2020'
            return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y')

    return return_date


# Load the json

# In[ ]:


get_ipython().run_cell_magic('time', '', '# thanks to Frank Mitchell\njson_filenames = glob.glob(f\'{root_path}/**/*.json\', recursive=True)\ndf_data = pd.DataFrame()\n\n# set a break_limit for quick test (-1 for off)\nbreak_limit = -1\nprint_debug = False\n\nfor i,file_name in enumerate(json_filenames):\n    if(print_debug):print(file_name)\n    \n    # get the sha\n    sha = file_name.split(\'/\')[6][:-5]\n    if(print_debug):print(sha)\n    \n    # get the all_sources information\n    df_metadata_sha = df_metadata[df_metadata[\'sha\'] == sha]\n   \n    if(df_metadata_sha.shape[0] > 0):\n        s_metadata_sha = df_metadata_sha.iloc[0]\n    \n        # treat only if full text\n        if(s_metadata_sha[\'has_full_text\']):\n            dict_to_append = {}\n            dict_to_append[\'sha\'] = sha\n            dict_to_append[\'dir\'] = file_name.split(\'/\')[4]\n\n            # publish time into datetime format        \n            datetime_publish_time = publish_time_to_datetime(s_metadata_sha[\'publish_time\'])\n\n            if(datetime_publish_time is not None):\n                dict_to_append[\'publish_time\'] = datetime_publish_time\n                dict_to_append[\'title\'] = s_metadata_sha[\'title\']\n\n                # thanks to Frank Mitchell\n                with open(file_name) as json_data:\n                    data = json.load(json_data)\n\n                    # get abstract\n                    abstract_list = [data[\'abstract\'][x][\'text\'] for x in range(len(data[\'abstract\']))]            \n                    abstract = "\\n ".join(abstract_list)\n                    dict_to_append[\'abstract\'] = abstract\n\n\n                    # get body\n                    body_list = [data[\'body_text\'][x][\'text\'] for x in range(len(data[\'body_text\']))]            \n                    body = "\\n ".join(body_list)\n                    dict_to_append[\'body\'] = body\n\n\n                df_data = df_data.append(dict_to_append, ignore_index=True)\n\n    else:\n        if(print_debug):print(\'not found\')\n                \n    if (break_limit != -1):\n        if (i>break_limit):\n            break')


# In[ ]:


# set sha as index
df_data.index = df_data['sha']
df_data = df_data.drop(['sha'], axis =1)


# In[ ]:


df_data


# # **Publish date analysis**

# In[ ]:


df_publish_month = df_data.title.groupby(df_data['publish_time'].dt.to_period("M")).count()

source = ColumnDataSource(data=dict(
    month = df_publish_month.index,
    month_tooltips = df_publish_month.index.strftime('%Y/%m'),
    publication_count = df_publish_month.values
))

tooltips = [('month','@month_tooltips'),('publication_count','@publication_count')]
tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset', HoverTool(tooltips=tooltips, names=['hover_tool'])]
p = figure(plot_height=600,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll='wheel_zoom')
p.line('month','publication_count',source=source)
p.xaxis.formatter=DatetimeTickFormatter(months=["%Y/%m"])
p.title.text = 'Publication count per Month'
p.xaxis[0].axis_label = 'Months'
p.yaxis[0].axis_label = 'Publication count'
show(p)


# There are two signs which show that the publication dates entered in metadata.csv are sometimes incorrect:
# - there is a publication peak in December each year
# - publications have publication dates in the future

# # **Publication processing**

# Concatenation

# In[ ]:


get_ipython().run_cell_magic('time', '', "title_weight = 4\nabstract_weight = 2\nbody_weight = 1\n\ndef concat(s_publication):\n    s_return = ''\n    \n    # title\n    if(str(s_publication['title']) != 'nan'):\n        for i in range(title_weight + 1):\n            s_return = s_return + s_publication['title'] + ' '\n\n    # abstract\n    for i in range(abstract_weight + 1):\n        s_return = s_return + s_publication['abstract'] + ' '\n        \n    # body\n    for i in range(body_weight + 1):\n        s_return = s_return + s_publication['body'] + ' '\n        \n    return s_return\n\ndf_data['publication_processing'] = df_data.apply(concat, axis=1)")


# In[ ]:


# to release memory
df_data = df_data.drop([['title','abstract','body']], axis = 1)


# Cleaning : lower and new line removal

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_data['publication_processing'] = df_data['publication_processing'].str.lower()\ndf_data['publication_processing'] = df_data['publication_processing'].str.replace('\\n',' ')")


# Tokenize

# In[ ]:


get_ipython().run_cell_magic('time', '', "# keep only alpha\ntokenizer = nltk.RegexpTokenizer('[A-Za-z]+')\n# by step to prevent memory error\nstep = 500\nstop = int(df_data.shape[0]/step)+1\ns_temp = pd.Series()\nfor i in range(stop):\n    if(i == stop - 1):\n        print('tokenize publication %s to %s' % (i*step,df_data.shape[0]))\n    else:\n        print('tokenize publication %s to %s' % (i*step,(i+1)*step -1))\n    df_data['publication_processing'].iloc[i*step:(i+1)*step] = df_data['publication_processing'].iloc[i*step:(i+1)*step].apply(lambda x:tokenizer.tokenize(x))")


# run before

# Remove the stop words

# In[ ]:


get_ipython().run_cell_magic('time', '', "list_stopwords_english = list(nltk.corpus.stopwords.words('english'))\ndf_data['publication_processing'] = df_data['publication_processing'].apply(\n    lambda x:[w for w in x if not w in list_stopwords_english])")


# Lemmatize

# In[ ]:


get_ipython().run_cell_magic('time', '', "lemmatizer = WordNetLemmatizer()\n\ndef lemmatize_list(list_word):\n    list_return = []\n    for str_word in list_word:\n        list_return.append(lemmatizer.lemmatize(str_word))\n    return list_return\n    \ndf_data['publication_processing'] = df_data['publication_processing'].apply(\n    lemmatize_list)")


# **Term Frequency**

# In[ ]:


get_ipython().run_cell_magic('time', '', "# int8 msut be sufficient (0-255) for term frequency\ndtype='int8'\ncv = CountVectorizer(analyzer=lambda x: x, dtype=dtype)\ncounted_values = cv.fit_transform(df_data['publication_processing']).toarray()\ndf_tf = pd.DataFrame(\n    counted_values,\n    columns=cv.get_feature_names(),\n    index=df_data['publication_processing'].index\n)\n# to sparse\ndf_tf = df_tf.astype(pd.SparseDtype(dtype, 0))")


# In[ ]:


df_tf


# In[ ]:


get_ipython().run_cell_magic('time', '', 's_word_use = df_tf[df_tf>0].count().sort_values(ascending = False)/df_tf.shape[0]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'source = ColumnDataSource(data=dict(\n    word = s_word_use.index,\n    word_x = list(range(s_word_use.shape[0])),\n    use = s_word_use.values,\n    use_tooltips = s_word_use.apply(lambda x: \'%i %%\' % (100*x)).values\n))\n\ntooltips = [(\'word\',\'@word\'),(\'use\',\'@use_tooltips\')]\ntools = [\'pan\', \'box_zoom\', \'wheel_zoom\', \'reset\', HoverTool(tooltips=tooltips, names=[\'hover_tool\'])]\np = figure(plot_height=600,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll=\'wheel_zoom\')\np.line(\'word_x\',\'use\',source=source)\np.title.text = \'Word use\'\n\np.xaxis[0].axis_label = \'Word\'\ndict_x_overrides = pd.DataFrame(s_word_use.index)[0].astype(\'str\').to_dict()\np.xaxis.major_label_overrides = dict_x_overrides\np.xaxis.major_label_orientation = "vertical"\n\np.yaxis[0].axis_label = \'Use\'\np.yaxis.formatter=NumeralTickFormatter(format="0 %%")\nshow(p)')


# Problem :
# - How to differiante common word such as also, study, etc..  from medical word as virus, pathology, etc..
# - How to manage publication in no english language
