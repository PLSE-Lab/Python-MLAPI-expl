#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input_area').style.visibility === "hidden";
 
 } else {
 $('div.input_area').style.visibility = "";
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code is hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# ## Introduction ##
# 
# Some of the code is so long, it's hard for readers to scroll through it all..testing out a method for hiding code until the viewer wants to see it.
# 
# See the cell below, which contains the  code:
# 

# In[ ]:


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input_area').hidden();
 } else {
 $('div.input_area').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code is hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# In[ ]:


from IPython.display import HTML


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output




# In[ ]:


print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input_area').hidden();
 } else {
 $('div.input_area').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code is hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')


# In[ ]:


HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input_area').hidden();
 } else {
 $('div.input_area').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code is hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

