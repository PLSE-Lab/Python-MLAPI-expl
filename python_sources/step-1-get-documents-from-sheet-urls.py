#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# 
# There is data inside `thruuu.xlsx`.
# 
# This notebook will download all the PDFs.

# ## Installs

# In[ ]:


get_ipython().system('pip install pathvalidate==2.3.0')


# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import requests
from urllib.parse import urlparse
from pathvalidate import sanitize_filename


# ## Globals / Config

# ### Create `downloads` directory

# In[ ]:


get_ipython().system('pwd')
get_ipython().system('echo "====="')
get_ipython().system('mkdir downloads')
get_ipython().system('ls')


# In[ ]:


EXCEL_FILE = "/kaggle/input/syllabus-corpus/thruuu.xlsx"
OUTPUT_FOLDER = "/kaggle/working/downloads"
CSV_FILENAME = "downloads.csv"
ARCHIVE_FILENAME = "downloads.tar.gz"

# Set the environment variables within this notebook
# to allow for referencing via in shell commands
# e.g. `!echo $EXCEL_FILE`
os.environ["EXCEL_FILE"] = EXCEL_FILE
os.environ["OUTPUT_FOLDER"] = OUTPUT_FOLDER
os.environ["CSV_FILENAME"] = CSV_FILENAME
os.environ["ARCHIVE_FILENAME"] = ARCHIVE_FILENAME


# In[ ]:


get_ipython().system('echo $EXCEL_FILE "\\n"')
get_ipython().system('echo $OUTPUT_FOLDER "\\n"')
get_ipython().system('echo $CSV_FILENAME "\\n"')
get_ipython().system('echo $ARCHIVE_FILENAME "\\n"')


# ### Fun Fact! [Pandas can read Excel](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html)
# 
# 

# In[ ]:


df = pd.read_excel(EXCEL_FILE, sheet_name="SERP")


# In[ ]:


df.head()


# In[ ]:


print(df.shape)


# In[ ]:


print(len(df["Title"].unique()))
df[df["Title"].duplicated()]


# ### make an extractor for creating useful filenames for each row

# In[ ]:


def sanitize(url):
    # extra sanitize because Kaggle does not like "~"
    return (
        sanitize_filename(url)
        .replace("~", "")
        .replace("&", "")
        .replace("=", "")
        .replace("#", "")
        .replace("%", "")
        .replace("$", "")
        .replace("@", "")
    )


# In[ ]:


sanitize("wowowo&=#owowowow/http://wow.wow.com/~wow/woow")


# In[ ]:


def _extract_last_path_part(url):
    return sanitize(os.path.split(urlparse(url).path)[1])


# In[ ]:


_extract_last_path_part("https://www.google.com/wow/ok/wo&$@%w")


# ### Now download

# In[ ]:


for index, row in df.iterrows():
    pos = row["Position"]
    serp_title = row["Title"]
    url = row["URL"]
    # fname = sanitize(url)
    fname = _extract_last_path_part(url)
    fpath = os.path.join(OUTPUT_FOLDER, f"{pos}___{fname}")
    response = requests.get(url)
    print(fpath, " "*50, "\r" ,end="")
    with open(fpath, 'wb') as f:
        f.write(response.content)


# In[ ]:


get_ipython().system('ls downloads | head -5')


# In[ ]:


get_ipython().system('ls downloads | wc -l')


# In[ ]:


get_ipython().system('xxd downloads/100__* | head -5')


# # Save archive of downloads

# In[ ]:


print(ARCHIVE_FILENAME)
get_ipython().system('echo $ARCHIVE_FILENAME')


# In[ ]:


get_ipython().system('tar -cvzf $ARCHIVE_FILENAME downloads/*')


# In[ ]:


# !rm -rf downloads
get_ipython().system('ls -lah')


# # Save dataframe too

# In[ ]:


df.to_csv(CSV_FILENAME, index=False)


# In[ ]:


pd.read_csv(CSV_FILENAME).head()


# In[ ]:


get_ipython().system('ls')


# # Verify untar/unzip

# In[ ]:


# ...........................................
# ...........................................
# ...........................................
# !tar -xvzf $ARCHIVE_FILENAME
# downloads/1___LA114-213Lec.pdf
# downloads/2___syllabus.pdf
# downloads/3___syllabus.pdf
# downloads/4___Syllabus-Lean-Sigma-Green-Belt-Cert12.pdf
# downloads/5___syllabus.pdf
# downloads/6___syllabus.pdf
# downloads/7___viewcontent.cgi
# downloads/8___syllabus.pdf
# ...........................................
# ...........................................
# ...........................................
# !ls downloads
# 1___LA114-213Lec.pdf			       5___syllabus.pdf
# 2___syllabus.pdf			       6___syllabus.pdf
# 3___syllabus.pdf			       7___viewcontent.cgi
# 4___Syllabus-Lean-Sigma-Green-Belt-Cert12.pdf  8___syllabus.pdf
# ...........................................
# ...........................................
# ...........................................


# In[ ]:




