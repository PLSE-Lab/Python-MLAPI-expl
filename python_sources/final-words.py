#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from collections import Counter, defaultdict
from nltk.corpus import stopwords
import wordcloud
import matplotlib.pyplot as plt

a = pd.read_csv("../input/death-row.csv")
b = pd.read_csv("../input/offenders.csv", encoding='latin1')

s=set(stopwords.words('english'))


st = ""
for i in b['Last Statement']:
    st += i
st = [x for x in st.split() if x not in s]
li =Counter(st).most_common(40)

d = defaultdict(int)
for i in li:
    d[i[0]] = i[1]
#print(d)
cloud = wordcloud.WordCloud()
x = cloud.generate_from_frequencies(d)
plt.imshow(x)
plt.show()


# In[ ]:





# In[ ]:




