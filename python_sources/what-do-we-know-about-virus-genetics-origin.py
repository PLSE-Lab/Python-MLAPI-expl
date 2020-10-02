#!/usr/bin/env python
# coding: utf-8

# # What do we know about virus genetics, origin, and evolution?
# ## Task Details
# ### What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?
# 
# ### Specifically, we want to know what the literature reports about:
# 
# * Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.
# * Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.
# * Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.
# * Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.
# * Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.
# * Experimental infections to test host range for this pathogen.
# * Animal host(s) and any evidence of continued spill-over to humans
# * Socioeconomic and behavioral risk factors for this spill-over
# * Sustainable risk reduction strategies

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import functools
from IPython.core.display import display, HTML
Tasks3 = pd.read_csv('/kaggle/input/Task3.csv')
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', 500)
Tasks3_table=HTML(Tasks3.to_html(escape=False,index=False))
display(Tasks3_table)

