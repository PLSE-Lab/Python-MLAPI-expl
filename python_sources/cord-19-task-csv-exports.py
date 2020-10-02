#!/usr/bin/env python
# coding: utf-8

# # Task CSV Exports
# The following are direct CSV export files of the query result tables. These can be used to import data into another system/location.

# In[ ]:


import os
import os.path
import pandas as pd
import shutil

from IPython.display import display, FileLinks

def formatter(dirname, fnames, included_suffixes):
    names = []
    for name in sorted(fnames):
        if name.endswith(".csv"):
            names.append("<a href='%s/%s'>%s</a><br/>" % (dirname, name, name))
    
    return names

def files(name):
    if not os.path.exists(name):
        # Copy files from input to output
        shutil.copytree(os.path.join("../input", name), name)

    # List directory contents
    display(FileLinks("%s" % name, recursive=False, notebook_display_formatter=formatter))


# # [Task 1: Population](https://www.kaggle.com/davidmezzetti/cord-19-population)

# In[ ]:


files("cord-19-population")


# # [Task 2: Relevant Factors](https://www.kaggle.com/davidmezzetti/cord-19-relevant-factors)

# In[ ]:


files("cord-19-relevant-factors")


# # [Task 3: Patient Descriptions](https://www.kaggle.com/davidmezzetti/cord-19-patient-descriptions)

# In[ ]:


files("cord-19-patient-descriptions")


# # [Task 4: Models and Open Questions](https://www.kaggle.com/davidmezzetti/cord-19-models-and-open-questions)

# In[ ]:


files("cord-19-models-and-open-questions")


# # [Task 5: Materials](https://www.kaggle.com/davidmezzetti/cord-19-materials)

# In[ ]:


files("cord-19-materials")


# # [Task 6: Diagnostics](https://www.kaggle.com/davidmezzetti/cord-19-diagnostics)

# In[ ]:


files("cord-19-diagnostics")


# # [Task 7: Therapeutics](https://www.kaggle.com/davidmezzetti/cord-19-therapeutics)

# In[ ]:


files("cord-19-therapeutics")


# # [Task 8: Risk Factors](https://www.kaggle.com/davidmezzetti/cord-19-risk-factors)

# In[ ]:


files("cord-19-risk-factors")

