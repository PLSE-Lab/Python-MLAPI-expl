#!/usr/bin/env python
# coding: utf-8

# This notebook aims to:
# - Prepare a list of papers and their relevance to the task under consideration.
# - Prepare a list and a webpage of most relevant papers and their abstracts.
# - Display top 10 most relevant papers and their abstracts.

# In[ ]:


zabt = "Papers on Virus Genetics"
znam = "Papers_on_Virus_Genetics"


# Outside this notebook: take the task's specification; make a unique list of words; remove common words; and optionally sort them.

# In[ ]:


zwds = "agreements animal asia behavioral binding circulation continued coordinating coronaviruses determine development diagnostics differences dissemination distribution diverse epidemic evidence evolution experimental factors farmers genetic genomes genomic geographic host human-animal humans infected infections inform information interface leveraged livestock management measures mechanism mixed multi-lateral nagoya origin pathogen played protocol range rapid real-time receptor reduction reservoir role sample sars-cov-2 sequencing, serve socioeconomic southeast spill-over strain strategies surveillance sustainable temporal therapeutics track tracking understand variations virus whether wildlife"


# Import python packages: os, pandas, json, IPython, and spacy.

# In[ ]:


import os
import pandas as pd
import json
from IPython.core.display import display, HTML
# !pip uninstall spacy # Uncomment this if installed version of spacy fails.
# !pip install spacy # Uncomment this if spacy package is not installed.
import spacy
# !python -m spacy download en # Uncomment this if en language is not loaded in spacy package. 
nlp = spacy.load('en')


# 
# 
# Apply spacy's nlp tool to the set of selected words.

# In[ ]:


zchk = nlp(zwds)


# Specify the location of files of papers provided by this challenge.

# In[ ]:


ztop = '/kaggle/input/CORD-19-research-challenge'


# Make an empty dataframe, to populate later.

# In[ ]:


zdf0 = pd.DataFrame(columns=['Folder', 'File', 'Match'])


# Go through each file, review the Abstract text contained in it, compute its relevance to the task, and add it to the dataframe.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nfor zsub, zdir, zfis in os.walk(ztop):\n\n    for zfil in zfis:\n        if zfil.endswith(".json"):\n            \n            with open(zsub + os.sep + zfil) as f:\n                zout = json.load(f)\n            f.close()\n            \n            zout = " ".join([part[\'text\'] for part in zout[\'abstract\']])\n            zout = zchk.similarity(nlp(zout))\n            \n            zdf0 = zdf0.append({\'Folder\': zsub.replace(ztop, ""), \'File\': zfil, \'Match\': zout}, ignore_index=True)\n            \nprint(zdf0.head(4))')


# Export this dataframe as a csv file.

# In[ ]:


zdf0.to_csv(znam + '_Check.csv', index = False)


# Make a subset dataframe of records with more than 60% relevance.

# In[ ]:


zdf6 = zdf0[zdf0.Match > 0.6].sort_values(by=['Match'], axis=0, ascending=False, inplace=False)
print(zdf6.head(4))


# Export this subset dataframe as another csv file.

# In[ ]:


zdf6.to_csv(znam + '_Relevant.csv', index = False)


# Make a webapage html of list and abstracts of papers with more than 60% relevance.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nzht0 = "<html>\\n<head>\\n"\nzht0 = zht0 + "<title>Relevant Papers for Vaccines and Therapeutics</title>\\n"\nzht0 = zht0 + "<script>\\nfunction openPop(x) {\\nei = document.getElementById(\'pop_\' + x);\\n"\nzht0 = zht0 + "ei.style.display=\'block\';\\nec = document.getElementsByClassName(\'pip\');\\nvar i;\\n"\nzht0 = zht0 + "for (i = 0; i < ec.length; i++) {\\nif ( ec[i] != ei) { ec[i].style.display=\'none\'; }; }; }\\n"\nzht0 = zht0 + "function shutPop(x) { document.getElementById(\'pop_\' + x).style.display=\'none\'; }\\n</script>\\n"\nzht0 = zht0 + "<style>table, th, td { border: 1px solid black; }</style>\\n"\nzht0 = zht0 + "</head>\\n<body>\\n"\nzht0 = zht0 + "<h1>" + zabt + "</h1>\\n"\nzht0 = zht0 + "<p>The following is a list of relevant papers.</p><br />\\n"\nzht0 = zht0 + "<p>Click on a Title to pop up its Abstract.</p><br />\\n"\nzht0 = zht0 + "<table>\\n<tbody>\\n<tr><th>Title</th>\\n<th>Abstract</th></tr>\\n"')


# In[ ]:


zht6 = zht0 # zht6 is to be saved later as a file.
zhtd = zht0 # zhtd is a smaller version of zht6, for displaying in this notebook.

for indx, cont in zdf6.iterrows():
    
    with open(ztop + os.sep + cont['Folder'] + os.sep + cont['File']) as f:
        ztxt = json.load(f)
        f.close()
        
    ztxt = " ".join([part['text'] for part in ztxt['abstract']])
    
    zhta = "<tr><td><div onClick=openPop(" + str(indx) + ")>" + str(cont['File']) + "</div></td>\n"    
    zhta = zhta + "<td><div onClick=shutPop(" + str(indx) + ") class='pip' id='pop_" + str(indx) + "' style='display:none;'>" + ztxt + "</div></td></tr>\n"
    
    zht6 = zht6 + zhta
    if indx < 10:
        zhtd = zhtd + zhta

zht6 = zht6 + "</body>\n</html>"
zhtd = zhtd + "</body>\n</html>"


# Save the webpage html as a file.

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\nzout = open(znam + "_Relevant_10.html","a")\nzout.write(zht6)\nzout.close()')


# Display the smaller html as a webpage here.

# In[ ]:


display(HTML(zhtd))

