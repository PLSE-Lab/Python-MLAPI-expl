#!/usr/bin/env python
# coding: utf-8

# # Set Lists
# 
# This notebook generates files that list all cards in a set in the order which I use to archive a set. It goes by collector number, which isn't in the data source, but is (hopefully) what is used as an index

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

colorWheel = ['White', 'Blue', 'Black', 'Red', 'Green', 'Colorless', 'Multi']
keeps = ['name', 'rarity']

raw = pd.read_json('../input/AllSets-x.json')
raw.shape

#setStartDate = '2003-07-28' #8th Edition and later
setStartDate = '2013-09-26' #Theros and later

sets = {}
mtg = []
for col in raw.columns.values:
    release = pd.DataFrame(raw[col]['cards'])
    release = release.loc[:, keeps]
    #release['modernLegal'] = release.legalities.apply(lambda l: 'none' if not isinstance(l,list) else next((legals['legality'] for legals in l if legals['format'] == 'Modern'),'none'))
    #release = release[release.modernLegal == 'Legal']
    #print(release.modernLegal.unique())
    #release = release[release.legalities.isnotnull() and release.legalities]
    if raw[col]['releaseDate'] > setStartDate and (raw[col]['type'] == 'core' or raw[col]['type'] == 'expansion'): #8th edition was released 2003-07-29, so this should include all modern sets
        release['index'] = release.reset_index().index
        release['setName'] = raw[col]['name']
        release['releaseDate'] = raw[col]['releaseDate']
        mtg.append(release)
        sets[col] = release
del release, raw
print(len(sets))


# In[ ]:


mtg.sort(key=lambda x: x['releaseDate'][0])


# In[ ]:


output = '''\\documentclass[10pt]{article}
\\usepackage[left=3cm, right=2cm, top=1cm, bottom=1cm]{geometry}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{hyperref}
\\usepackage[latin1]{inputenc}

\\begin{document}
\\pagestyle{empty}

\\twocolumn '''

for curSet in mtg:
    df = curSet
    df.sort_values(by=['index'])
    df['output'] = df['index'].apply(lambda x: str(x+1).zfill(3)) + '  ' + df.rarity.apply(lambda x: x[0]) + '  ' + df['index'].apply(lambda x: str(int(x / 9 + 1)).zfill(2) + ':'+str(x % 9 + 1)) +'   ' + df['name']
    setname = df['setName'][0]
    output += '\\section{'+setname+'}\n\\begin{description}\n\\setlength\\itemsep{-0.5em}\n'
    output += df.output.apply(lambda x: '\t\\item ' + x).str.cat(sep='\n')
    output +='\n\\end{description}\n\\clearpage\n'
    
    
output += '\\end{document}'

f = open('sets.tex','w')
f.write(output)
f.close()

