#!/usr/bin/env python
# coding: utf-8

# # Retrieve school boroughs
# 
# In this kernel I will retrieve the boroughs of each school, based on their zip codes.
# 
# Information was found on this [site][1].
# 
# [1]: https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm

# In[ ]:


import re

import parsel


results = {}  # dictionary the maps (zip code => borough)

# scrap information from the web page

html = open(r'../input/nyczipcodes/NYC Neighborhood ZIP Code Definitions.html', encoding='utf8').read()
sel = parsel.Selector(html)

cur_borough = None
_rows = sel.css('tr')[1:]
for _r in _rows:
    borough = _r.css('[headers="header1"]::text').extract_first()
    if borough:
        cur_borough = borough    
    zip_codes = _r.css('[headers="header3"]::text').extract_first()
    zip_codes = zip_codes.strip()  # remove beginning space    
    zip_codes = re.split(r',\s?', zip_codes)  # split on the comma
    for zc in zip_codes:
        results[zc] = cur_borough

# input missing values

missing = {
    '10282': 'Manhattan',
    '11001': 'Queens',
    '11109': 'Queens',
    '10311': 'Staten Island'    
}
results.update(missing)

list(results.items())[:5]


# In[ ]:


import pandas as pd


# create table with results
df = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv', index_col='Location Code')
end_df = df['Zip'].apply(lambda x: results[str(x)]).rename('Borough').to_frame()

end_df.sample(20, random_state=1)


# In[ ]:


end_df.to_csv('NYC Schools Boroughs.csv')

