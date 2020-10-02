import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests

from_year = 2000
to_year = 2019
df = pd.DataFrame(columns=[ 'count', 'gender', 'value'], index=range(from_year, to_year+1))
# kpi = 'N00945'
kpi = 'N15033'
municipalityId = 1860
for year in range(from_year, to_year+1):
    # https://realpython.com/api-integration-in-python/
    resp = requests.get('http://api.kolada.se/v2/data/kpi/{}/municipality/{}/year/{}'.format(kpi,municipalityId, year))
    #print(resp)
    if resp.status_code != 200:
       print('Error')
       continue 

    js = resp.json()
    #print('{} '.format(js))
    if (js['count']>0):
        df.loc[year] = [js['values'][0]['values'][0]['count'], js['values'][0]['values'][0]['gender'], js['values'][0]['values'][0]['value']]
    
print(df.dropna())
df.dropna().to_csv('output.csv')