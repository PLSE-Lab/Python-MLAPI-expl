#!python3
import json
from sys import exit
import cerberus as crs
import pandas as pd
import copy

sFileName = '../input/networks_1'

with open (sFileName, 'r') as f:
    n = json.load(f)

s = {
    'networks': {
         'type': 'list',
         'schema':
            {
                'type': 'dict', 'schema':
                 {
                     'id': {'type': 'string'},
                     'location':
                     {
                         'type': 'dict',
                         'schema':
                          {
                                          'city': {'type': 'string'},
                                          'country': {'type': 'string'},
                                          'latitude': {'type': 'float'},
                                          'longitude' : {'type': 'float'}
                          },
                     },
                     'name': {'type': 'string'}
                 }
            }
     }
}

v = crs.Validator(s)
v.allow_unknown = True
if not v.validate (n) : 
    print (v.errors)
    exit ('Wrong input data json format') 
#calculate bike station number
countries = {}
for node in n['networks'] :
    cn = node['location']['country']
    city = node['location']['city']
    if cn in countries :
        countries[cn]['station'] += 1
        if not city in countries[cn]['cities'] :
            countries[cn]['cities'][city] = 1
            countries[cn]['city'] += 1
        else :
            countries[cn]['cities'][city] += 1

    else : 
        countries[cn] = {}
        countries[cn]['station'] = 1
        countries[cn]['cities'] = {city: 1}
        countries[cn]['city'] = 1

cs = pd.DataFrame([(x, countries[x]['city'], countries[x]['station']) for x in sorted(countries.keys())])
cs.columns = ('CC', 'cities', 'stations')
cs.to_csv ('station_number.csv')
