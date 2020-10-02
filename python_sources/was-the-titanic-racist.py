'''
Created on Oct 7, 2015
This script will not run on kaggle because it makes requests from
an external api, but I thought it would be interesting to post for others
to use on their own.

My result:
Race                N   Survival Rate
Africans          	19	0.21
British           	471	0.42
EastEuropean      	54	0.19
GreaterEastAsian  	25	0.48
IndianSubContinent	17	0.41
Jewish            	87	0.41
Muslim            	39	0.23
WestEuropean      	179	0.36

@author: JBroski
'''
import numpy as np
import pandas as pd
import requests
import json

def last_name(full_name):
    if '(' in full_name:
        return full_name.split('(')[1].strip(')').split(' ')[-1]
    else:
        return full_name.split(',')[0]
    
def first_name(full_name):
    if '(' in full_name:
        return full_name.split('(')[1].split(' ')[0]
    else:
        if '.' in full_name:
            return full_name.split('.')[1].strip(' ').split(' ')[0]
        else:
            return full_name.split(' ')[2]
    return 'NA'

def race(simple_name):
    return race_list[simple_name][1]['best']

titanic = pd.read_csv("train.csv")
titanic['first_name'] = titanic['Name'].apply(first_name)
titanic['last_name'] = titanic['Name'].apply(last_name)
titanic['simple_name'] = titanic['first_name'] + ' ' + titanic['last_name']

payload = {'names': titanic['simple_name'].values.tolist()}
headers = {'content-type':'application/json'}
r = requests.post('http://www.textmap.com/ethnicity_api/api',data=json.dumps(payload),headers=headers)

race_list = json.loads(r.text)
titanic['race'] = titanic['simple_name'].apply(race)

for s in np.unique(titanic['race']):
    count = np.sum(titanic['race']==s)
    percentSurvived = 1.0*np.sum(titanic[titanic['Survived']==1]['race']==s)/count
    out = '{:18}\t{}\t{:3.2}'.format(s, count, percentSurvived)
    print(out)