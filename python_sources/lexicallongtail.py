import sqlite3
import pandas as pd
import numpy as np
import re

sql_conn = sqlite3.connect('../input/database.sqlite')

res = pd.read_sql("SELECT body "
                  "FROM May2015 "
                  "LIMIT 1000000"
                  ,sql_conn)

frequencies = {}
lastsentence = {}

for i, body in enumerate(res['body']):
    words=re.sub(r'\W+', ' ', re.sub(r'_',' ',re.sub(r'\?',' ',body))).split()
    for wd in words:
        if len(wd)<25 and len(wd)>2 and wd.find('http')==-1: 
            frequencies[wd]=frequencies.get(wd,0)+1
            context=body.replace('\n','').replace('\r','')
            pos=context.find(wd)
            if len(context)<300: lastsentence[wd]=context
            if len(context)>=300:
                lastsentence[wd]=context[max(pos-150,0):min(pos+150,len(context)-1)]
                

f = open('results.txt', 'w')

f.write("Words only found once, and context:\r\n")

for word, frequency in frequencies.items():
    if frequency==1:
        f.write(word+"\t\t\t"+lastsentence[word]+'\r\n')
