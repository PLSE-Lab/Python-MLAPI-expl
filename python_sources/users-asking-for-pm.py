import sqlite3
import pandas as pd
import numpy as np
from textblob import TextBlob
from collections import Counter
from flask import jsonify
conn = sqlite3.connect('../input/database.sqlite')

users = conn.execute("SELECT distinct(author) FROM May2015 WHERE lower(author) LIKE '%pm_me%' OR '%pm-me%' ")

fusers = users.fetchall()

user_list = [x[0] for x in fusers]

print ("Number of users asking or not asking for PMs are  " , len(user_list))

keywords =  {
                'head':         ['eye', 'eyes', 'nose', 'ears', 'lips', 'hair','teeth'],
               'torso':         ['breast','boob','nipple','stomach','tit', 'intestines','abs', 'appendix'],
               'limbs':          ['hand','leg','feet','ankle','arms'],
                'male genitals':     ['penis', 'cock','dick', 'balls','test'],
                'female genitals':         ['vagina','pussy', 'clit'],
                'behind':           ['ass','butt','bum'],
            'NoPMs':                ['dont_pm']
            }
total = 0
for key, words in keywords.items():
    key_count =0
    key_count = sum(a.lower() in b.lower() for a in words for b in user_list)
    total += key_count
    print ("Number of users asking for ", key, "parts are ", key_count)
    print()
    print (list(b for a in words for b in user_list if a.lower() in b.lower()))
    print()

print ("Number of users asking for body parts are ", total)

