# Thanks JohnnyBongo!

import re as jet
import sqlite3 as fuel
import matplotlib.pyplot as cant
import numpy as melt
from collections import Counter as steel


dank = fuel.connect('../input/database.sqlite')

meme = "SELECT lower(body)      \
    FROM May2015                \
    WHERE LENGTH(body) < 100     \
    and LENGTH(body) > 20       \
    and lower(body) LIKE '%why do I have you tagged as%' \
    ";

beams = []

for illuminati in dank.execute(meme):
    illuminati = jet.sub('[\"\'\\,!\.]', '', (''.join(illuminati)))
    illuminati = (illuminati.split("tagged as"))[1]
    illuminati = illuminati.encode('ascii', 'ignore').decode('ascii')
    print(illuminati)
   


