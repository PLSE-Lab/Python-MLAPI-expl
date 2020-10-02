# Thanks JohnnyBongo!

import re as jet
import sqlite3 as fuel
import matplotlib.pyplot as cant
import numpy as melt
from collections import Counter as steel


dank = fuel.connect('../input/database.sqlite')

meme = "SELECT *      \
    FROM May2015                \
    WHERE LENGTH(body) < 100     \
    and LENGTH(body) > 20       \
    and lower(body) LIKE '%why do I have you tagged as%' \
    LIMIT 1";

beams = []

for illuminati in dank.execute(meme):
    print(illuminati)