# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import time
import json
start = time.time()
with open("../input/freecodecamp_casual_chatroom_01.json", "r") as fin:
    dataset01 = json.load(fin)
stop = time.time()
#count number of records
print( "the dataset 01 has about {} messages".format( len( dataset01 ) ) )
print( "opening the dataset took about {:.2f} secs".format( stop-start ) )

##finding camperbot
camperbot = []
start = time.time()
for rec in dataset01:
    assert rec != None, "there is a rec evaluated as None"
    if rec['fromUser'] == None: continue
    assert rec['fromUser'] != None, "there is a rec without data from user"
    if rec['fromUser']['username'] == 'camperbot':
        camperbot.append(rec['sent'][:10])

stop = time.time()
print()
print()
print( "In this dataset there were about {} messages from CamperBot, the bot of the chatroom.\nThat is {:.1%} of all the messages.".format( len( camperbot ), len( camperbot )/len( dataset01 ) ) )
print()
print( "The first camperbot's message was sent {} and the last one {}.".format(camperbot[0],camperbot[-1]))
print()
print( "Finding camperbot messages using a simple for-loop in this Kaggle dataset took {:.2f}secs.\nThere are about 60,000 more users only in this dataset waiting for you... :)".format( stop-start ) )