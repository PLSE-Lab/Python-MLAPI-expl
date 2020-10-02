# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import difflib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


def analiza(row):
	#print str(row['id'])+" "+str(row['question1']).lower()+" "+str(row['question2']).lower()
	f1 = str(row['id'])+" "+str(row['question1']).lower()
	f2 = str(row['question2']).lower()
	puntaje = difflib.SequenceMatcher(None,f1,f2).ratio()	
	print (str(row['id'])+" "+str(puntaje)+" "+str(row['is_duplicate']))


train = pd.read_csv("../input/train.csv")
train.apply(analiza,axis=1,raw=True)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.