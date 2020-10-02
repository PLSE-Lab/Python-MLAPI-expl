# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json
#import os
#print(os.listdir('../input/'))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
dataj={}
with open('../input/wordinfo.dll') as json_file:  
	dataj = json.load(json_file)
	
print(dataj['never'][0]['emotions'])
print(dataj['never'][1]['pos'])
print(dataj['never'][2]['tense'])
# Any results you write to the current directory are saved as output.