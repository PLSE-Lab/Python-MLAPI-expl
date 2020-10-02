# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def ssplit(x):
    splitlist = list(x.split(','))
    length = len(splitlist)
    if length>=1:
        return splitlist[1]
    else:
        pass
def split2(x1):
    splitlist = x1.split(',')
    length = len(splitlist)
    if length>=2:
        return splitlist
    else:
        pass

db_raw = pd.read_csv('../input/IMDB-Movie-Data.csv')
db = pd.DataFrame(db_raw)
db['Genre1'] = db['Genre'].apply(lambda x: x.split(',')[0])
db['Genre2'] = db['Genre'].apply(lambda x: x.split(',')[1] if len(list(x.split(',')))>1 else False )
db['Genre3'] = db['Genre'].apply(lambda x: x.split(',')[2] if len(list(x.split(',')))>2 else False)
print(db['Title'].head(5),db['Genre1'].head(5),db['Genre2'].head(5),db['Genre3'].head(5))
db = pd.DataFrame(db)
sb.factorplot(x = 'Genre1',y = 'Rating', data = db, kind='point')
plt.show()
sb.factorplot(x = 'Genre2',y = 'Rating', data = db, kind='bar')
plt.show()
sb.factorplot(x = 'Genre3',y = 'Rating', data = db, kind='strip')
plt.show()

# Any results you write to the current directory are saved as output.