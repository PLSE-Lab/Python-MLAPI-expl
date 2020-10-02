# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import operator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def maxDay(days):
    return max(days.items(), key=operator.itemgetter(1))[0]
# Any results you write to the current directory are saved as output.
def writeInFile(path, array):
    f = open(path, 'w')
    print('id,nextvisit',file=f)
    for idx, visitor in enumerate(array):
        print(str(idx+1)+',',maxDay(visitor), file=f)
    f.close()
    
def main():
    train = pd.read_csv('../input/train.csv')
    visitors = []
    for idx, visitor_id in enumerate(train['id']):
        visitors.append({'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0})
        dates = train['visits'][idx].split(' ')[1:]
        dates = list(map(lambda x:int(x),dates))
        for date in dates:
            visitors[idx][str((date-1)%7+1)]+=date
    writeInFile('solution.csv', visitors)
    
if __name__ == '__main__':
    main()