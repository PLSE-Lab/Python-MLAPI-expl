# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


combined = pd.read_csv('../input/train.csv')
# Any results you write to the current directory are saved as output.

def setAge(a, b, all_means):

    if (np.isnan(a) and b[1] == 'Mr'):
        return all_means[0]
    if (np.isnan(a) and b[1] == 'Mrs'):
        return all_means[1]
    if (np.isnan(a) and b[1] == 'Miss'):
        return all_means[2]
    if (np.isnan(a) and b[1] == 'Royalty'):
        return all_means[3]
    if (np.isnan(a) and b[1] == 'Officer'):
        return all_means[4]
    if (np.isnan(a) and b[1] == 'Master'):
        return all_means[5]

    return a
    
def process_age():
    global combined
    combined['Title'] = combined[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
    mean_mr = combined[(combined.Title == 'Mr') & (pd.notnull(combined.Age))]['Age'].median()
    mean_mrs = combined[(combined.Title == 'Mrs') & (pd.notnull(combined.Age))]['Age'].median()
    mean_miss = combined[(combined.Title == 'Miss') & (pd.notnull(combined.Age))]['Age'].median()
    mean_roy = combined[(combined.Title == 'Royalty') & (pd.notnull(combined.Age))]['Age'].median()
    mean_officer = combined[(combined.Title == 'Officer') & (pd.notnull(combined.Age))]['Age'].median()
    mean_master = combined[(combined.Title == 'Master') & (pd.notnull(combined.Age))]['Age'].median()
    all_means = [mean_mr, mean_mrs, mean_miss, mean_roy, mean_officer, mean_master]
    combined['Age'] = combined['Age'].apply(lambda dic: setAge(dic, combined['Title'], all_means))
        
process_age()