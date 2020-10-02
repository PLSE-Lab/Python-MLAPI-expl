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
from sklearn import linear_model

train_chunks = pd.read_csv('../input/train.csv', chunksize=1000)
clf = linear_model.SGDClassifier()
y_all = range(0,100)
dest = pd.read_csv('../input/destinations.csv')

for chunk in train_chunks:
    join = chunk.join(dest, on='srch_destination_id', how='left', lsuffix='l', rsuffix='r').fillna(0)
    array = join.drop(['date_time', 'srch_ci', 'srch_co', 'srch_destination_idl', 'srch_destination_idr'], axis = 1)
    X = array.drop('hotel_cluster', axis = 1).as_matrix()
    Y = array['hotel_cluster'].as_matrix()
    clf.partial_fit(X, Y, y_all)
    
col = ['site_name', 'posa_continent', 'user_location_country', 'user_location_region', 'user_location_city', 'orig_destination_distance', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_type_id', 'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market']
test = pd.read_csv('test.csv', chunksize=1000)
col = col + [c for c in dest.columns if c != 'srch_destination_id']

i = 0
for chunk in test:
    print(i)
    array = chunk.join(dest, on='srch_destination_id', how='left', lsuffix='l', rsuffix='r').fillna(0)
    array['cnt'] = 1
    array['is_booking'] = 1
    X = array[col].as_matrix()
    for row in X:
        submission.append((i, clf.predict(row)))
        i += 1
        
with open('submission.csv', 'w') as f:
    f.write('id,hotel_cluster')
    for idx,row in enumerate(test_matrix):
        f.write(str(idx)+','+str(int(clf.predict(row)))+'\n')