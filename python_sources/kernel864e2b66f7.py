# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_values.csv', index_col='building_id')
test_df = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/test_values.csv',  index_col='building_id')
target_df = pd.read_csv('https://s3.amazonaws.com/drivendata/data/57/public/train_labels.csv', index_col='building_id')
train_df.head()
target_df.head()
idx = train_df.shape[0]
data_df = pd.concat([train_df, test_df], sort=False)
data_df.shape
cat_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'land_surface_condition', 'foundation_type', 'roof_type', 
                    'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
data_cat = pd.DataFrame(index = data_df.index, 
                  data = data_df, 
                  columns = cat_features )
data_cat.head()
data_cat.shape
data_num = data_df.drop(columns = cat_features)
num_features = data_num.columns
data_num.shape
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(data_cat)
data_cat_encoded = enc.transform(data_cat)
data_cat_encoded.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()

data_num_scaled = scaler.fit_transform(data_num)
data_num_scaled.shape
from scipy.sparse import coo_matrix, hstack
data_num_scaled = coo_matrix(data_num_scaled)
data_num_scaled
data = hstack((data_cat_encoded,data_num_scaled))
data = data.astype(dtype='float16')
X_train = data.tocsr()[:idx]
X_test = data.tocsr()[idx:]
y_train = target_df['damage_grade'].values
from sklearn.model_selection import train_test_split
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train_split, y_train_split)
y_pred = model .predict(X_valid_split)
from sklearn.metrics import f1_score
f1_score(y_valid_split, y_pred, average='micro')
model.fit(X_train, y_train)
y_pred = model .predict(X_test)
predicted_df = pd.DataFrame(y_pred.astype(np.int8), index = test_df.index, columns=['damage_grade'])
predicted_df.to_csv('baseline.csv')
