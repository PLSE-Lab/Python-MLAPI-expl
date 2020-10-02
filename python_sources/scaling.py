# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Any results you write to the current directory are saved as output.
# Scaling data
df_grid = pd.read_feather('/kaggle/input/data-prep/df_sales.feather')
def downcast_dtypes(df):
    '''
        Changes column types in the dataframe: 
                
                `float64` type to `float32`
                `int64`   type to `int32`
    '''
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float16)
    df[int_cols]   = df[int_cols].astype(np.int16)    
    return df
shop_revenue_cols = [col for col in df_grid.columns if 'shop_revenue' in col]
cols_to_downcast = list(set(df_grid.columns) - set(shop_revenue_cols))
df_grid[shop_revenue_cols] = df_grid[shop_revenue_cols].astype(np.float32) 
df_grid[cols_to_downcast] = downcast_dtypes(df_grid[cols_to_downcast])
cat_features = ['shop_id','item_id','item_category_id','item_price_bin','month','date_block_num','city_code','type_code','subtype_code']
cols_to_drop = ['item_cnt_month'] + cat_features
df_grid.drop(cols_to_drop, axis=1, inplace=True)
scaler = StandardScaler()
cols = df_grid.columns
df_grid = scaler.fit_transform(df_grid)
df_grid = pd.DataFrame(df_grid, columns=cols)
df_grid.to_feather('df_scaled.feather')