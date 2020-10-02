
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.

df = pd.read_csv('/kaggle/input/india-air-quality-data/data.csv' , encoding='mac_roman')

# Any results you write to the current directory are saved as output.

#df.head()
#df.info()

#drop off the stn_code, sampling date, pm2.5, spm
df.drop(['stn_code','sampling_date','pm2_5','spm','agency','location_monitoring_station'] , axis=1 , inplace=True)
df['date'] = pd.to_datetime(df['date'])

df = df[df['date'].isna()==False]

df['type'] = df['type'].fillna('NA')

#COnsolidate the Location Types

res_str='Residential|RIRUO'
ind_str = 'Industrial'
sen_str = 'Sensitive'

rro_mask = df['type'].str.contains(res_str , regex=True)
ind_mask = df['type'].str.contains(ind_str)
sen_mask = df['type'].str.contains(sen_str)

df['type'][rro_mask] = 'RRO'
df['type'][ind_mask] = 'Industrial'
df['type'][sen_mask] = 'Sensitive'

#df['agency'].fillna('NA',inplace=True)
#df['location_monitoring_station'].fillna('NA',inplace=True)

#Remove the outliers S02

Q1=df['so2'].quantile(0.25)
Q3=df['so2'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['so2']<(Q1-1.5*IQR))|(df['so2']>(Q3+1.5*IQR)))]

df['so2'] = df['so2'].fillna(method='ffill')

#Remove the outliers NO2

Q1=df['no2'].quantile(0.25)
Q3=df['no2'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['no2']<(Q1-1.5*IQR))|(df['no2']>(Q3+1.5*IQR)))]

df['no2'] = df['no2'].fillna(method='ffill')


#Remove the outliers RSPM

Q1=df['rspm'].quantile(0.25)
Q3=df['rspm'].quantile(0.75)
IQR=Q3-Q1
df=df[~((df['rspm']<(Q1-1.5*IQR))|(df['rspm']>(Q3+1.5*IQR)))]

df['rspm'] = df['rspm'].fillna(method='ffill')

#OUTPUT FINAL CSV FILE 
df.to_csv('/kaggle/working/df.csv')