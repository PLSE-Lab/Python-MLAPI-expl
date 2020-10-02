import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# load in dataset
data = pd.read_csv('/kaggle/input/find-the-masks-us-data/data-us.csv')
data = data[['Timestamp', 'State?']]
# print(data.columns)

# replacing the header
new_header = data.iloc[0]
data = data[1:]
data.columns = new_header

# convert timestamp to Datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# dataset for March
march = data[data['timestamp'] <= '2020-03-31']
march = march.groupby('state').timestamp.count().reset_index()
march['month'] = 'March'

# dataset for April
april = data[(data['timestamp'] > '2020-03-31') & (data['timestamp'] <= '2020-04-30')]
april = april.groupby('state').timestamp.count().reset_index()
april['month'] = 'April'

# dataset for May
may = data[(data['timestamp'] > '2020-04-30') & (data['timestamp'] <= '2020-05-31')]
may = may.groupby('state').timestamp.count().reset_index()
may['month'] = 'May'

# dataset for June
june = data[(data['timestamp'] > '2020-05-31') & (data['timestamp'] <= '2020-06-30')]
june = june.groupby('state').timestamp.count().reset_index()
june['month'] = 'June'

# combining and exporting the datasets
data = pd.concat([march, april, may, june])
data['country'] = 'United States'
# print(data)
data.to_csv('ftm_statemonthneeds.csv', index=False)