#!/usr/bin/env python
# coding: utf-8

# # Samsung Data
# I almost always have my phone with me and so it is collecting lots of useful data. Here I start to breakdown all the data that the Samung Health app has collected and see what useful insights I can gain by analyzing it.
# ## Instructions
# You can download the health data from Samsung by following these instructions https://health.apps.samsung.com/notice/100068 which then creates a bunch of folders, csv, and json files on your phone. Here is how you can start to go through some of it meaningfully. A lot the files are not particularly interesting but we show short previews anyways. I have not fully exploited the Samsung features but have done a fair amount of activity with my phone (a Galaxy Note) and there seems to be enough data to play around a bit.  
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
samsung_base_dir = os.path.join('..', 'input', 'samsung')


# In[ ]:


samsung_dump_dirs = glob(os.path.join(samsung_base_dir, '*'))
samsung_dump_dir = os.path.basename(samsung_dump_dirs[0])
print(len(samsung_dump_dirs), 'dumps found, taking first:', samsung_dump_dir)

samsung_csv_paths = glob(os.path.join(samsung_base_dir, samsung_dump_dir, '*.csv'))
print(len(samsung_csv_paths), 'csvs found')
samsung_json_paths = glob(os.path.join(samsung_base_dir, samsung_dump_dir, 'jsons', '*',  '*.json'))
print(len(samsung_json_paths), 'jsons found')


# # Process CSV

# In[ ]:


from IPython.display import display
sam_readcsv = lambda x: pd.read_csv(x, skiprows=1)
all_csv_df = {os.path.basename(j).replace('com.samsung.', ''): sam_readcsv(j) for j in samsung_csv_paths}
for k, v in all_csv_df.items():
    print(k, 'readings:', v.shape[0])
    display(v.sample(2 if v.shape[0]>2 else 1))


# ## Step Counts

# In[ ]:


step_df = pd.concat([v for k,v in all_csv_df.items() if 'step_daily_trend' in k])
# fix times
for c_col in ['create_time', 'update_time']:
    step_df[c_col] = pd.to_datetime(step_df[c_col])

step_df = step_df.sort_values('create_time', ascending = True)
pd.to_datetime(step_df['create_time'])
step_df.head(3)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize = (20, 10))
ax1.plot(step_df['create_time'], step_df['count'], '-', label = 'Steps')
ax1.plot(step_df['create_time'], step_df['distance'], '-', label = 'Meters')
ax1.plot(step_df['create_time'], step_df['calorie'], '-', label = 'Calories')
ax1.legend()
fig.autofmt_xdate(rotation = 45)
#ax1.set_xticks(ax1.get_xticks()[::15]);


# In[ ]:


sns.pairplot(hue = 'deviceuuid', data = step_df[['count', 'distance', 'calorie', 'speed', 'deviceuuid']])


# # Day of Week

# In[ ]:


step_df['day_name'] = step_df['create_time'].dt.day_name()
step_df['is_weekend'] = step_df['day_name'].map(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')
sns.pairplot(hue = 'is_weekend', data = step_df[['count', 'distance', 'calorie', 'speed', 'day_name', 'is_weekend']])


# In[ ]:


step_df['hour'] = step_df['create_time'].dt.hour
sns.pairplot(hue = 'is_weekend', data = step_df[['count', 'distance', 'calorie', 'speed', 'hour', 'is_weekend']])


# # Process the JSON 

# In[ ]:


from itertools import groupby, chain
import json
sam_json_dict = {}
for fold_id, files in groupby(samsung_json_paths, 
                              lambda x: os.path.basename(os.path.dirname(x)).replace('com.samsung.', '')):
    c_files = list(files)
    
    c_json_data = [json.load(open(c_file, 'r')) for c_file in c_files]
    sam_json_dict[fold_id] = list(chain(*c_json_data)) if isinstance(c_json_data[0], list) else c_json_data
    print(fold_id+',', 'files:', len(c_files), 'readings:', len(sam_json_dict[fold_id]))
    print('\tPreview:', str(sam_json_dict[fold_id][0])[:80])


# ## Stress

# In[ ]:


stress_df = pd.DataFrame(sam_json_dict['shealth.stress'])
sns.pairplot(stress_df)


# In[ ]:


step_df = pd.DataFrame(sam_json_dict['shealth.step_daily_trend'])
sns.pairplot(step_df)


# In[ ]:


exercise_df = pd.DataFrame(sam_json_dict['health.exercise'])
exercise_df.sample(5)


# In[ ]:


exercise_df['heart_rate'].hist()


# In[ ]:


exercise_df['accuracy'].hist()


# In[ ]:




