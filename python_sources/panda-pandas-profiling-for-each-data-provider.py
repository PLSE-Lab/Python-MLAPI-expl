#!/usr/bin/env python
# coding: utf-8

# # PANDA: pandas-profiling for each data provider

# In[ ]:


import re
import time
import pandas as pd
import numpy as np
from openslide import OpenSlide
import openslide
from pathlib import Path
from pandas_profiling import ProfileReport


# In[ ]:


def create_os_df(path, index, columns):
    os_df = pd.DataFrame(columns=columns).set_index(index)
    i = 0
    start = time.time()
    files = list(path.glob('*.tiff'))
    max = len(files)
    for file in files:
        os=OpenSlide(str(file))
        prp = os.properties
        m = re.match(r'([a-z0-9]*)(_mask)?',file.stem)
        image_id = str(m.groups(0)[0])
        data = [ 
            os.dimensions[0],
            os.dimensions[1],
            os.level_dimensions[0][0],
            os.level_dimensions[0][1],
            os.level_dimensions[1][0],
            os.level_dimensions[1][1],
            os.level_dimensions[2][0],
            os.level_dimensions[2][1],
            os.level_downsamples[0],
            os.level_downsamples[1],
            os.level_downsamples[2],
            prp[openslide.PROPERTY_NAME_QUICKHASH1] if openslide.PROPERTY_NAME_QUICKHASH1 in prp else np.nan,
        ]

        se = pd.Series(data, index=os_df.columns, name=image_id)
        os_df = os_df.append(se)
        os.close()
        i += 1
#         if i % 100 == 0: print('progress: {}/{}'.format(i, max))
    elapsed_time = time.time() - start
    print ("elapsed_time:{0} [sec]".format(elapsed_time))
    return os_df

data_dir_path = Path('../input/prostate-cancer-grade-assessment/train_images/')
label_dir_path = Path('../input/prostate-cancer-grade-assessment/train_label_masks/')
index = 'image_id'
columns = [
    index,
    'dimensions_x',
    'dimensions_y',
    'level_dimensions_0_x',
    'level_dimensions_0_y',
    'level_dimensions_1_x',
    'level_dimensions_1_y',
    'level_dimensions_2_x',
    'level_dimensions_2_y',
    'level_downsamples_0',
    'level_downsamples_1',
    'level_downsamples_2',
    'quickhash1',
]

label_columns = [
    index,
    'label_dimensions_x',
    'label_dimensions_y',
    'label_level_dimensions_0_x',
    'label_level_dimensions_0_y',
    'label_level_dimensions_1_x',
    'label_level_dimensions_1_y',
    'label_level_dimensions_2_x',
    'label_level_dimensions_2_y',
    'label_level_downsamples_0',
    'label_level_downsamples_1',
    'label_level_downsamples_2',
    'label_quickhash1',
]


# ## Create Dataframes

# In[ ]:


csv_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')
data_df = create_os_df(data_dir_path, index, columns)
label_df = create_os_df(label_dir_path, index, label_columns)


# ## Combine Dataframes

# In[ ]:


os_df =  pd.DataFrame.join(pd.DataFrame.join(csv_df, data_df, on='image_id', how='left'), label_df, on='image_id', how='left')


# ## Profiling report for karolinska's data

# In[ ]:


karolinska = ProfileReport(os_df[os_df['data_provider'] == 'karolinska'], title="Profiling report for radboud's data")
karolinska.to_file(output_file="ProfileReport_karolinska.html")


# In[ ]:


karolinska.to_widgets()


# ## Profiling report for radboud's data

# In[ ]:


radboud = ProfileReport(os_df[os_df['data_provider'] == 'radboud'], title="Profiling report for radboud's data")
radboud.to_file(output_file="ProfileReport_radboud.html")


# In[ ]:


radboud.to_widgets()


# In[ ]:


os_df.to_csv('joined.csv')


# In[ ]:




