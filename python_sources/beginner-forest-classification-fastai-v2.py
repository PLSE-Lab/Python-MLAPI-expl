#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib as plt
import seaborn as sns
from fastai.tabular import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def reverse_onehot(df, subset, reverse_name):
    df_new = pd.DataFrame()
    df_new = df.drop(subset, axis = 1)
    temp = df[subset]
    df_new[reverse_name] = temp.idxmax(axis = 1).astype(str)
    df_new[reverse_name] = df_new[reverse_name].apply(lambda x: int(str(x)[len(reverse_name):]))
    return df_new  


# In[ ]:


df_train=pd.read_csv('/kaggle/input/learn-together/train.csv')
subset = ['Soil_Type'+ str(i) for i in range(1,41)]
df1 = reverse_onehot(df_train, subset, 'Soil_Type')
subset1 = ['Wilderness_Area'+ str(i) for i in range(1,5)]
df = reverse_onehot(df1, subset1, 'Wilderness_Area')
df.head()


# In[ ]:


cols_with_missing = [col for col in df.columns
                     if df[col].isnull().any()]
print('Columns with missing values:')
print(cols_with_missing)


# In[ ]:


s = (df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# In[ ]:


numerical_cols = [cname for cname in df.columns if 
                df[cname].dtype in ['int64', 'float64']]
print(len(numerical_cols))


# In[ ]:


dep_var = 'Cover_Type'
cont_names = numerical_cols[1:11]
cont_names.extend(numerical_cols[12:14])
print(cont_names)
procs = [FillMissing, Categorify, Normalize]
df_test=pd.read_csv('/kaggle/input/learn-together/test.csv',index_col=['Id'])
id=df_test.index
test = reverse_onehot(df_test, subset, 'Soil_Type')
test = reverse_onehot(test, subset1, 'Wilderness_Area')


# In[ ]:


data = (TabularList.from_df(df,path='.', cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(12096,15020)))
                           .label_from_df(cols=dep_var)
                           .add_test((TabularList.from_df(test,path='.',cont_names=cont_names, procs=procs)))
                            .databunch()
                           )


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


learn = tabular_learner(data, layers=[500,500,500,500], metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(30,1e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


predictions, *_ = learn.get_preds(DatasetType.Test)


# In[ ]:


labels=[]
for i in range(0,len(predictions)):
    labels.append(np.argmax(list(predictions[i]))+1)


# In[ ]:


print(len(labels))
print(id.shape)
out=pd.DataFrame({'Id':id,'Cover_Type':labels})
out.head(5)
out.to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')

