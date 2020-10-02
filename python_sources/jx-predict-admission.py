#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Library Import

# In[ ]:


import fastai
print(fastai.version.__version__)
from fastai.tabular import * 

from pathlib import Path
import seaborn as sns

from IPython.display import display
from IPython.display import HTML
import altair as alt
from altair.vega import v3


# In[ ]:


##-----------------------------------------------------------
# This whole section 
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
    "This code block sets up embedded rendering in HTML output and<br/>",
    "provides the function `render(chart, id='vega-chart')` for use below."
)))


# In[ ]:


admission = Path('../input')
admission.ls()


# ## Basic Data Housekeeping
#     1. Columns name cleanup
#     2. Check if missing data exists (na)

# In[ ]:


df = pd.read_csv(admission / 'Admission_Predict_Ver1.1.csv')
print('Original columns: ', df.columns)
col_name_map = {}
for col in df.columns:
    col_name_map[col] = col.rstrip()
df.rename(columns=col_name_map, inplace=True)
print('Cleaned columns: ', df.columns)
print('Shape of data: ', df.shape)
display(df.head())
print('Are there any missing data?')
display(df.isna().any())


# ## Data Visualization
# ### [Observation] **Strong linear correlation between `Chance of Admit` and : **
# 1. `GRE Score`
# 2. `TOEFL Score`
# 3. `SOP`
# 4. `CGPA`
# 

# In[ ]:


g = sns.PairGrid(df[df.columns[1:]], diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(sns.scatterplot)
g.map_diag(sns.kdeplot, lw=3)


# ## ML with `fastai`
# > Using Neutral Network
# ### Create tabular `DataBunch`

# In[ ]:


dep_var = 'Chance of Admit'
cont_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']
cat_names = ['Research']
procs = [Categorify, Normalize]


# In[ ]:


data = (
    TabularList.from_df(df, cat_names=cat_names, cont_names=cont_names, procs=procs)
               .split_by_idx(valid_idx=range(400,500))
               .label_from_df(cols=dep_var)
               .databunch()
)


# In[ ]:


data.show_batch()


# > **side question**: how to get the whole dataframe for training and validation? `data.train_ds.to_df()` doesn't seem to work ...

# In[ ]:


# data.train_ds.to_df() is giving error "AttributeError: 'int' object has no attribute 'relative_to'"


# ### Get processed data as `DataFrame` from `DataLoader` so that `sklearn` can access
# > This is a workaround as `data.train_ds.to_df()` is not working

# In[ ]:


import numpy as np
from typing import List, Tuple


# In[ ]:


def convert_ddl_to_df(ddl:fastai.basic_data.DeviceDataLoader, 
                      cat_names:list, 
                      cont_names:list)->Tuple[pd.DataFrame, pd.DataFrame]:
    ''' Convert a `fastai.basic_data.DeviceDataLoader` instance into 
        two `pandas.DataFrame`s: the features and the target.
    '''
    list_data_array = list()
    for (x_cat, x_cont),y in ddl:
        tmp_array = np.concatenate((np.array(x_cat),np.array(x_cont),np.array(y).reshape(-1,1)), axis=1)
        list_data_array.append(tmp_array)
    data_array = np.concatenate(list_data_array, axis=0)
    
    columns = []
    for names in (cat_names, cont_names, ['target',]):
        columns.extend(names)
    
    df = pd.DataFrame(data_array, columns=columns)
    return df[columns[:-1]], df[['target']]


# In[ ]:


dataloader_train = data.dl(DatasetType.Train)
dataloader_valid = data.dl(DatasetType.Valid)
X_train, Y_train = convert_ddl_to_df(dataloader_train, cat_names=cat_names, cont_names=cont_names)
X_valid, Y_valid = convert_ddl_to_df(dataloader_valid, cat_names=cat_names, cont_names=cont_names)


# ### Train with NN

# In[ ]:


learn = tabular_learner(data, layers=[100,100,], metrics=[root_mean_squared_error, mean_squared_logarithmic_error, r2_score])


# In[ ]:


learn.fit_one_cycle(30, 1e-2)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


print(learn.summary())


# ## Inference

# In[ ]:


row = df.iloc[-10]
row['Chance of Admit']


# In[ ]:


learn.predict(row)


# ### Train with RF

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score


# In[ ]:


doe_RF = []
for n in range(10,200,20):
    RFR_model = RandomForestRegressor(n_jobs=-1)
    RFR_model.set_params(n_estimators=n)
    RFR_model.fit(X_train, Y_train)
    doe_RF.append(
        (n, mean_squared_log_error(Y_valid, RFR_model.predict(X_valid)), r2_score(Y_valid, RFR_model.predict(X_valid)))
    )
df_doe_RF = pd.DataFrame(doe_RF, columns=['n_estimator','msle','r2_score'])


# In[ ]:


base=alt.Chart(df_doe_RF).encode(
    alt.X('n_estimator:Q')
)
msle = base.mark_line(color='red').encode(
    alt.Y('msle:Q'),
)
r2 = base.mark_line(color='green').encode(
    alt.Y('r2_score:Q')
)
render(alt.layer(msle,r2).resolve_scale(y='independent'))
display(df_doe_RF)


# ## Compare Neutral Network result with RandomForest result
# > it seems that the performance are comparable, while RF is doing a little bit better

# In[ ]:


df_nn_record = pd.DataFrame(np.array(learn.recorder.metrics), columns=['root_mean_squared_error','mean_squared_logarithmic_error','r2_score'])
print(df_nn_record.max().r2_score, df_nn_record.min().mean_squared_logarithmic_error)


# In[ ]:


print(df_doe_RF.max().r2_score, df_doe_RF.min().msle)


# ## Try Interpreting Feature Importance (from the RF model)
# 1. tree-specific feature importance from `RandomForestRegresser` build-in attribute

# In[ ]:


df = pd.DataFrame({'FN':X_train.columns, 
                   'FI':RFR_model.feature_importances_})


# In[ ]:


def plot_fi(df, f_name, f_imp):
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(f'{f_imp}:Q'),
        alt.Y(f'{f_name}:N', sort=alt.EncodingSortField(
                field=f"{f_imp}",  # The field to use for the sort
                op="sum",  # The operation to run on the field prior to sorting
                order="descending"  # The order to sort in
            ))
    )
    return chart


# In[ ]:


render(plot_fi(df, 'FN', 'FI'))


# 2. permutation feature importance (__model-agnostic__)
# 
# > For each column, permute all the values (in that column), and train a new model with the modified dataset and see how the model performs (wrt the metric of interest)
# 
# > Get the average decrease of performace for each column (feature), which indicates the importance

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics.scorer import make_scorer


# In[ ]:


perm_importance = PermutationImportance(RFR_model, scoring=make_scorer(r2_score),
                                   n_iter=50, random_state=42, cv="prefit")
perm_importance.fit(X_valid, Y_valid)


# In[ ]:


df_imp = eli5.explain_weights_df(perm_importance)
df_label = pd.DataFrame({'feature': [ "x" + str(i) for i in range(len(X_valid.columns))], 'feature_name': X_valid.columns.values})
df_imp = pd.merge(df_label, df_imp, on='feature', how='inner', validate="one_to_one")


# In[ ]:


render(plot_fi(df_imp, 'feature_name', 'weight'))


# In[ ]:




