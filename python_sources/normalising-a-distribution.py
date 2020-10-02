#!/usr/bin/env python
# coding: utf-8

# # Normalising a distribution

# ### Inspirations from https://chrisalbon.com/python/data_wrangling/pandas_normalize_column/ and https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame.
# 
# #### Imported from https://github.com/neomatrix369/awesome-ai-ml-dl/blob/master/notebooks/data/data-processing/Normalising-a-distribution.ipynb (main repo: https://github.com/neomatrix369/awesome-ai-ml-dl/).
# 
# Note: few of the ideas here are experimental

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (20,3)


# In[ ]:


# data = {'score': [234, 24, 14, 27,-74,45,73,-18,59,160] }
num_of_points=20
scale = 0.1
centre_of_distribution = 0.0
data = {'score': np.random.normal(centre_of_distribution, scale, num_of_points) }


# In[ ]:


df = pd.DataFrame(data)
df


# # Raw data

# In[ ]:


df['score'].plot(kind='bar')


# In[ ]:


df['score'].plot()


# In[ ]:


df['score'].describe()


# # Removing minimum value from all numbers (inspecting a step)

# In[ ]:


df['score_min_removed'] = df['score'] - df['score'].min()


# In[ ]:


df[['score','score_min_removed']].plot()


# #### Note: don't be surprised that the plot of `score_min_removed` looks higher on the number-line axis than `score`. `min` value is a negative number, removing a negative value from a positive value only increases the positive value i.e. `x - (-y) = x + y`

# In[ ]:


df[['score','score_min_removed']].plot(kind='bar')


# # Removing absolute minimum value from all numbers (inspecting a step)

# In[ ]:


df['score_abs_min_removed'] = df['score'] - abs(df['score'].min())


# In[ ]:


df[['score','score_abs_min_removed']].plot()


# In[ ]:


df[['score','score_abs_min_removed']].plot(kind='bar')


# # Removing maximum value from all numbers (inspecting a step)

# In[ ]:


df['score_max_removed'] = df['score'] - df['score'].max()


# In[ ]:


df[['score','score_max_removed']].plot()


# In[ ]:


df[['score','score_max_removed']].plot(kind='bar')


# # Removing mean from all numbers (inspecting a step)

# In[ ]:


df['score_mean_removed'] = df['score'] - df['score'].mean()


# In[ ]:


df[['score','score_mean_removed']].plot()


# In[ ]:


df[['score','score_mean_removed']].plot(kind='bar')


# # Removing median from all numbers (inspecting a step)

# In[ ]:


df['score_median_removed'] = df['score'] - df['score'].median()


# In[ ]:


df[['score','score_median_removed']].plot()


# In[ ]:


df[['score','score_median_removed']].plot(kind='bar')


# # Removing trimean from all numbers (inspecting a step)

# In[ ]:


def trimean(values):
    return (np.quantile(values, 0.25) + (2 * np.quantile(values, 0.50)) + np.quantile(values, 0.75))/4


# In[ ]:


df['score_trimean_removed'] = df['score'] - trimean(df['score'])


# In[ ]:


df[['score','score_trimean_removed']].plot()


# In[ ]:


df[['score','score_trimean_removed']].plot(kind='bar')


# # Comparing: raw, minimum, absolute minimum, maximum values removed from raw, mean removed from raw, median removed from raw and trimean removed from raw

# In[ ]:


df[['score','score_min_removed', 'score_abs_min_removed','score_max_removed', 'score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot()


# In[ ]:


df[['score','score_min_removed', 'score_abs_min_removed', 'score_max_removed', 'score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot(kind='bar')


# ### Interesting to see that the plots of `score_mean_removed`, `score_median_removed` and `score_trimean_removed` are quite close to each other

# In[ ]:


df[['score','score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot()


# In[ ]:


df[['score','score_mean_removed', 'score_median_removed', 'score_trimean_removed']].plot(kind='bar')


# # Normalise mean (using scikit-learn's normalize function)

# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


values = normalize(np.array(df['score']).reshape(1,-1))
print(values[0])
df['score_sklearn_normalize'] = values[0]


# In[ ]:


df['score_sklearn_normalize'].plot()


# # Normalise by mean method

# In[ ]:


def normalise_mean(data):
    return (data - data.mean()) / data.std()


# In[ ]:


df['score_normalise_mean'] = normalise_mean(df['score'])
df['score_normalise_mean']


# In[ ]:


df['score_normalise_mean'].plot()


# # Normalise by min-max method

# In[ ]:


def normalise_min_max(data):
    return (data - data.max()) / (data.max() - data.min())


# In[ ]:


df['score_normalise_min_max'] = normalise_min_max(df['score'])
df['score_normalise_min_max']


# In[ ]:


df['score_normalise_min_max'].plot()


# In[ ]:


import numpy as np


# # Normalise using exp

# In[ ]:


df['score_exp'] = df['score'].apply(np.exp)
df['score_exp']


# In[ ]:


df['score_exp'].plot()


# # Normalise using natural log (base e)

# In[ ]:


df['score_log_base_e'] = df['score'].apply(np.log)
df['score_log_base_e']


# In[ ]:


df['score_log_base_e'].plot()


# # Normalise using log (base 10)

# In[ ]:


df['score_log_base_10'] = df['score'].apply(np.log10)
df['score_log_base_10']


# In[ ]:


df['score_log_base_10'].plot()


# # Comparing all Normalise actions together (mean, min-max, scikitlearn normalize, exp, log base e, log base 10)

# In[ ]:


df.columns


# In[ ]:


columns_to_show = ['score_sklearn_normalize', 'score_normalise_mean', 'score_normalise_min_max', 
                   'score_exp', 'score_log_base_e', 'score_log_base_10']
plt.plot(df[columns_to_show])
plt.legend(columns_to_show)


# #### `score_normalise_mean` is a lot less smoother than `score_sklearn_normalize` or `score_normalise_min_max`, while the log variants are discontinuous. Although discontinuous, they do trace the deviations of the other plots.
