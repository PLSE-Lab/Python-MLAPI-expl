#!/usr/bin/env python
# coding: utf-8

# # Additional data used
# I have taken additional data ( holiday calendar for Russia and currency data for the period of the transactions) as well as translated data for the shops and item categories from https://www.kaggle.com/kazimanil/predict-future-sales-supplementary?select=shops-translated.csv
# 
# **Helpful links**
# 1. https://www.kaggle.com/kazimanil/predicting-future-sales/execution
# 2. https://www.kaggle.com/pradnyatandeluk/sales-data-visualisation-using-translations/notebook

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
item_categories = pd.read_csv("/kaggle/input/translated-sales-data/item_categories_eng.csv")
shops = pd.read_csv("/kaggle/input/translated-sales-data/shops_eng.csv")


# #  Shops translated preview

# In[ ]:


print(shops.shape)


# In[ ]:


shops.head()


# #  Item categories translated preview

# In[ ]:


print(item_categories.shape)


# In[ ]:


item_categories.head()


# In[ ]:


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set3')
    plt.title(title)
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[ ]:


plot_count(shops, 'City', 'Distribution of cities for shops (count & percent)', size=5)


# In[ ]:


plot_count(item_categories, 'item_cat1', 'Distribution of item categories 1 (count & percent)', size=5)


# In[ ]:


plot_count(item_categories, 'item_cat2', 'Distribution of item categories 2 (count & percent)', size=5)


# In[ ]:


pd.merge(train, shops, on="shop_id", how="left")


# In[ ]:


pd.merge(train, item_categories, on="item_id", how="left")


# In[ ]:


def plot_distplot_grouped(df, feature):
    classes = list(df[feature].unique())
    print(classes)
    group_labels = []     
    hist_data = []
    for item in classes:
        crt_class = df.loc[df[feature]==item]["step"]
        group_labels.append(f"{item}")
        hist_data.append(crt_class)
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig['layout'].update(title=f'Payments Transactions Time Density Plot - grouped by `{feature}`', xaxis=dict(title='Time [step]'))
    iplot(fig, filename='dist_only')   


# In[ ]:


plot_distplot_grouped(train, 'item_cat1')

