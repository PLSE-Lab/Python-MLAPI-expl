#!/usr/bin/env python
# coding: utf-8

# ##Is there a correlation between energy(calories) AND proteins, carbs, or fats?
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
food_doc = pd.read_csv('../input/FoodFacts.csv')
food_doc_specs = food_doc[food_doc.energy_100g.notnull() & food_doc.carbohydrates_100g.notnull() & food_doc.fat_100g.notnull() & food_doc.proteins_100g.notnull() & food_doc.main_category_en.notnull() & food_doc['nutrition_score_uk_100g']]
food_doc_specs = food_doc_specs[['code','generic_name','energy_100g','carbohydrates_100g','fat_100g','proteins_100g', 'main_category_en', 'nutrition_score_uk_100g']]
food_doc_counts = food_doc_specs.groupby('main_category_en').filter(lambda x:len(x)>60)

fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize = [15,8], sharex = True)
plt.xlim((0,5000))
plot_carbs = sns.regplot(x = "energy_100g", y = "carbohydrates_100g", fit_reg = False, ax = ax1, data = food_doc_specs)
plot_fat = sns.regplot(x = "energy_100g", y = "fat_100g", ax = ax2, fit_reg = False, data = food_doc_specs)
plot_protein = sns.regplot(x = "energy_100g", y = "proteins_100g", fit_reg = False, ax = ax3, data = food_doc_specs)


# 1) There is a somewhat positive correlation between energy(calories) and carbs/fat. As the number of calories increases, the number of fats and increases.
# 
# 2) Protein maintains a more linear relationship with calories. 

# ##Do relationships between carbs and calories change for different categories of food products?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

