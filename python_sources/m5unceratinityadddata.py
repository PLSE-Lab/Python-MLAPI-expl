#!/usr/bin/env python
# coding: utf-8

# # Here we have used Gradient Bossting aaproach Total Sales cases:
# 
# <hr>
# 
# 1. Data Preparation for all levels of Aggregation is done at this notebook: https://www.kaggle.com/kamalnaithani/m5unceratinityadddata
# 2. With this approach we can merge sales level data as well
# 3. Link for each hierachy seperate level is : https://www.kaggle.com/kamalnaithani/m5uncertainity-total-gradient-boosting
# 4. Applying gradient boosting approcah and mergin the output of all final models
# 5. Do upvote in case you find this notebook helpful
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gc


# In[ ]:


#============================#
def get_cat(inp):
    tokens = inp.split("_")
    return tokens[0]
#============================#
def get_dept(inp):
    tokens = inp.split("_")
    return tokens[0] + "_" + tokens[1]
#============================#


# In[ ]:


#Building all the aggregation levels


# In[ ]:


l12 = pd.read_csv("../input/m5-forecasting-uncertainty/sales_train_evaluation.csv")


# In[ ]:


l12.head()


# In[ ]:



l12.id = l12.id.str.replace('_evaluation', '')


# In[ ]:


l12.head()


# In[ ]:


COLS = [f"d_{i+1}" for i in range(1941)]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("State & Item")\nl11 = l12.groupby([\'state_id\',\'item_id\']).sum().reset_index()\nl11["store_id"] = l11["state_id"]\nl11["cat_id"] = l11["item_id"].apply(get_cat)\nl11["dept_id"] = l11["item_id"].apply(get_dept)\nl11["id"] = l11["state_id"] + "_" + l11["item_id"]\nprint("Item")\nl10 = l12.groupby(\'item_id\').sum().reset_index()\nl10[\'id\'] = l10[\'item_id\'] + \'_X\'\nl10["cat_id"] = l10["item_id"].apply(get_cat)\nl10["dept_id"] = l10["item_id"].apply(get_dept)\nl10["store_id"] = \'X\'\nl10["state_id"] = \'X\'\nprint("Store & Dept")\nl9 = l12.groupby([\'store_id\',\'dept_id\']).sum().reset_index()\nl9["cat_id"] = l9["dept_id"].apply(get_cat)\nl9["state_id"] = l9["store_id"].apply(get_cat)\nl9["item_id"] = l9["dept_id"]\nl9["id"] = l9["store_id"] + \'_\' + l9["dept_id"]\nprint("Store & Cat")\nl8 = l12.groupby([\'store_id\',\'cat_id\']).sum().reset_index()\nl8[\'dept_id\'] = l8[\'cat_id\']\nl8[\'item_id\'] = l8[\'cat_id\']\nl8[\'state_id\'] = l8[\'store_id\'].apply(get_cat)\nl8["id"] = l8["store_id"] + \'_\' + l8["cat_id"]\nprint("State & Dept")\nl7 = l12.groupby([\'state_id\',\'dept_id\']).sum().reset_index()\nl7["store_id"] = l7["state_id"]\nl7["cat_id"] = l7["dept_id"].apply(get_cat)\nl7["item_id"] = l7["dept_id"]\nl7["id"] = l7["state_id"] + \'_\' + l7["dept_id"]\nprint("State & Cat")\nl6 = l12.groupby([\'state_id\',\'cat_id\']).sum().reset_index()\nl6["store_id"] = l6["state_id"]\nl6["dept_id"] = l6["cat_id"]\nl6["item_id"] = l6["cat_id"]\nl6["id"] = l6["state_id"] + "_" + l6["cat_id"]\nprint("Dept")\nl5 = l12.groupby(\'dept_id\').sum().reset_index()\nl5["cat_id"] = l5["dept_id"].apply(get_cat)\nl5["item_id"] = l5["dept_id"]\nl5["state_id"] = "X"\nl5["store_id"] = "X"\nl5["id"] = l5["dept_id"] + "_X"\nprint("Cat")\nl4 = l12.groupby(\'cat_id\').sum().reset_index()\nl4["store_id"] = l4["cat_id"]\nl4["item_id"] = l4["cat_id"]\nl4["store_id"] = "X"\nl4["state_id"] = "X"\nl4["id"] = l4["cat_id"] + "_X"\nprint("Store")\nl3 = l12.groupby(\'store_id\').sum().reset_index()\nl3["state_id"] = l3["store_id"].apply(get_cat)\nl3["cat_id"] = "X"\nl3["dept_id"] = "X"\nl3["item_id"] = "X"\nl3["id"] = l3["store_id"] + "_X"\nprint("State")\nl2 = l12.groupby(\'state_id\').sum().reset_index()\nl2["store_id"] = l2["state_id"]\nl2["cat_id"] = "X"\nl2["dept_id"] = "X"\nl2["item_id"] = "X"\nl2["id"] = l2["state_id"] + "_X"\nprint("Total")\nl1 = l12[COLS].sum(axis=0).values\nl1 = pd.DataFrame(l1).T\nl1.columns = COLS\nl1["id"] = \'Total_X\'\nl1[\'state_id\'] = \'X\'\nl1[\'store_id\'] = \'X\'\nl1[\'cat_id\'] = \'X\'\nl1[\'dept_id\'] = \'X\'\nl1[\'item_id\'] = \'X\'')


# In[ ]:


l11.head()


# In[ ]:


df = pd.DataFrame()
df = df.append([l12, l11, l10, l9, l8, l7, l6, l5, l4, l3, l2, l1])


# In[ ]:


df.shape


# In[ ]:


sub = pd.read_csv("../input/m5-forecasting-uncertainty/sample_submission.csv")
sub['id'] = sub.id.str.replace('_evaluation', '')
grps =sub.iloc[-42840:, 0].unique()
grps = [col.replace("_0.995","") for col in grps]


# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, df[col].nunique())


# In[ ]:


#Computing scale and start date


# In[ ]:


X = df[COLS].values


# **Cumsum function generally take the sum from day 1 to day 1941 with respect to last value whether it is 0 0r not and then sum it for example:d1=d1, d2=d1+d2, d3=d1+d2+d3 and so on**
# 
# <hr>
# x=x>1 will give true and false value for each cell

# In[ ]:


X = df[COLS].values
x = (X>0).cumsum(1)
x = x>0
st = x.argmax(1)
den = 1941 - st - 2
diff = np.abs(X[:,1:] - X[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


st


# In[ ]:


df["start"] = st
df["scale"] = norm


# In[ ]:


df.head(5)


# In[ ]:


plt.plot(X[-1]/norm[-1])
plt.show()


# In[ ]:


df.to_csv("sales.csv", index=False)


# 
# **1.Getting State and Item Details seperately**

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l11[col].nunique())


# In[ ]:


X_State_Item = l11[COLS].values


# In[ ]:



x1 = (X_State_Item>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_State_Item[:,1:] - X_State_Item[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l11["start"] = st
l11["scale"] = norm
l11.head()


# In[ ]:


l11.to_pickle('State_Item_1.pkl')
#l11.to_csv('State_Item_1.csv')


# In[ ]:


l11.head()


# In[ ]:


l11.info()


# In[ ]:


del l11


# **2.Adding Total Sales**

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l1[col].nunique())


# In[ ]:


X_Total = l1[COLS].values


# In[ ]:


X_Total


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l1["start"] = st
l1["scale"] = norm
l1.head()


# In[ ]:


l1.to_pickle('TotalSales.pkl')


# In[ ]:


del l1


# # 3.Adding Item Details

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l10[col].nunique())


# In[ ]:


X_Total = l10[COLS].values


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l10["start"] = st
l10["scale"] = norm
l10.head()


# In[ ]:


l10.to_pickle('Items.pkl')


# In[ ]:


del l10


# # 4.Store and Department

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l9[col].nunique())


# In[ ]:


X_Total = l9[COLS].values


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l9["start"] = st
l9["scale"] = norm
l9.head()


# In[ ]:


l9.to_pickle('Store_Dept.pkl')


# In[ ]:


del l9


# # 5.Store and Category

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l8[col].nunique())


# In[ ]:


X_Total = l8[COLS].values


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l8["start"] = st
l8["scale"] = norm
l8.head()


# In[ ]:


l8.to_pickle('Store_Cat.pkl')


# In[ ]:


del l8


# # 6.State & Dept

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l7[col].nunique())


# In[ ]:


X_Total = l7[COLS].values


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l7["start"] = st
l7["scale"] = norm
l7.head()


# In[ ]:


l7.to_pickle('State_Dept.pkl')


# In[ ]:


del l7


# # 7. State&Category

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l6[col].nunique())


# In[ ]:


X_Total = l6[COLS].values


# In[ ]:



x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l6["start"] = st
l6["scale"] = norm
l6.head()


# In[ ]:


l6.to_pickle('State_Category.pkl')


# In[ ]:


del l6


# # 8.Department

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l5[col].nunique())


# In[ ]:


X_Total = l5[COLS].values
x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l5["start"] = st
l5["scale"] = norm
l5.head()


# In[ ]:


l5.to_pickle('Department.pkl')
del l5


# # 9. Category

# In[ ]:


l4


# In[ ]:


for col in ['id','item_id','cat_id','store_id','state_id']:
    print(col, l4[col].nunique())


# In[ ]:


X_Total = l4[COLS].values
x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l4["start"] = st
l4["scale"] = norm
l4.head()


# In[ ]:


l4.to_pickle('Category.pkl')
del l4


# # 10. Store

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l3[col].nunique())


# In[ ]:


X_Total = l3[COLS].values
x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l3["start"] = st
l3["scale"] = norm
l3.head()


# In[ ]:


l3.to_pickle('Store.pkl')
del l3


# # 11. State

# In[ ]:


for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:
    print(col, l2[col].nunique())


# In[ ]:


X_Total = l2[COLS].values
x1 = (X_Total>0).cumsum(1)
x1 = x1>0
st = x1.argmax(1)
den = 1941 - st - 2
diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])
norm = diff.sum(1) / den


# In[ ]:


l2["start"] = st
l2["scale"] = norm
l2.head()


# In[ ]:


l2.to_pickle('State.pkl')
del l2


# In[ ]:





# In[ ]:




