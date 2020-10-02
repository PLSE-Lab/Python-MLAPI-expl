#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# graphs
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16, 9


# In[ ]:


# read csv
cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', index_col='date', parse_dates=True)
spr = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
eva = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')


# In[ ]:


df01 = eva.loc[:, 'd_1':].sum()
df02 = eva.groupby('state_id').sum()
df03 = eva.groupby('store_id').sum()
df04 = eva.groupby('cat_id').sum()
df05 = eva.groupby('dept_id').sum()
df06 = eva.groupby(['state_id', 'cat_id']).sum()
df07 = eva.groupby(['state_id', 'dept_id']).sum()
df08 = eva.groupby(['store_id', 'cat_id']).sum()
df09 = eva.groupby(['store_id', 'dept_id']).sum()

# FOODS_3_090 as an example for lower level
df10 = eva[eva['item_id']=='FOODS_3_090'].groupby('item_id').sum()
df11 = eva[eva['item_id']=='FOODS_3_090'].groupby('state_id').sum()
df12 = eva[eva['item_id']=='FOODS_3_090'].drop(['id', 'dept_id','cat_id', 'state_id'], axis=1).set_index(['item_id','store_id'])


# # Graph for level 01 and 02
# * Sales volume has been going up since 2011.
# * We see five days of Christmas with low sales volume. It implies Walmart closes stores on Christmas.
# * CA has largest sales volume among three states but the CA includes four stores whereas others three each.

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2, ylim=[-1000, 30000])
ax3 = fig.add_subplot(2, 2, 3, ylim=[-1000, 30000])
ax4 = fig.add_subplot(2, 2, 4, ylim=[-1000, 30000])

t = cal.index[0:1941]

c1,c2,c3,c4 = 'blue','green','red','black'
l1,l2,l3,l4 = 'LV1', 'LV2 CA', 'LV2 TX', 'LV2 WI'

ax1.plot(t, df01, color=c1, label=l1)
ax2.plot(t, df02.loc['CA'], color=c2, label=l2)
ax3.plot(t, df02.loc['TX'], color=c3, label=l3)
ax4.plot(t, df02.loc['WI'], color=c4, label=l4)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# # Graph for level 3
# * We can see jumps in WI_1 and WI_2 when it spent two years.
# * We also see a jump in CA_2 in mid-2016.

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(5, 2, 1)
ax2 = fig.add_subplot(5, 2, 2)
ax3 = fig.add_subplot(5, 2, 3)
ax4 = fig.add_subplot(5, 2, 4)
ax5 = fig.add_subplot(5, 2, 5)
ax6 = fig.add_subplot(5, 2, 6)
ax7 = fig.add_subplot(5, 2, 7)
ax8 = fig.add_subplot(5, 2, 8)
ax9 = fig.add_subplot(5, 2, 9)
ax10 = fig.add_subplot(5, 2, 10)

t = cal.index[0:1941]
idx = df03.index

c1,c2,c3 = 'blue','green','red'

ax1.plot(t, df03.loc[idx[0]], color=c1, label=idx[0])
ax2.plot(t, df03.loc[idx[1]], color=c1, label=idx[1])
ax3.plot(t, df03.loc[idx[2]], color=c1, label=idx[2])
ax4.plot(t, df03.loc[idx[3]], color=c1, label=idx[3])
ax5.plot(t, df03.loc[idx[4]], color=c2, label=idx[4])
ax6.plot(t, df03.loc[idx[5]], color=c2, label=idx[5])
ax7.plot(t, df03.loc[idx[6]], color=c2, label=idx[6])
ax8.plot(t, df03.loc[idx[7]], color=c3, label=idx[7])
ax9.plot(t, df03.loc[idx[8]], color=c3, label=idx[8])
ax10.plot(t, df03.loc[idx[9]], color=c3, label=idx[9])


ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')
ax10.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()
del idx


# * Here we see 28 days moving average.
# * There are some stores with stable sales growth such as CA_1, CA_4 and TX_3.
# * We also see some stores reaching plateau such as CA_3.
# * We see interesting patterns of sales volume change in CA_2, WI_1 and WI_2.
# * Some seems to have seasonality and some move at random.

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(5, 2, 1)
ax2 = fig.add_subplot(5, 2, 2)
ax3 = fig.add_subplot(5, 2, 3)
ax4 = fig.add_subplot(5, 2, 4)
ax5 = fig.add_subplot(5, 2, 5)
ax6 = fig.add_subplot(5, 2, 6)
ax7 = fig.add_subplot(5, 2, 7)
ax8 = fig.add_subplot(5, 2, 8)
ax9 = fig.add_subplot(5, 2, 9)
ax10 = fig.add_subplot(5, 2, 10)

t = cal.index[0:1941]
idx = df03.index

c1,c2,c3 = 'blue','green','red'

ax1.plot(t, df03.loc[idx[0]].rolling(28).mean(), color=c1, label=idx[0])
ax2.plot(t, df03.loc[idx[1]].rolling(28).mean(), color=c1, label=idx[1])
ax3.plot(t, df03.loc[idx[2]].rolling(28).mean(), color=c1, label=idx[2])
ax4.plot(t, df03.loc[idx[3]].rolling(28).mean(), color=c1, label=idx[3])
ax5.plot(t, df03.loc[idx[4]].rolling(28).mean(), color=c2, label=idx[4])
ax6.plot(t, df03.loc[idx[5]].rolling(28).mean(), color=c2, label=idx[5])
ax7.plot(t, df03.loc[idx[6]].rolling(28).mean(), color=c2, label=idx[6])
ax8.plot(t, df03.loc[idx[7]].rolling(28).mean(), color=c3, label=idx[7])
ax9.plot(t, df03.loc[idx[8]].rolling(28).mean(), color=c3, label=idx[8])
ax10.plot(t, df03.loc[idx[9]].rolling(28).mean(), color=c3, label=idx[9])


ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')
ax10.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()
del idx


# In[ ]:


df03.mean(axis=1)


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

t = cal.index[0:1941]
idx = df04.index

c1,c2,c3 = 'blue','green','red'

ax1.plot(t, df04.loc[idx[0]], color=c1, label=idx[0])
ax2.plot(t, df04.loc[idx[1]], color=c2, label=idx[1])
ax3.plot(t, df04.loc[idx[2]], color=c3, label=idx[2])

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()
del idx


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(4, 2, 1)
ax2 = fig.add_subplot(4, 2, 2)
ax3 = fig.add_subplot(4, 2, 3)
ax4 = fig.add_subplot(4, 2, 5)
ax5 = fig.add_subplot(4, 2, 6)
ax6 = fig.add_subplot(4, 2, 7)
ax7 = fig.add_subplot(4, 2, 8)

t = cal.index[0:1941]
idx = df05.index

c1,c2,c3 = 'blue','green','red'

ax1.plot(t, df05.loc[idx[0]], color=c1, label=idx[0])
ax2.plot(t, df05.loc[idx[1]], color=c1, label=idx[1])
ax3.plot(t, df05.loc[idx[2]], color=c1, label=idx[2])
ax4.plot(t, df05.loc[idx[3]], color=c2, label=idx[3])
ax5.plot(t, df05.loc[idx[4]], color=c2, label=idx[4])
ax6.plot(t, df05.loc[idx[5]], color=c3, label=idx[5])
ax7.plot(t, df05.loc[idx[6]], color=c3, label=idx[6])

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()
del idx


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

t = cal.index[0:1941]
c1,c2,c3 = 'blue','green','red'
l1,l2,l3 = 'FOODS', 'HOBBIES', 'HOUSEHOLD'
l4,l5,l6 = 'CA_', 'TX_', 'WI_'

ax1.plot(t, df06.loc[('CA', 'FOODS')], color=c1, label=l4 + l1)
ax2.plot(t, df06.loc[('CA', 'HOBBIES')], color=c2, label=l4 + l2)
ax3.plot(t, df06.loc[('CA', 'HOUSEHOLD')], color=c3, label=l4 + l3)
ax4.plot(t, df06.loc[('TX', 'FOODS')], color=c1, label=l5+l1)
ax5.plot(t, df06.loc[('TX', 'HOBBIES')], color=c2, label=l5+l2)
ax6.plot(t, df06.loc[('TX', 'HOUSEHOLD')], color=c3, label=l5+l3)
ax7.plot(t, df06.loc[('WI', 'FOODS')], color=c1, label=l6+l1)
ax8.plot(t, df06.loc[('WI', 'HOBBIES')], color=c2, label=l6+l2)
ax9.plot(t, df06.loc[('WI', 'HOUSEHOLD')], color=c3, label=l6+l3)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

t = cal.index[0:1941]
c1,c2,c3 = 'blue','green','red'
l1,l2,l3 = 'FOODS_1', 'HOBBIES_1', 'HOUSEHOLD_1'
l4,l5,l6 = 'CA_', 'TX_', 'WI_'

ax1.plot(t, df07.loc[('CA', l1)], color=c1, label=l4 + l1)
ax2.plot(t, df07.loc[('CA', l2)], color=c2, label=l4 + l2)
ax3.plot(t, df07.loc[('CA', l3)], color=c3, label=l4 + l3)
ax4.plot(t, df07.loc[('TX', l1)], color=c1, label=l5+l1)
ax5.plot(t, df07.loc[('TX', l2)], color=c2, label=l5+l2)
ax6.plot(t, df07.loc[('TX', l3)], color=c3, label=l5+l3)
ax7.plot(t, df07.loc[('WI', l1)], color=c1, label=l6+l1)
ax8.plot(t, df07.loc[('WI', l2)], color=c2, label=l6+l2)
ax9.plot(t, df07.loc[('WI', l3)], color=c3, label=l6+l3)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

t = cal.index[0:1941]
c1,c2,c3 = 'blue','green','red'
l1,l2,l3 = 'FOODS', 'HOBBIES', 'HOUSEHOLD'
l4,l5,l6 = 'CA_1', 'TX_1', 'WI_1'

ax1.plot(t, df08.loc[(l4, l1)], color=c1, label=l4+l1)
ax2.plot(t, df08.loc[(l4, l2)], color=c2, label=l4+l2)
ax3.plot(t, df08.loc[(l4, l3)], color=c3, label=l4+l3)
ax4.plot(t, df08.loc[(l5, l1)], color=c1, label=l5+l1)
ax5.plot(t, df08.loc[(l5, l2)], color=c2, label=l5+l2)
ax6.plot(t, df08.loc[(l5, l3)], color=c3, label=l5+l3)
ax7.plot(t, df08.loc[(l6, l1)], color=c1, label=l6+l1)
ax8.plot(t, df08.loc[(l6, l2)], color=c2, label=l6+l2)
ax9.plot(t, df08.loc[(l6, l3)], color=c3, label=l6+l3)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

t = cal.index[0:1941]
c1,c2,c3 = 'blue','green','red'
l1,l2,l3 = 'FOODS_3', 'HOBBIES_2', 'HOUSEHOLD_2'
l4,l5,l6 = 'CA_1', 'TX_1', 'WI_1'

ax1.plot(t, df09.loc[(l4, l1)], color=c1, label=l4+l1)
ax2.plot(t, df09.loc[(l4, l2)], color=c2, label=l4+l2)
ax3.plot(t, df09.loc[(l4, l3)], color=c3, label=l4+l3)
ax4.plot(t, df09.loc[(l5, l1)], color=c1, label=l5+l1)
ax5.plot(t, df09.loc[(l5, l2)], color=c2, label=l5+l2)
ax6.plot(t, df09.loc[(l5, l3)], color=c3, label=l5+l3)
ax7.plot(t, df09.loc[(l6, l1)], color=c1, label=l6+l1)
ax8.plot(t, df09.loc[(l6, l2)], color=c2, label=l6+l2)
ax9.plot(t, df09.loc[(l6, l3)], color=c3, label=l6+l3)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# * We see stable sales volumen in FOODS_1 in WI since 2011, but we do see jumps in FOODS_2 and FOODS_3 in WI.
# * Other than FOODS_2 in WI, we see stable demand in FOODS. On the other hand, we see spikes in HOBBIES.

# # Single product level
# * Above we see aggregated data and here and below we see separated data in single products.
# * Below wee see a lot of sales lacks probely according to seasonality whereas we often see stable sales in aggregated level.
# * Predicting lower level sales one by one would be a tough project.

# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

t = cal.index[0:1941]

c1,c2,c3,c4 = 'blue','green','red','black'
l1,l2,l3,l4 = 'LV10', 'LV11 CA', 'LV11 TX', 'LV11 WI'

ax1.plot(t, df10.T, color=c1, label=l1)
ax2.plot(t, df11.loc['CA'], color=c2, label=l2)
ax3.plot(t, df11.loc['TX'], color=c3, label=l3)
ax4.plot(t, df11.loc['WI'], color=c4, label=l4)

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# In[ ]:


fig = plt.figure()

ax1 = fig.add_subplot(5, 2, 1)
ax2 = fig.add_subplot(5, 2, 2)
ax3 = fig.add_subplot(5, 2, 3)
ax4 = fig.add_subplot(5, 2, 4)
ax5 = fig.add_subplot(5, 2, 5)
ax6 = fig.add_subplot(5, 2, 6)
ax7 = fig.add_subplot(5, 2, 7)
ax8 = fig.add_subplot(5, 2, 8)
ax9 = fig.add_subplot(5, 2, 9)
ax10 = fig.add_subplot(5, 2, 10)

t = cal.index[0:1941]
c1,c2,c3 = 'blue','green','red'
idx = eva['store_id'].unique()
item = 'FOODS_3_090'

ax1.plot(t, df12.loc[(item, idx[0])], color=c1, label=item + idx[0])
ax2.plot(t, df12.loc[(item, idx[1])], color=c1, label=item + idx[1])
ax3.plot(t, df12.loc[(item, idx[2])], color=c1, label=item + idx[2])
ax4.plot(t, df12.loc[(item, idx[3])], color=c1, label=item + idx[3])
ax5.plot(t, df12.loc[(item, idx[4])], color=c2, label=item + idx[4])
ax6.plot(t, df12.loc[(item, idx[5])], color=c2, label=item + idx[5])
ax7.plot(t, df12.loc[(item, idx[6])], color=c2, label=item + idx[6])
ax8.plot(t, df12.loc[(item, idx[7])], color=c3, label=item + idx[7])
ax9.plot(t, df12.loc[(item, idx[8])], color=c3, label=item + idx[8])
ax10.plot(t, df12.loc[(item, idx[9])], color=c3, label=item + idx[9])

ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper left')
ax3.legend(loc = 'upper left')
ax4.legend(loc = 'upper left')
ax5.legend(loc = 'upper left')
ax6.legend(loc = 'upper left')
ax7.legend(loc = 'upper left')
ax8.legend(loc = 'upper left')
ax9.legend(loc = 'upper left')
ax10.legend(loc = 'upper left')

fig.tight_layout() 
plt.show()


# In[ ]:




