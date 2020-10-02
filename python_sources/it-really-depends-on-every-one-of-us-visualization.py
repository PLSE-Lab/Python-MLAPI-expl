#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### This is an analysis kernel that gives some insight with regard to how much impact administrative orders have on the development of current situation through data visualization. I hope this kernel can help people better understand our responsibility and power in this pendemic.     
# If you like my plot, please upvote:)

# In[ ]:


import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import gc


# In[ ]:


train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
train['Province_State']=train['Province_State'].fillna('Unknown')
def keepmonthday(x):
    x_seg=x.split('-')
    return x_seg[1]+'-'+x_seg[2]
train['Date']=train['Date'].apply(lambda x: keepmonthday(x))


# Here I included a data from https://www.kaggle.com/koryto/countryinfo. This dataset includes many awesome features and updates regularly. I will only use features that related to the start date of administrative orders in this kernel.

# In[ ]:


compre_df = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")
compre_df['region']=compre_df['region'].fillna('Unknown')


# In[ ]:


nanidex_compre=compre_df[['quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].isna().all(axis=1)


# In[ ]:


nonna_compre_df=compre_df.loc[~nanidex_compre,['region', 'country','quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].reset_index(drop=True)


# In[ ]:


train=train.merge(nonna_compre_df, how='left', left_on=['Country_Region','Province_State'], right_on=['country','region'])
train.drop(['country','region'], axis=1, inplace=True)


# In[ ]:


nanidx_train=train[['quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].isna().all(axis=1)


# In[ ]:


nonna_train=train.loc[~nanidx_train,:].reset_index(drop=True)


# In[ ]:


compare_cols=['quarantine', 'schools','publicplace', 'gathering', 'nonessential']


# In[ ]:


nonna_train['measures']=nonna_train[compare_cols].notna().sum(axis=1)


# In[ ]:


def keepmonthday2(x):
    if x is np.nan:
        return x
    else:
        x_seg=x.split('/')
        return format(int(x_seg[0]), '02')+'-'+format(int(x_seg[1]), '02')


# In[ ]:


for c in compare_cols:
    nonna_train[c]=nonna_train[c].apply(lambda x: keepmonthday2(x))


# In[ ]:


nonna_train_group=nonna_train.groupby(['Country_Region','Province_State'])


# In[ ]:


import matplotlib.transforms as mtransforms
fig_row=2
numfigs=len(nonna_train_group)
fig,ax=plt.subplots(int(np.ceil(numfigs/fig_row)),fig_row,figsize=(18,int(np.ceil(numfigs/fig_row))*(18/fig_row)))
legendlist=[]
for i,agroup in enumerate(nonna_train_group):
    #print(agroup[0])
    ridx=int(i//fig_row)
    cidx=int(i%fig_row)
    mycolumns=agroup[1][['Date', 'ConfirmedCases']]
    aline=mycolumns.set_index('Date')
    ax[ridx,cidx].plot(aline)
    row_interest=agroup[1].iloc[0,:]
    for c in compare_cols:
        xvalue=row_interest[c]
        if xvalue is np.nan:
            yvalue=np.nan
        else:
            yvalue=mycolumns.loc[mycolumns['Date']==xvalue,'ConfirmedCases']
        ax[ridx,cidx].plot(xvalue,yvalue,marker='o', markersize=8, label=c)
        if c=='gathering':
            txt=row_interest['gatheringlimit']
            if txt is np.nan:
                print(txt is np.nan)
                txt=str(int(txt))
            trans_offset = mtransforms.offset_copy(ax[ridx,cidx].transData, fig=fig,
                                       x=-0.2, y=0.10, units='inches')
            ax[ridx,cidx].text(xvalue,yvalue,txt, transform=trans_offset)
    plt.setp(ax[ridx,cidx].xaxis.get_majorticklabels(), rotation=45)
    ax[ridx,cidx].set_title('{}_{}'.format(agroup[0][0],agroup[0][1]))
    ax[ridx,cidx].set_xlabel('Date')
    ax[ridx,cidx].set_ylabel('ConfirmedCases')
    ax[ridx,cidx].legend(loc='upper left')
fig.tight_layout() 
fig.show()


# Under the assumption that all data are right, it is quite clear from our figures that administative orders don't help much in bending the curve. In few figures, curve had a litte twist around the date of orders, however the changes are too ambiguaous to be judged as a significant connection. Does this mean that enforing quarantine, closing schools, shutting down public place, limiting gathering and close none essential business are useless? Of cause not, from what we heard and watched in news throughout the pendemic, it is known that not everyone is following the guide. Administative orders are declarations. It depends on each person to put them in to action.    
# 
# COVID-19's character of spread before showing symptom tests our responsibility to our community. When we do not have syptom, will we still willing to restrict our activity to protect others. When we look at two countries that bend the curve Hubei, China and South Korea. China has strong administative action power which we can assume that quarantine order was enforced. That is staying home, shutting down none essential, social distancing. Therefore the curve bend later on. South Korea only had a closing school order. But we know they are rigorously follow the guide from the news. They bend the curve too. The take away here is that it's not what we say but do that matters. And one irresponsible carrier can infect a bunch. 
# 
# Another thing is that this gonna be a long fight. Even under strong measures South Korea hasn't reach the flat point. And it tooks a long time after the quarantine order did Hubei, China reach the flat point. It takes some while for people's action to take effect.   
# 
# To sum up, everyone needs chip in following the guide of local or global health authority. Please think a little more about others when you can't stand the boring and loneliness of staying home. Make some change to your social habit for the sake of others. Lastly let's have more patience and persistency, because we will win this fight eventually.
# 
# ##### Stay safe everyone. Good luck on the competition and I will see you in the next analysis.
