#!/usr/bin/env python
# coding: utf-8

# I will be updating the notebook with more analysis.It is working notebook.
# 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


suppliers = pd.read_csv("/kaggle/input/prozorro-public-procurement-dataset/Suppliers.csv")
procure_df = pd.read_csv("/kaggle/input/prozorro-public-procurement-dataset/Competitive_procurements.csv")


# In[ ]:


## getting to know shape and columns of data
print("-----------------------------------------------")
print("column names in both data frames")
print("-----------------------------------------------")
print(procure_df.columns,suppliers.columns)
print("-----------------------------------------------")

## checking level of data for suppliers df as mentioned in description that it is at lot_id and date level
print(f"""
Competitive_procurements_shape : {procure_df.shape}
supplier_shape : {suppliers.shape}
supplier lot_id shape : {suppliers[['lot_announce_date','lot_id']].drop_duplicates().shape}
    """)


# In[ ]:


## checking for missing values
plt.figure(figsize=(15,4))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1,2,1)
(procure_df.isnull().sum()*100/procure_df.shape[0]).plot(kind='barh')
plt.title("Competitive_procurements")
plt.xlabel("%Misssing")
plt.grid()

plt.subplot(1,2,2)
(suppliers.isnull().sum()*100/suppliers.shape[0]).plot(kind='barh')
plt.title("Suppliers")
plt.xlabel("%Misssing")
plt.grid()


# In[ ]:


## are all competative procruments presents in suppliers df ?
supplier_compet = suppliers[suppliers['lot_competitiveness']==1].lot_id.unique()
competative = procure_df.lot_id.unique()
print(f"""
supplier lot_ids : {len(set(supplier_compet))}
competitive lot_ids : {len(set(competative))}
common lot_ids : {len(set(supplier_compet) & set(competative))}
""")


# In[ ]:


## how #procurements follow across years ? 
## what is the proportion in two categories of competitive and no-competitive ?

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
suppliers.groupby(['lot_announce_year']).lot_id.nunique().plot(kind='bar')
plt.ylabel("#lot_id unique")
plt.subplot(1,2,2)
plot_df = pd.DataFrame(suppliers.groupby(['lot_announce_year','lot_competitiveness']).lot_id.nunique()).reset_index()
sns.barplot(x='lot_announce_year',y='lot_id',hue='lot_competitiveness',data=plot_df)

print("""
1. Total #procurements are increasing but 
2. #competitive procurements are not showing significant change in terms of quantity
""")


# In[ ]:


## is there any relation of #procurements in accordance with month

suppliers['lot_announce_date'] = pd.to_datetime(suppliers['lot_announce_date'] )
suppliers['lot_announce_month'] = suppliers['lot_announce_date'].dt.month

plt.figure(figsize=(15,4))
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,2,1)
suppliers.groupby(['lot_announce_month']).lot_id.nunique().plot(kind='bar')
plt.ylabel("#lot_id unique")
plt.subplot(2,2,2)
plot_df = pd.DataFrame(suppliers.groupby(['lot_announce_month','lot_competitiveness']).lot_id.nunique()).reset_index()
sns.barplot(x='lot_announce_month',y='lot_id',hue='lot_competitiveness',data=plot_df)
plt.subplot(2,2,3)
plot_df = pd.DataFrame(suppliers.groupby(['lot_announce_month','lot_announce_year']).lot_id.nunique()).reset_index()
sns.lineplot(x='lot_announce_month',y='lot_id',hue='lot_announce_year',data=plot_df)
plt.grid()


# 1. At the end of the year there is sharp increment in #procurements , may be govt. tries to meet the requirements by the end of the year.
# 2. Again there is unsignificant change in competitive procurements

# In[ ]:


## distribution of lot_procur_type 

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
suppliers.groupby(['lot_procur_type']).lot_id.nunique().plot(kind='barh')
plt.ylabel("#lot_id unique")
plt.subplot(1,2,2)
plot_df = pd.DataFrame(suppliers.groupby(['lot_announce_year','lot_procur_type']).lot_id.nunique()).reset_index()
sns.barplot(x='lot_announce_year',y='lot_id',hue='lot_procur_type',data=plot_df)


# 1. Reporting procurements are highest in number
# 2. there is growth in #Above Threshold UA 

# ### what are various types of procurement are like type of goods?

# In[ ]:


def lot_cpv_decompose(col):
    unique_lot_cpv = {val: idx for idx,val in enumerate(suppliers[col].unique())}
    lot_cpv_df = pd.DataFrame(unique_lot_cpv,index=[0]).T.reset_index()
    lot_cpv_df.columns=[col,f'{col}_id']
    lot_cpv_df[f'{col}_0'] = lot_cpv_df[col].apply(lambda x: x.split("_")[0])
    lot_cpv_df[f'{col}_1'] = lot_cpv_df[col].apply(lambda x: x.split("_")[1])
    lot_cpv_df[f'{col}_00'] = lot_cpv_df[col].apply(lambda x: x.split("_")[0].split("-")[0])
    lot_cpv_df[f'{col}_01'] = lot_cpv_df[col].apply(lambda x: 99 if len(x.split("_")[0].split("-"))<2 
                                                                else x.split("_")[0].split("-")[1])
    return lot_cpv_df,unique_lot_cpv


# In[ ]:


lot_cpv_df,lot_cpv_dict = lot_cpv_decompose("lot_cpv")
lot_cpv_4_df,lot_cpv_4_dict = lot_cpv_decompose("lot_cpv_4_digs")
lot_cpv_2_df,lot_cpv_2_dict = lot_cpv_decompose("lot_cpv_2_digs")

suppliers['lot_cpv_id'] = suppliers.lot_cpv.apply(lambda x : lot_cpv_dict[x])
suppliers['lot_cpv_4_digs_id'] = suppliers.lot_cpv_4_digs.apply(lambda x : lot_cpv_4_dict[x])
suppliers['lot_cpv_2_digs_id'] = suppliers.lot_cpv_2_digs.apply(lambda x : lot_cpv_2_dict[x])


# ##### Looking for wholistic number distribution for different supplies

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['lot_cpv_id']).lot_id.nunique()).reset_index()
plot_df.columns = ['lot_cpv_id','#lot_id']
plot_df = pd.merge(plot_df,lot_cpv_df,on=['lot_cpv_id'])
plot_df['lot_cpv_id'] = plot_df['lot_cpv_id'].astype(str)

plt.barh("lot_cpv_1","#lot_id",data=plot_df.sort_values(by=['#lot_id'],ascending=False).head(20))
plt.xticks(rotation=90)
plt.title("lot_cpv_1")


# 1. Most of the procurements are for Overhaul & Refurbishment Work , Electricity and other building completion work - In short construction supplies
# 2. Second highest is Office equipments and supplies
# 3. Petroleum and Pharmaceutical supplies

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['lot_cpv_4_digs_id']).lot_id.nunique()).reset_index()
plot_df.columns = ['lot_cpv_4_digs_id','#lot_id']
plot_df = pd.merge(plot_df,lot_cpv_4_df,on=['lot_cpv_4_digs_id'])
plot_df['lot_cpv_4_digs_id'] = plot_df['lot_cpv_4_digs_id'].astype(str)

plt.barh("lot_cpv_4_digs_1","#lot_id",data=plot_df.sort_values(by=['#lot_id'],ascending=False).head(20))
plt.xticks(rotation=90)
plt.title("lot_cpv_4_1")


# 1. Here also priority is construction and then pharmaceutical supplies

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['lot_cpv_2_digs_id']).lot_id.nunique()).reset_index()
plot_df.columns = ['lot_cpv_2_digs_id','#lot_id']
plot_df = pd.merge(plot_df,lot_cpv_2_df,on=['lot_cpv_2_digs_id'])
plot_df['lot_cpv_2_digs_id'] = plot_df['lot_cpv_2_digs_id'].astype(str)

plt.barh("lot_cpv_2_digs_1","#lot_id",data=plot_df.sort_values(by=['#lot_id'],ascending=False).head(20))
plt.xticks(rotation=90)
plt.title("lot_cpv_2_1")


# 1. Most procurments are for construction work

# #### no luck with checking on most prominent words to pull out the meaning of supplies

# In[ ]:


from collections import Counter
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

for i in lot_cpv_df['lot_cpv_01'].unique():
    plt.figure(figsize=(15,5))
    plt.subplots_adjust(wspace=0.4)
    word_ls = [i.split(" ") for i in lot_cpv_df[lot_cpv_df['lot_cpv_01']==i]['lot_cpv_1'].to_list()]
    word_ls = [j for i in word_ls for j in i ]
    word_ls = [word for word in word_ls if word not in stopwords_dict]
    counts = Counter(word_ls)
    most_ = counts.most_common(10)
    plt.subplot(1,3,1)
    plt.barh([i[0] for i in most_],[i[1] for i in most_])
    plt.title(f"lot_cpv_df {i}")

    word_ls = [i.split(" ") for i in lot_cpv_2_df[lot_cpv_2_df['lot_cpv_2_digs_01']==i]['lot_cpv_2_digs_1'].to_list()]
    word_ls = [j for i in word_ls for j in i ]
    word_ls = [word for word in word_ls if word not in stopwords_dict]
    counts = Counter(word_ls)
    most_ = counts.most_common(10)
    plt.subplot(1,3,2)
    plt.barh([i[0] for i in most_],[i[1] for i in most_])
    plt.title(f"lot_cpv_2_df {i}")


    word_ls = [i.split(" ") for i in lot_cpv_4_df[lot_cpv_4_df['lot_cpv_4_digs_01']==i]['lot_cpv_4_digs_1'].to_list()]
    word_ls = [j for i in word_ls for j in i ]
    word_ls = [word for word in word_ls if word not in stopwords_dict]
    counts = Counter(word_ls)
    most_ = counts.most_common(10)
    plt.subplot(1,3,3)
    plt.barh([i[0] for i in most_],[i[1] for i in most_])
    plt.title(f"lot_cpv_4_df {i}")


# ### let's dig into organisers and participants

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['organizer_code']).lot_id.nunique()).reset_index()
plot_df.columns = ['organizer_code','#lot_id']
plot_df.sort_values(by=['#lot_id'],ascending=False,inplace=True)
plot_df['organizer_code'] = plot_df['organizer_code'].astype(str)
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1,2,1)
plt.barh("organizer_code","#lot_id",data = plot_df.head(10))
plt.title("organizer_code")

plot_df = pd.DataFrame(suppliers.groupby(['supplier_code']).lot_id.nunique()).reset_index()
plot_df.columns = ['supplier_code','#lot_id']
plot_df.sort_values(by=['#lot_id'],ascending=False,inplace=True)
plot_df['supplier_code'] = plot_df['supplier_code'].astype(str)
plt.subplot(1,2,2)
plt.barh("supplier_code","#lot_id",data = plot_df.head(10))
plt.title("supplier_code")


# Organizer 3366078 and supplier 32490244 has made to the highest position because most of the contract in his/her quota is Reporting ones which we will see below

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['organizer_code','lot_procur_type']).lot_id.nunique()).reset_index()
plot_df.columns = ['organizer_code','lot_procur_type','#lot_id']
plot_df.sort_values(by=['lot_procur_type','#lot_id'],ascending=False,inplace=True)
plot_df['organizer_code'] = plot_df['organizer_code'].astype(str)

plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
cnt = 1
for typ in plot_df.lot_procur_type.unique():
    plt.subplot(4,2,cnt)
    cnt+=1
    plt.barh("organizer_code","#lot_id",data = plot_df[plot_df.lot_procur_type==typ].head(10))
    plt.title(typ)


# 1. looks like 34022 organizer is kind of monopoly in Above Threshold UA Defense
# 2. In negotiations 2 organisers are dominating the market 
# 3. In Above Threshold EU 3 organizers are dominating
# 4. Reporting also has two major organisers having maximum market share
# 5. Remaining ones on right side plots are uniform market shares
# 

# In[ ]:


plot_df = pd.DataFrame(suppliers.groupby(['supplier_code','lot_procur_type']).lot_id.nunique()).reset_index()
plot_df.columns = ['supplier_code','lot_procur_type','#lot_id']
plot_df.sort_values(by=['lot_procur_type','#lot_id'],ascending=False,inplace=True)
plot_df['supplier_code'] = plot_df['supplier_code'].astype(str)

plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
cnt = 1
for typ in plot_df.lot_procur_type.unique():
    plt.subplot(4,2,cnt)
    cnt+=1
    plt.barh("supplier_code","#lot_id",data = plot_df[plot_df.lot_procur_type==typ].head(10))
    plt.title(typ)


# 1. Only one peculiar thing here supplier 39273420 is holding major market in Negotiation and Above Threshold UA category
# 2. Apart from Above Threshold US Defense , there is high percentage of market share hold by 2-3 suppliers

# In[ ]:




