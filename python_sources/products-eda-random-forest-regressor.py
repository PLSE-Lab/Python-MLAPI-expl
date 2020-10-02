#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from string import ascii_letters
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import linkage,dendrogram


# In[ ]:


train_df = pd.read_csv("../input/mercaritest/train.tsv", delimiter='\t')
test_df = pd.read_csv("../input/mercaritest/test.tsv", delimiter='\t')


# In[ ]:


train_df["brand_name"] = train_df["brand_name"].fillna("NoBrand")
test_df["brand_name"] = test_df["brand_name"].fillna("NoBrand")


# In[ ]:


train_df["category_name"] = train_df["category_name"].fillna("No/No/No")
test_df["category_name"] = test_df["category_name"].fillna("No/No/No")


# In[ ]:


target = train_df.price
train_features = train_df.drop('price',axis=1)
test = test_df
train_features['is_train'] = 1
test['is_train'] = 0
train_test = pd.concat([train_features,test],axis=0)


# In[ ]:


def split(txt):
    try :
        return txt.split("/")
    except :
        return ("No Label", "No Label", "No Label")


# In[ ]:


train_test['general_category']='' 
train_test['subcategory_1'] = '' 
train_test['subcategory_2'] = ''


# In[ ]:


train_test['general_category'],train_test['subcategory_1'],train_test['subcategory_2'] = zip(*train_test['category_name'].apply(lambda x: split(x)))


# In[ ]:


print(train_test['general_category'].nunique(),
train_test['subcategory_1'].nunique(),
train_test['subcategory_2'].nunique())


# In[ ]:


train_test.head()


# In[ ]:


train_df['general_category']='' 
train_df['subcategory_1'] = '' 
train_df['subcategory_2'] = ''
train_df['general_category'], train_df['subcategory_1'], train_df['subcategory_2'] = zip(*train_df['category_name'].apply(lambda x: split(x)))


# In[ ]:


GCP1 = train_df.groupby(["general_category"])["price"].mean()
print(GCP1)


# In[ ]:


train_test['GC']=0
train_test.loc[ (train_test.general_category=='Beauty'),'GC']=19.7
train_test.loc[ (train_test.general_category=='Electronics'),'GC']=35.2
train_test.loc[ (train_test.general_category=='Handmade'),'GC']=18.2
train_test.loc[ (train_test.general_category=='Home'),'GC']=24.5
train_test.loc[ (train_test.general_category=='Kids'),'GC']=20.6
train_test.loc[ (train_test.general_category=='Men'),'GC']=34.7
train_test.loc[ (train_test.general_category=='No Label'),'GC']=25.4
train_test.loc[ (train_test.general_category=='Other'),'GC']=20.8
train_test.loc[ (train_test.general_category=='Sports & Outdoors'),'GC']=25.5
train_test.loc[ (train_test.general_category=='Vintage & Collectibles'),'GC']=27.3
train_test.loc[ (train_test.general_category=='Women'),'GC']=28.9


# In[ ]:


GCP2 = train_df.groupby(["subcategory_1"])["price"].mean()
print(GCP2>50)


# In[ ]:


train_test['SC']=0
train_test.loc[(train_test.subcategory_1=="Bags and Purses"),'SC']=1
train_test.loc[(train_test.subcategory_1=="Cameras & Photography"),'SC']=1
train_test.loc[(train_test.subcategory_1=="Computers & Tablets"),'SC']=1
train_test.loc[(train_test.subcategory_1=="Strollers"),'SC']=1
train_test.loc[(train_test.subcategory_1=="Women's Handbags"),'SC']=1


# In[ ]:


train_test.item_description = train_test.item_description.fillna('NA')


# In[ ]:


train_test['des_len'] = train_test.item_description.apply(lambda x : len(x))


# In[ ]:


train_test['name_len'] = train_test.name.apply(lambda x : len(x))


# In[ ]:


train_test.head()


# In[ ]:


GCP3 = train_df.groupby(["subcategory_2"])["price"].mean()
print(GCP3.mean())


# In[ ]:


train_test['brand_name'].nunique()


# In[ ]:


train_df.groupby(['brand_name'])["price"].mean().sort_values(ascending=False)[:50]


# In[ ]:


x = train_df.groupby(['brand_name'])["item_condition_id"].mean()


# In[ ]:


y = train_df.groupby(['brand_name'])["price"].mean()


# In[ ]:


plt.scatter(x,y)


# In[ ]:


train_test['NiceCondition']=0
train_test.loc[(train_test.item_condition_id==2),'NiceCondition']=1
train_test.loc[(train_test.item_condition_id==3),'NiceCondition']=1


# In[ ]:


x = train_df.groupby(['brand_name'])["shipping"].mean()
y = train_df.groupby(['brand_name'])["price"].mean()


# In[ ]:


plt.scatter(x,y)


# In[ ]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
le = LabelEncoder()

le.fit(np.hstack([train_test.name]))
train_test['name_id'] = le.transform(train_test.name)

le.fit(np.hstack([train_test.subcategory_2]))
train_test['category2'] = le.transform(train_test.subcategory_2)

le.fit(np.hstack([train_test.brand_name]))
train_test['brand'] = le.transform(train_test.brand_name)

le.fit(np.hstack([train_test.subcategory_1]))
train_test['category1'] = le.transform(train_test.subcategory_1)

le.fit(np.hstack([train_test.general_category]))
train_test['general_category1'] = le.transform(train_test.general_category)


# In[ ]:


train_test.head()


# In[ ]:


tr = train_test.drop(['brand_name','category_name','is_train','item_description','name','shipping','SC','test_id','train_id'
                    ,'general_category','subcategory_1','general_category1','subcategory_2'],axis=1)


# In[ ]:


model = KMeans(n_clusters = 12)
scaler = StandardScaler()
model.fit(tr)
labels = model.predict(tr)
cluster = make_pipeline(scaler,model)


# In[ ]:


train_test['cluster']=labels


# In[ ]:


clusters  = pd.get_dummies(train_test['cluster'],prefix='Cluster',drop_first=False)

train_test = pd.concat([train_test,clusters],axis=1).drop('cluster',axis=1)


# In[ ]:


train_test.head()


# In[ ]:


'''from keras.preprocessing.text import Tokenizer'''


# In[ ]:


'''text = np.hstack([train_test.item_description.str.lower(), 
                      train_test.name.str.lower()])'''


# In[ ]:


'''tok_raw = Tokenizer()
tok_raw.fit_on_texts(text)
train_test["seq_item_description"] = tok_raw.texts_to_sequences(train_test.item_description.str.lower())
train_test["seq_name"] = tok_raw.texts_to_sequences(train_test.name.str.lower())'''


# In[ ]:


conditions  = pd.get_dummies(train_test['item_condition_id'],prefix='Condition',drop_first=False)

train_test = pd.concat([train_test,conditions],axis=1).drop('item_condition_id',axis=1)


# In[ ]:


shippings  = pd.get_dummies(train_test['shipping'],prefix='Shipping',drop_first=True)

train_test = pd.concat([train_test,shippings],axis=1).drop('shipping',axis=1)


# In[ ]:


corr = train_test.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


train_test.head()


# In[ ]:


train_test.info()


# In[ ]:


train = train_test[train_test.is_train == 1].drop(['brand_name','category_name','is_train','item_description',
                        'name','test_id','train_id','general_category','subcategory_1','subcategory_2','general_category1','NiceCoondition'],axis=1)

test = train_test[train_test.is_train == 0].drop(['is_train','test_id','train_id','brand_name','category_name','item_description',
                        'name','general_category','subcategory_1','subcategory_2','general_category1','NiceCondition'],axis=1)


# In[ ]:


len(train_test)


# In[ ]:


len(train)


# In[ ]:


len(test)


# In[ ]:


len(target)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=150,n_jobs=-1,min_samples_leaf=5, max_features=0.5)
model.fit(train,target)


# In[ ]:


predict_test= model.predict(test)
test['test_id'] = range(len(test))
submission = pd.concat([test[['test_id']],pd.DataFrame(predict_test)],axis=1)
submission.columns = ['test_id', 'price']
submission.to_csv('submission.csv',index=False)


# In[ ]:




