#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


zomato = pd.read_csv("../input/zomato-csv/zomato.csv",encoding='utf-8')
country_code = pd.read_csv("../input/country-code-data-converted-into-csv/Country-Code.csv")


# ## Data Import and Collation

# In[ ]:


json_files = glob.glob("../input/zomato-restaurants-data/*.json")


# In[ ]:


def json_to_df(file):
    master_df =  pd.DataFrame()
    with open(file) as file:
        test = json.load(file)

    master_df = pd.DataFrame()
    for i in test:
        try :
            for j in i['restaurants']:
                master_df= master_df.append(pd.DataFrame(j).transpose())
        except:
            continue
    return master_df


# In[ ]:


df_ls = []
for file in json_files:
    df_ls.append(json_to_df(file))


# In[ ]:


ls_27 = df_ls[0].columns
ls_23 = df_ls[4].columns

uncommon_cols = list(set(ls_27) ^ set(ls_23))


# In[ ]:


master_df = pd.DataFrame()
for i in range(len(df_ls)):
    if i!=4:
        print("start {}".format(len(master_df)))
        temp_df= pd.DataFrame(df_ls[i])
        master_df = master_df.append(temp_df)
        print("end {}".format(len(master_df)))
    else :
        print("start {}".format(len(master_df)))
        temp_df= pd.DataFrame(df_ls[i])
        for col in uncommon_cols:
            temp_df[col] = ""
        master_df = master_df.append(temp_df)
        print("end {}".format(len(master_df)))
    


# In[ ]:


master_df.head(2)


# In[ ]:


master_df['res_id']= master_df.apply(lambda x: x['R']['res_id'],axis=1)

master_df['rating_text'] = master_df.apply(lambda x : x['user_rating']['rating_text'],axis=1)
master_df['rating_color'] = master_df.apply(lambda x : x['user_rating']['rating_color'],axis=1)
master_df['aggregate_rating'] = master_df.apply(lambda x : x['user_rating']['aggregate_rating'],axis=1)
master_df['votes'] = master_df.apply(lambda x: x['user_rating']['votes'],axis =1)

master_df['city'] = master_df.apply(lambda x : x['location']['city'],axis =1)
master_df['locality_verbose'] = master_df.apply(lambda x : x['location']['locality_verbose'],axis =1)
master_df['locality'] = master_df.apply(lambda x : x['location']['locality'],axis =1)


# In[ ]:


cols_req = ['res_id','average_cost_for_two', 'cuisines', 'currency', 'has_online_delivery', 'has_table_booking', 
            'is_delivering_now', 'name', 'price_range','switch_to_order_menu','rating_text', 'rating_color', 'aggregate_rating',
            'votes','city','locality_verbose', 'locality']


# In[ ]:


master_df_sub = master_df[cols_req].drop_duplicates()


# In[ ]:


master_df_sub.info()


# ## Univariate

# In[ ]:


def continuous_var(x):
    sns.distplot(x)


# In[ ]:


x = [float(i) for i in list(master_df_sub['aggregate_rating'])]
continuous_var(x)


# In[ ]:


x = [float(i) for i in list(master_df_sub['votes'])]
continuous_var(x)


# In[ ]:


df = pd.DataFrame(master_df_sub['price_range'].value_counts())
df['var'] = df.index
sns.barplot(x='var',y="price_range",data =df)


# In[ ]:


df = pd.DataFrame(master_df_sub['currency'].value_counts())
df['var'] = df.index
sns.barplot(x='var',y="currency",data =df)


# In[ ]:


#cuisines   
## standardise and create new groups
f, ax = plt.subplots(figsize=(15, 6))
df = pd.DataFrame(master_df_sub['cuisines'].value_counts())
df['var'] = df.index
df = df.sort_values("cuisines",ascending=False).head(10)
sns.barplot(x='var',y="cuisines",data =df,ax=ax)


# In[ ]:


## map city to countries for seg
f, ax = plt.subplots(figsize=(15, 6))
df = pd.DataFrame(master_df_sub['city'].value_counts())
df['var'] = df.index
df = df.sort_values("city",ascending=False).head(10)
sns.barplot(x='var',y="city",data =df,ax=ax)


# In[ ]:


#rating_color 
## convert color codes to color name
f, ax = plt.subplots(figsize=(5, 6))
df = pd.DataFrame(master_df_sub['rating_color'].value_counts())
df['var'] = df.index
# df = df.sort_values("rating_color",ascending=False).head(10)
sns.barplot(x='var',y="rating_color",data =df,ax=ax)


# In[ ]:



# f, ax = plt.subplots(figsize=(5, 6))
df = pd.DataFrame(master_df_sub['has_online_delivery'].value_counts())
df['var'] = df.index
# df = df.sort_values("rating_color",ascending=False).head(10)
sns.barplot(x='var',y="has_online_delivery",data =df)


# In[ ]:



# f, ax = plt.subplots(figsize=(5, 6))
df = pd.DataFrame(master_df_sub['has_table_booking'].value_counts())
df['var'] = df.index
# df = df.sort_values("rating_color",ascending=False).head(10)
sns.barplot(x='var',y="has_table_booking",data =df)


# In[ ]:


#locality              
## map city to countries for seg
f, ax = plt.subplots(figsize=(15, 6))
df = pd.DataFrame(master_df_sub['locality'].value_counts())
df['var'] = df.index
df = df.sort_values("locality",ascending=False).head(10)
sns.barplot(x='var',y="locality",data =df,ax=ax)


# In[ ]:



# f, ax = plt.subplots(figsize=(5, 6))
df = pd.DataFrame(master_df_sub['rating_text'].value_counts())
df['var'] = df.index
# df = df.sort_values("rating_color",ascending=False).head(10)
sns.barplot(x='var',y="rating_text",data =df)


# In[ ]:





# ## Modification

# In[ ]:


master_df_sub = pd.merge(master_df_sub,zomato[['Restaurant ID','Country Code']],left_on='res_id',right_on='Restaurant ID')


# In[ ]:


master_df_sub = pd.merge(master_df_sub,country_code,on='Country Code')


# In[ ]:


color_name_dict = {'5BA829':'green','3F7E00':'dark green','FF7800':'orange','9ACD32':'yellow green','CDD614':"yellow black",
                    'FFBA00':"orange black","CBCBC8":'grey',"DE1D0F":'red'}


# In[ ]:


master_df_sub['rating_color_name'] = master_df_sub.apply(lambda x: color_name_dict[x['rating_color']],axis =1)


# In[ ]:


master_df_sub.cuisines.unique()


# ## Modifying cuisines 

# In[ ]:


# !pip install -U gensim
# !pip install --upgrade pip


# In[ ]:


import gensim


# In[ ]:


import gensim
import numpy as np
global model
model = gensim.models.KeyedVectors.load_word2vec_format('../input/google-word-to-vec-model/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[ ]:


from gensim.parsing.preprocessing import *


# In[ ]:


unique_cuisines =  master_df_sub.cuisines.unique()


# In[ ]:


def word_vec(cuisines_str):
    cuisines = cuisines_str.split(",")
    tokens = [remove_stopwords(strip_multiple_whitespaces(strip_tags(strip_punctuation(strip_numeric(i))))).split(" ") 
              for i in cuisines] 
    word_vec = []
    
    for token in tokens:
        temp_vec = []
        for sub in token:
            try :
                temp_vec.append(model[sub])
            except :
                print("NA for {}".format(sub))
        if len(temp_vec)>1:    
            avg_vec = np.mean(temp_vec,axis=0)
        elif len(temp_vec)==1 :
            avg_vec = temp_vec[0]
        else :
            avg_vec = " "
        
        word_vec.append({sub :avg_vec})
            
    return word_vec
        


# In[ ]:


uniq_cui_vec = [word_vec(i) for i in unique_cuisines]


# In[ ]:


uniq_cui_each = [j  for i in uniq_cui_vec for j in i]


# In[ ]:


uniq_dish = list(set([k.strip() for j in [i.split(",") for i in unique_cuisines] for k in j]))


# In[ ]:





# ## Univariate cont..

# In[ ]:


#rating_color 
## convert color codes to color name
f, ax = plt.subplots(figsize=(10, 6))
df = pd.DataFrame(master_df_sub['rating_color_name'].value_counts())
df['var'] = df.index
# df = df.sort_values("rating_color",ascending=False).head(10)
sns.barplot(x='var',y="rating_color_name",data =df,ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 6))
df = pd.DataFrame(master_df_sub['Country'].value_counts())
df['var'] = df.index
df = df.sort_values("Country",ascending=False).head(10)
sns.barplot(x='var',y="Country",data =df,ax=ax)


# In[ ]:





# ## Only India Data : Max records

# In[ ]:


india_df = master_df_sub[master_df_sub.Country=='India']


# In[ ]:


india_df['average_cost_for_two'] = india_df.apply(lambda x : float(x['average_cost_for_two']),axis =1)


# In[ ]:


pd.DataFrame(india_df.city.value_counts())


# ## Delhi NCR Region

# In[ ]:


## taking only top 4 citis consist of Delhi and NCR region

india_df_sub =india_df[india_df.city.isin(['New Delhi','Gurgaon','Noida','Faridabad'])]


# In[ ]:


f, ax = plt.subplots(figsize=(15, 6))
sns.catplot(x="city", y="average_cost_for_two", kind="swarm",hue="rating_text" ,data=india_df_sub,ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.catplot(x="city", y="average_cost_for_two", kind="swarm",hue="price_range" ,data=india_df_sub,ax=ax)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 6))
sns.catplot(x="city", y="average_cost_for_two", kind="violin",hue="has_online_delivery" ,data=india_df_sub,ax=ax)


# In[ ]:


#has_table_booking       
f, ax = plt.subplots(figsize=(15, 6))
sns.catplot(x="city", y="average_cost_for_two", kind="violin",hue="has_table_booking" ,data=india_df_sub,ax=ax)


# In[ ]:





# ## Clustering

# In[ ]:


india_df_sub.columns


# In[ ]:


def try_model(word):
    try :
        return model[word]
    except :
        return 'NA'


# In[ ]:


def cuisines_vec(cui_ls):
#     print(cui_ls)
    words = [i for i in cui_ls.split(",")]
#     print(words)
    words_vec = []
    for i in words:
        if len(i.split(" "))<1:
            temp_vec =  try_model(i.strip())
        else :
            temp_vec =  [try_model(j.strip()) for j in i.split(" ") if j.strip()!=""]
            temp_vec = np.mean([i for i in temp_vec if i!='NA'],axis =0)
        words_vec.append(temp_vec)
#     print((words_vec))
    return np.mean(words_vec,axis =0)


# In[ ]:


india_df_sub["cuisines_vec"] =  india_df_sub.apply(lambda x : cuisines_vec(x['cuisines']),axis =1)


# In[ ]:


def string_to_vec(rating_text):
    ls = [j.strip() for i in rating_text.split(" ") for j in i.split("-") if len(j.strip())>1]
    words_vec = []
    for word in ls:
        try :
            words_vec.append(model[word])
        except : continue
    return np.mean(words_vec,axis =0)


# In[ ]:


india_df_sub["rating_vec"] = india_df_sub.apply(lambda x : string_to_vec(x['rating_text']),axis =1)
india_df_sub["city_vec"] = india_df_sub.apply(lambda x : string_to_vec(x['city']),axis =1)
india_df_sub["locality_vec"] = india_df_sub.apply(lambda x : string_to_vec(x['locality']),axis =1)


# In[ ]:


india_df_sub["rating_color_vec"] = india_df_sub.apply(lambda x : string_to_vec(x['rating_color_name']),axis =1)


# In[ ]:


from sklearn import preprocessing

def normaliz(col):
    x = np.array(col).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled


# In[ ]:


india_df_sub['avg_cost'] = normaliz(india_df_sub.average_cost_for_two)
india_df_sub['price_range_norm'] = normaliz(india_df_sub.price_range)
india_df_sub['agg_rate_norm'] = normaliz(india_df_sub.aggregate_rating)
india_df_sub['votes_norm'] = normaliz(india_df_sub.votes)


# In[ ]:


cols_cluster = ['has_online_delivery', 'has_table_booking', 'cuisines_vec', 'rating_vec', 
                'city_vec', 'locality_vec', 'avg_cost','rating_color_vec', 'price_range_norm', 
                'agg_rate_norm', 'votes_norm']


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


def vec_column(columns_ls):
    ls = []
    for col in columns_ls:
        try :ls.append(col.tolist())
        except : ls.append([col])    
    
    array_vec = np.hstack(ls).tolist()
    
    return array_vec


# In[ ]:


india_df_sub['vec'] = india_df_sub.apply(lambda x: vec_column([x['has_online_delivery'],x['has_table_booking'],
                                                             x['cuisines_vec'],x['rating_vec'],x['city_vec'],
                                                             x['locality_vec'],x['avg_cost'],x['rating_color_vec'],
                                                             x['price_range_norm'],x['agg_rate_norm'],x['votes_norm']])
                                        ,axis =1)


# In[ ]:


master_ls = []
for i in india_df_sub['vec']:
    master_ls.append(i)
    
X_df = pd.DataFrame(master_ls).fillna(0.0)
X= X_df.as_matrix()


# In[ ]:


X_df.to_csv("cluster_train_data.csv")


# In[ ]:


X = pd.read_csv("cluster_train_data.csv").as_matrix()


# In[ ]:


from scipy.spatial.distance import cdist


# In[ ]:


distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
#     cluster_labels = kmeanModel.labels_
    
#     pca = PCA(n_components=2)
#     pca.fit(X)
#     cluster_vis_df = pd.DataFrame(pca.transform(X),columns = ['X','Y'])
#     cluster_vis_df['cluster'] = cluster_labels
    
#     facet = sns.lmplot(data=cluster_vis_df, x='X', y='Y', hue='cluster', 
#                    fit_reg=False, legend=True, legend_out=True)


# In[ ]:


df = pd.DataFrame([[i,distortions[i-1]] for i in K],columns =['num_clust','distortion'])
ax = sns.pointplot(x="num_clust", y="distortion",data=df)


# In[ ]:


kmeans_clust = KMeans(n_clusters=6).fit(X)
cluster_labels = kmeans_clust.labels_


# In[ ]:


india_df_sub['cluster'] = cluster_labels


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=3)
pca.fit(X)
cluster_vis_df = pd.DataFrame(pca.transform(X),columns = ['X','Y','Z'])
cluster_vis_df['cluster'] = cluster_labels


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cluster_vis_df['X'], cluster_vis_df['Y'], cluster_vis_df['Z'], c=cluster_labels, s=60)
ax.view_init(30, 185)
plt.show()


# In[ ]:





# ## Clutser Profiling

# In[ ]:


india_df_sub.columns


# In[ ]:


master_df_loc = master_df[['res_id','location']]
master_df_loc['lat'] = master_df_loc.apply(lambda x : float(x['location']['latitude']),axis=1)
master_df_loc['lang'] = master_df_loc.apply(lambda x : float(x['location']['longitude']),axis=1)


# In[ ]:


india_df_sub_loc = pd.merge(india_df_sub,master_df_loc[['res_id','lat','lang']].drop_duplicates(),on=['res_id'])


# In[ ]:


len(india_df_sub_loc)


# In[ ]:


facet = sns.lmplot(data=india_df_sub_loc, x='lat', y='lang', hue='cluster', 
                   fit_reg=False, legend=True, legend_out=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




