#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import json
# with open('../input/yelp_academic_dataset_business.json') as f:
#     d = json.load(f)
#https://stackoverflow.com/questions/21058935/python-json-loads-shows-valueerror-extra-data
checkins = []
for line in open('../input/yelp_academic_dataset_checkin.json', 'r'):
    checkins.append(json.loads(line))
#https://www.kaggle.com/xhlulu/convert-json-to-csv
df_checkins = pd.DataFrame(checkins)  
df_checkins.info()


# In[ ]:


df_checkins.head()


# In[ ]:


#https://stackoverflow.com/questions/21058935/python-json-loads-shows-valueerror-extra-data
yelps = []
for line in open('../input/yelp_academic_dataset_business.json', 'r'):
    yelps.append(json.loads(line))
yelps[0]
#https://www.kaggle.com/xhlulu/convert-json-to-csv
df = pd.DataFrame(yelps)
df.info()


# In[ ]:


df.head()


# In[ ]:


_ = df.city.value_counts()[:20].plot(kind='barh')


# In[ ]:


df_state_val_counts = df.state.value_counts()[:20].reset_index()
df_state_val_counts.columns = ['state', 'counts']


# In[ ]:


#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

data = [go.Bar(x=df_state_val_counts.index,
            y=df_state_val_counts.state)]
init_notebook_mode(connected=True)

#iplot(data, filename='jupyter-basic_bar')


fig = {
    'data': [
        go.Bar({
            'x': df_state_val_counts.state,
            'y': df_state_val_counts.counts,
            
        } )
    ],
    'layout': {
        'xaxis': {'title': 'State'},
        'yaxis': {'title': "Counts", 'type': 'log'}
    }
}

# IPython notebook
# py.iplot(fig, filename='pandas/grouped-scatter')

iplot(fig, filename='State wise Counts')


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


df.state.nunique()


# In[ ]:


_ = df.state.value_counts()[:20].iplot(kind='barh', logx=True)


# In[ ]:


_ = df.is_open.value_counts()


# In[ ]:


_ = df.postal_code.value_counts().iplot(kind='hist', logy=True)


# In[ ]:


_ = df.review_count.iplot(kind='hist',bins=80, logy=True)


# In[ ]:


_ = df.stars.value_counts().iplot(kind='bar',logy=True)


# In[ ]:


_ = df.review_count.iplot(kind='box',logy=True)


# In[ ]:


#_ = df.pivot_table(index=df.index, columns='stars',values='review_count').iplot(kind='box',logy=True)


# In[ ]:


df.pivot_table(index='state', columns='stars',values='review_count', aggfunc=np.sum)


# In[ ]:


_ = df.pivot_table(index='state', columns='stars',values='review_count', aggfunc=np.sum).iplot(kind='box',logy=True)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


_ = sns.jointplot(x="stars", y="review_count", data=df)


# In[ ]:


sns.relplot( x="state", y="review_count", data=df, aspect=3)


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="state", data=df)


# In[ ]:


#sns.relplot( x="state", y="review_count", data=df, aspect=3,hue='stars');


# In[ ]:


_ = sns.catplot(x="state", y="review_count", row="stars", data=df, aspect=3)


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(8,8)})
_ = df.stars.value_counts().plot(kind='barh')


# In[ ]:


df.attributes.isna().value_counts()


# In[ ]:


df['attributes'][:5]


# In[ ]:


#https://stackoverflow.com/questions/43934304/how-to-test-a-variable-is-null-in-python
#https://stackoverflow.com/questions/56222852/pandas-fillna-to-empty-dictionary
df['attributes'] = df['attributes'].apply(lambda x: {} if x is None else x)


# In[ ]:


df['attributes'][:5]


# In[ ]:


df_attributes = pd.io.json.json_normalize(df.attributes)


# In[ ]:


df_attributes.info()


# In[ ]:


df_attributes['RestaurantsPriceRange2'] = df_attributes['RestaurantsPriceRange2'].str.replace('None','0').astype(float)


# In[ ]:


_ = df_attributes.RestaurantsPriceRange2.value_counts().iplot(kind='bar',logy=True)


# In[ ]:


df_attributes.head()


# In[ ]:


df_attributes.WiFi.value_counts()


# In[ ]:


df_attributes.OutdoorSeating.value_counts()


# In[ ]:


df_attributes.RestaurantsPriceRange2.value_counts()


# In[ ]:


pd.crosstab(df.stars, df_attributes.RestaurantsPriceRange2, margins=True)


# In[ ]:


pd.crosstab(df.stars, df_attributes.RestaurantsPriceRange2, margins=True, normalize='index')


# In[ ]:


_=sns.heatmap(pd.crosstab(df.stars, df_attributes.RestaurantsPriceRange2,normalize='index'))


# In[ ]:


pd.crosstab(df_attributes.RestaurantsPriceRange2, df.stars,normalize='index', margins=True)


# In[ ]:


_=sns.heatmap(pd.crosstab(df_attributes.RestaurantsPriceRange2, df.stars,normalize='index'))


# In[ ]:


_ = pd.crosstab(df.stars, df_attributes.RestaurantsPriceRange2).plot()


# In[ ]:


_ = sns.jointplot(x=df_attributes.RestaurantsPriceRange2, y=df.review_count)


# In[ ]:


pd.crosstab(df.stars, df_attributes.GoodForKids=='True', margins=True)


# In[ ]:


#https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
from scipy import stats
stats.chi2_contingency(pd.crosstab(df.stars, df_attributes.GoodForKids=='True'))


# In[ ]:


pd.crosstab(df.state, df_attributes.GoodForKids=='True', margins=True)


# In[ ]:


_=pd.crosstab(df.state, df_attributes.GoodForKids=='True').iplot(kind='bar', logy=True)


# In[ ]:


#https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
from scipy import stats
stats.chi2_contingency(pd.crosstab(df.state, df_attributes.GoodForKids=='True'))


# In[ ]:


pd.crosstab(df_attributes.RestaurantsPriceRange2, df_attributes.GoodForKids=='True', margins=True)


# In[ ]:


pd.crosstab(df_attributes.RestaurantsPriceRange2, df_attributes.GoodForKids=='True', margins=True).iplot(kind='bar', logy=True)


# In[ ]:


#https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
from scipy import stats
stats.chi2_contingency(pd.crosstab(df_attributes.RestaurantsPriceRange2, df_attributes.GoodForKids=='True'))


# In[ ]:


pd.crosstab(df_attributes.BusinessAcceptsCreditCards=='True', df_attributes.GoodForKids=='True', margins=True)


# In[ ]:


#https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
from scipy import stats
stats.chi2_contingency(pd.crosstab(df_attributes.BusinessAcceptsCreditCards=='True', df_attributes.GoodForKids=='True'))


# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html

# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df_attributes.GoodForKids=='True', [df.stars, df_attributes.RestaurantsPriceRange2], margins=True).T


# In[ ]:


#https://towardsdatascience.com/running-chi-square-tests-in-python-with-die-roll-data-b9903817c51b
#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html
from scipy import stats
stats.chi2_contingency(pd.crosstab(df_attributes.GoodForKids=='True', [df.stars, df_attributes.RestaurantsPriceRange2]).T)


# In[ ]:


ax = pd.crosstab(df.stars, [df_attributes.WheelchairAccessible=='True', df_attributes.RestaurantsPriceRange2], 
            margins=True).iplot(kind='bar', logy=True)


# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df.stars, [df_attributes.WheelchairAccessible=='True', df_attributes.RestaurantsPriceRange2], margins=True)


# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df.stars, [df_attributes.WiFi.str.contains('free|paid', regex=True), df_attributes.RestaurantsPriceRange2], margins=True)


# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df.stars, [df_attributes.BikeParking=='True', df_attributes.RestaurantsPriceRange2], margins=True)


# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df.stars, [df_attributes.BusinessAcceptsCreditCards=='True', df_attributes.RestaurantsPriceRange2], margins=True)


# In[ ]:


#http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-19_17.html
pd.crosstab(df.stars, [df_attributes.BusinessParking.str.contains("'garage': True|'lot': True|'street': True", regex=True), 
                       df_attributes.RestaurantsPriceRange2], margins=True)


# In[ ]:


df_attributes.WiFi.str.contains('free|paid', regex=True).value_counts()


# In[ ]:


#df.loc[df_attributes.WiFi.str.contains('free|paid', regex=True)==True]


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df.loc[df_attributes.WiFi.str.contains('free|paid', regex=True)==True])


# In[ ]:


df_attributes.BikeParking.value_counts()


# In[ ]:


(df_attributes.BikeParking=='True').value_counts()


# In[ ]:


df_attributes.BusinessParking.value_counts()


# In[ ]:


df_attributes.BusinessParking.str.contains("'garage': True|'lot': True|'street': True", regex=True).value_counts()


# In[ ]:


df_attributes.WheelchairAccessible.value_counts()


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="state", data=df.loc[df_attributes.WheelchairAccessible=='True'])


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df)


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df.loc[df_attributes.WheelchairAccessible=='True'])


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df.loc[df_attributes.WheelchairAccessible!='True'])


# In[ ]:


sns.boxenplot(x=df.stars, y=df.review_count,
                    data=df.loc[df_attributes.WheelchairAccessible=='True'], palette="Set3")


# In[ ]:


df_attributes.GoodForKids.value_counts()


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="state", data=df.loc[df_attributes.GoodForKids=='True'])


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df.loc[df_attributes.GoodForKids=='True'])


# In[ ]:


df_attributes.BusinessAcceptsCreditCards.value_counts()


# In[ ]:


sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})
sns.countplot( x="stars", data=df.loc[df_attributes.BusinessAcceptsCreditCards=='True'])


# In[ ]:


df['categories'] = df['categories'].apply(lambda x: {} if x is None else x)


# In[ ]:


df_attributes.HasTV.value_counts()


# In[ ]:


df_attributes.OutdoorSeating.value_counts()


# In[ ]:


df_attributes.RestaurantsGoodForGroups.value_counts()


# In[ ]:


df_attributes.RestaurantsReservations.value_counts()


# In[ ]:


df_attributes.NoiseLevel.value_counts()


# In[ ]:


df_attributes.NoiseLevel.str.contains('quiet').value_counts()


# In[ ]:


df.hours.isna().value_counts()


# In[ ]:


df['hours'] = df['hours'].apply(lambda x: {} if x is None else x)


# In[ ]:


df_hours = pd.io.json.json_normalize(df.hours)


# In[ ]:


df_hours.info()


# In[ ]:


df_hours.describe()


# In[ ]:


df_hours.head()


# https://www.programcreek.com/python/example/100318/scipy.stats.chi2_contingency
# https://github.com/tompollard/tableone

# In[ ]:


get_ipython().system('pip install tableone')


# In[ ]:


from tableone import TableOne


# In[ ]:


data = pd.concat([df,df_attributes], axis=1)


# In[ ]:


data.info()


# In[ ]:


#features_table = TableOne(data, pval=False)


# In[ ]:


data['bool_WheelchairAccessible'] = data.WheelchairAccessible=='True'


# In[ ]:


data['bool_GoodForKids'] = data.GoodForKids=='True'


# In[ ]:


data['bool_RestaurantsGoodForGroups'] =  data.RestaurantsGoodForGroups=='True'


# In[ ]:


data['bool_BusinessParking'] = data.BusinessParking.str.contains("'garage': True|'lot': True|'street': True", regex=True)


# In[ ]:


data['bool_BikeParking'] =  data.BikeParking=='True'


# In[ ]:


data['bool_HasTV'] =  data.HasTV=='True'


# In[ ]:


data['bool_BusinessAcceptsCreditCards'] =  data.BusinessAcceptsCreditCards=='True'


# In[ ]:


data['bool_NoiseLevel_quiet'] =  data.NoiseLevel.str.contains('quiet')


# In[ ]:


columns= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV','bool_NoiseLevel_quiet']
features_table = TableOne(data, columns=columns, groupby = 'bool_GoodForKids', pval=True)


# In[ ]:


data[columns].info()


# In[ ]:


#https://seaborn.pydata.org/generated/seaborn.catplot.html
g = sns.catplot("bool_GoodForKids", col="bool_RestaurantsGoodForGroups",data=data[columns],
                row='bool_HasTV',
                kind="count", aspect=2)


# In[ ]:


df_stars_bool_GoodForKids_crosstab = pd.crosstab(data.stars,data.bool_GoodForKids,margins=True)


# In[ ]:


df_stars_bool_GoodForKids_crosstab


# In[ ]:


df_stars_bool_GoodForKids_crosstab.iloc[-1]


# In[ ]:


df_stars_bool_GoodForKids_crosstab/df_stars_bool_GoodForKids_crosstab.iloc[-1]


# In[ ]:


features_table


# In[ ]:


data['bool_OutdoorSeating'] =  data.OutdoorSeating=='True'


# In[ ]:


columns2= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 'bool_GoodForKids',
           'bool_BikeParking','bool_HasTV', 'bool_OutdoorSeating', 'bool_RestaurantsGoodForGroups',
          'bool_BusinessAcceptsCreditCards']


# In[ ]:


TableOne(data, columns=columns2, groupby = 'bool_WheelchairAccessible', pval=True)


# In[ ]:


from sklearn.cluster import MiniBatchKMeans
Nc = range(1, 5)
kmeans = [MiniBatchKMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(data[columns].fillna(99)).score(data[columns].fillna(99)) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[ ]:


kmeans_model = MiniBatchKMeans(n_clusters=4)
cluster_predict = kmeans_model.fit_predict(data[columns].fillna(99))


# In[ ]:


cluster_predict[:5]


# In[ ]:


data['cluster'] = cluster_predict


# In[ ]:


data['cluster'].value_counts()


# In[ ]:


columns3= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards', 'cluster']
TableOne(data, columns=columns3, groupby = 'cluster', pval=True)


# In[ ]:


columns3= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'cluster','bool_NoiseLevel_quiet' ,'bool_BusinessAcceptsCreditCards']
TableOne(data, columns=columns3, groupby = 'bool_GoodForKids', pval=True)


# In[ ]:


from sklearn.decomposition import NMF
model_NMF = NMF(n_components=4, init='random', random_state=0)
model_NMF_transform = model_NMF.fit_transform(data[columns3].fillna(99))


# In[ ]:


model_NMF_components = model_NMF.components_


# In[ ]:


model_NMF_components


# In[ ]:


pd.DataFrame(model_NMF_components,columns=columns3).T


# In[ ]:


model_NMF_transform[:5]


# In[ ]:


np.argmax(model_NMF_transform[0])


# In[ ]:


model_NMF_transform_arg_list = [np.argmax(row) for row in model_NMF_transform]


# In[ ]:


model_NMF_transform_arg_list[:5]


# In[ ]:


data['model_NMF_transform_arg'] = model_NMF_transform_arg_list


# In[ ]:


data['model_NMF_transform_arg'].value_counts()


# In[ ]:


columns4= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards', 'cluster','model_NMF_transform_arg']
TableOne(data, columns=columns4, groupby = 'model_NMF_transform_arg', pval=True)


# In[ ]:


TableOne(data, columns=columns4, groupby = 'bool_GoodForKids', pval=True)


# In[ ]:


from sklearn.preprocessing import Normalizer


# In[ ]:


model_NMF_transform_norm= Normalizer().fit_transform(model_NMF_transform)


# In[ ]:


model_NMF_transform_norm_arg_list = [np.argmax(row) for row in model_NMF_transform_norm]


# In[ ]:


model_NMF_transform_norm_arg_list[:5]


# In[ ]:


data['model_NMF_transform_norm_arg'] = model_NMF_transform_norm_arg_list


# In[ ]:


columns5= ['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards', 'cluster','model_NMF_transform_norm_arg']
TableOne(data, columns=columns4, groupby = 'model_NMF_transform_norm_arg', pval=True)


# In[ ]:


model_NMF_transform_norm.shape


# In[ ]:


data.loc[2,['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards','model_NMF_transform_norm_arg']]


# In[ ]:


model_NMF_transform_norm[2]


# In[ ]:


model_NMF_transform_norm.dot(model_NMF_transform_norm[2])


# In[ ]:


pd.Series(model_NMF_transform_norm.dot(model_NMF_transform_norm[2])).nlargest(10)


# In[ ]:


idx_similar = pd.Series(model_NMF_transform_norm.dot(model_NMF_transform_norm[2])).nlargest(10).index


# In[ ]:


idx_similar


# In[ ]:


data.loc[idx_similar,['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards','model_NMF_transform_norm_arg']]


# In[ ]:


data.loc[:20,['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards','model_NMF_transform_norm_arg']]


# In[ ]:


data.loc[5,['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards','model_NMF_transform_norm_arg']]


# In[ ]:


model_NMF_transform_norm.dot(model_NMF_transform_norm[5])


# In[ ]:


pd.Series(model_NMF_transform_norm.dot(model_NMF_transform_norm[5])).nlargest(10)


# In[ ]:


idx_similar = pd.Series(model_NMF_transform_norm.dot(model_NMF_transform_norm[5])).nlargest(10).index


# In[ ]:


data.loc[idx_similar,['stars','bool_WheelchairAccessible','RestaurantsPriceRange2', 'bool_BusinessParking', 
          'bool_GoodForKids','bool_RestaurantsGoodForGroups', 'bool_BikeParking', 'bool_HasTV',
           'bool_NoiseLevel_quiet', 'bool_BusinessAcceptsCreditCards','model_NMF_transform_norm_arg']]

