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


# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import catboost
import gensim
from shapely.geometry import  Point
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from matplotlib import cm
import urllib.request
import shutil
import zipfile
import os
import re
import contextily as ctx
import geoplot as gplt
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from pdpbox import pdp, get_dataset, info_plots
import shap


# In[ ]:


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip',parse_dates=['Dates'], date_parser=dateparse)
test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip',parse_dates=['Dates'], date_parser=dateparse)
############################
print('done')


# In[ ]:


print(train.duplicated().sum())
train.drop_duplicates(inplace=True)
assert train.duplicated().sum() ==0
############################
print('done')


# In[ ]:


print(train.columns.difference(test.columns))
# we need to predict the category of the crime. 
# find a way to use columns 'Descript' and 'Resolution' 


# In[ ]:


train.dtypes


# In[ ]:


#After cleaning the dataset from outliers and duplicates, we examine the variables.
# visualizing longitude and latitude point on the world map
# special thanks to  :- https://www.kaggle.com/yannisp/sf-crime-analysis-prediction/output
def create_gdf(df):
    gdf = df.copy()
    gdf['Coordinates'] = list(zip(gdf.X, gdf.Y))
    gdf.Coordinates = gdf.Coordinates.apply(Point)
    gdf = gpd.GeoDataFrame(
        gdf, geometry='Coordinates', crs={'init': 'epsg:4326'})
    return gdf

train_gdf = create_gdf(train)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(color='white', edgecolor='black')
train_gdf.plot(ax=ax, color='red')
plt.show()

# mapping X and Y shows that few points have erroronous longitude and latitude value and need to be corrected
# by suitable technique


# In[ ]:


print(train_gdf.loc[train_gdf.Y > 50].count()[0])
#train_gdf.loc[train_gdf.Y > 50].sample(5)
# all mislabelled X and Y values have X = -120.5 and Y =90.0


# In[ ]:


# We will replace the outlying coordinates with the average coordinates of the district they belong.
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='most_frequent')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])
    test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test.loc[test['PdDistrict'] == district, ['X', 'Y']])

train_gdf = create_gdf(train)

############################
print('done')


# In[ ]:


# Plot of incidence count in the day 
#Dates & Day of the week
#These variables are distributed uniformly between 1/1/2003 to 5/13/2015 (and Monday to Sunday) and split between the training and the testing dataset as mentioned before. We did not notice any anomalies on these variables.
#The median frequency of incidents is 389 per day with a standard deviation of 48.51.

col = sns.color_palette()

train['Date'] = train.Dates.dt.date
train['Hour'] = train.Dates.dt.hour

plt.figure(figsize=(10, 6))
data = train.groupby('Date').count().iloc[:, 0]
sns.kdeplot(data=data, shade=True)
plt.axvline(x=data.median(), ymax=0.95, linestyle='--', color=col[1])
plt.annotate(
    'Median: ' + str(data.median()),
    xy=(data.median(), 0.004),
    xytext=(200, 0.005),
    arrowprops=dict(arrowstyle='->', color=col[1], shrinkB=10))
plt.title(
    'Distribution of number of incidents per day', fontdict={'fontsize': 16})
plt.xlabel('Incidents')
plt.ylabel('Density')
plt.legend().remove()
plt.show()
############################
print('done')


# In[ ]:


# number of incidents by weekdays
# special thanks to :- https://www.kaggle.com/yannisp/sf-crime-analysis-prediction/output

data = train.groupby('DayOfWeek').count().iloc[:, 0]

data = data.reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
    'Sunday'
])

plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        data.index, (data.values / data.values.sum()) * 100,
        orient='v',
        palette=cm.ScalarMappable(cmap='Reds').to_rgba(data.values))

plt.title('Incidents per Weekday', fontdict={'fontsize': 16})
plt.xlabel('Weekday')
plt.ylabel('Incidents (%)')

############################
print('done')


# In[ ]:


# 'Incidents per Crime Category'
# special thanks to :- https://www.kaggle.com/yannisp/sf-crime-analysis-prediction/output

data = train.groupby('Category').count().iloc[:, 0].sort_values(
    ascending=False)
data = data.reindex(np.append(np.delete(data.index, 1), 'OTHER OFFENSES'))

plt.figure(figsize=(10, 10))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        (data.values / data.values.sum()) * 100,
        data.index,
        orient='h',
        palette="Reds_r")

plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})
plt.xlabel('Incidents (%)')

plt.show()

############################
print('done')


# In[ ]:


# 'Average number of incidents per hour'
# special thanks to :- https://www.kaggle.com/yannisp/sf-crime-analysis-prediction/output

data = train.groupby(['Hour', 'Date', 'Category'],
                     as_index=False).count().iloc[:, :4]
data.rename(columns={'Dates': 'Incidents'}, inplace=True)
data = data.groupby(['Hour', 'Category'], as_index=False).mean()
data = data.loc[data['Category'].isin(
    ['ROBBERY', 'GAMBLING', 'BURGLARY', 'ARSON', 'PROSTITUTION'])]

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 4))
ax = sns.lineplot(x='Hour', y='Incidents', data=data, hue='Category')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6)
plt.suptitle('Average number of incidents per hour')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

############################
print('done')


# In[ ]:


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

train = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip',parse_dates=['Dates'], date_parser=dateparse)
test = pd.read_csv('/kaggle/input/sf-crime/test.csv.zip',parse_dates=['Dates'], date_parser=dateparse)
############################
print('done')


# In[ ]:


# We will replace the outlying coordinates with the average coordinates of the district they belong.
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='most_frequent')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])
    test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test.loc[test['PdDistrict'] == district, ['X', 'Y']])
############################
print('done')
    


# In[ ]:


y_train = train['Category']
train_des = train['Descript']
train_res = train['Resolution']
train.drop(["Category", "Descript", "Resolution"], axis=1, inplace=True)

############################
print('done')
    


# In[ ]:


test_ID = test["Id"]
test.drop("Id", axis=1, inplace=True)

############################
print('done')


# In[ ]:


le = LabelEncoder()
y_train = le.fit_transform(y_train)
print(le.classes_)

############################
print('done')


# In[ ]:


#train.drop(['Address','clean_text'],axis=1,inplace = True)
#test.drop(['Address','clean_text'],axis=1,inplace = True)


# In[ ]:


num_train = train.shape[0]
all_data = pd.concat((train, test), ignore_index=True)

############################
print('done')


# In[ ]:


# related to data and address

def feature_engineering(data):
    date = pd.to_datetime(all_data['Dates'])
    data['year'] = date.dt.year
    data['month'] = date.dt.month
    data['day'] = date.dt.day
    data['hour'] = date.dt.hour
    data['minute'] = date.dt.minute
    data['special_time'] = data['minute'].isin([0, 30]).astype(int)
  # all_data['second'] = date.dt.second  # all zero
    data["n_days"] = (date - date.min()).apply(lambda x: x.days)
    data.drop("Dates", axis=1, inplace=True)
    data['block'] = data["Address"].str.contains("block", case=False)
    data['ST'] = data['Address'].str.contains('ST', case=False)
    #data.drop(['Address','clean_text'],axis=1,inplace=True)
    return data

all_data1 = feature_engineering(all_data)

############################
print('done')


# In[ ]:


# related to "X" and "Y"

def feature_engineering2(data):
    data["X+Y"] = data["X"] + data["Y"]
    data["X-Y"] = data["X"] - data["Y"]
    data["XY30_1"] = data["X"] * np.cos(np.pi / 6) + data["Y"] * np.sin(np.pi / 6)
    data["XY30_2"] = data["Y"] * np.cos(np.pi / 6) - data["X"] * np.sin(np.pi / 6)
    data["XY60_1"] = data["X"] * np.cos(np.pi / 3) + data["Y"] * np.sin(np.pi / 3)
    data["XY60_2"] = data["Y"] * np.cos(np.pi / 3) - data["X"] * np.sin(np.pi / 3)
    data["XY1"] = (data["X"] - data["X"].min()) ** 2 + (data["Y"] - data["Y"].min()) ** 2
    data["XY2"] = (data["X"].max() - data["X"]) ** 2 + (data["Y"] - data["Y"].min()) ** 2
    data["XY3"] = (data["X"] - data["X"].min()) ** 2 + (data["Y"].max() - data["Y"]) ** 2
    data["XY4"] = (data["X"].max() - data["X"]) ** 2 + (data["Y"].max() - data["Y"]) ** 2
    #data["XY5"] = (data["X"] - X_median) ** 2 + (data["Y"] - Y_median) ** 2
    pca = PCA(n_components=2).fit(data[["X", "Y"]])
    XYt = pca.transform(data[["X", "Y"]])
    data["XYpca1"] = XYt[:, 0]
    data["XYpca2"] = XYt[:, 1]
    #return data
    # n_components selected by aic/bic
    clf = GaussianMixture(n_components=150, covariance_type="diag",random_state=0).fit(data[["X", "Y"]])
    data["XYcluster"] = clf.predict(data[["X", "Y"]])
    return data

all_data2 = feature_engineering2(all_data1)

############################
print('done')


# In[ ]:


# cleaning "text" column of train and test dataset and saving it to column 'clean_txt'

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=True, preprocessor=None,tokenizer=lambda x : x.split(), analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 3), max_df=0.75, min_df=5, max_features=12500)

train_vect = vectorizer.fit_transform(train['Address'])
test_vect = vectorizer.transform(test['Address'])

##################
print('done')

#print(vectorizer.get_feature_names())


# In[ ]:


sentences = []
for s in all_data2["Address"]:
    sentences.append(s.split(" "))
address_model = gensim.models.Word2Vec(sentences, min_count=1)
encoded_address = np.zeros((all_data2.shape[0], 100))
for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        encoded_address[i] += address_model.wv[sentences[i][j]]
    encoded_address[i] /= len(sentences[i])

print('done')


# In[ ]:


all_data2.drop(['Address'],axis=1,inplace=True)

categorical_features = ["DayOfWeek", "PdDistrict", "block", "special_time", "XYcluster",'ST']
ct = ColumnTransformer(transformers=[("categorical_features", OrdinalEncoder(), categorical_features)],
                       remainder="passthrough")
all_data3 = ct.fit_transform(all_data2)

############################
print('done')


# In[ ]:


#train_data =sp.sparse.hstack((X_train, train_vec))
#test_data = np.hstack((X_test, test_vec))
#train_data = np.hstack(vectorizer.fit_transform(train.clean_text),train2)
#test_data = sp.sparse.hstack((vectorizer.transform(test.clean_text),test2),format='csr')
all_data4 = np.hstack((all_data3,encoded_address))
############################
print('done')


# In[ ]:


train2 = all_data3[:num_train]
test2 = all_data3[num_train:]

############################
print('done')


# In[ ]:


from scipy import sparse
#sA = sparse.csr_matrix(train2)
train_2 = sparse.hstack((train2.astype(float),train_vect))
test_2 = sparse.hstack((test2.astype(float),test_vect))

#test_2 = pd.concat((test2,test_vect))

############################
print('done')


# In[ ]:


from lightgbm import LGBMClassifier
model = LGBMClassifier(objective="multiclass", num_class=39, max_bin = 465, max_delta_step = 0.9,
                      learning_rate=0.4, num_leaves = 42, n_estimators=100, verbose=50)
model.fit(train2, y_train)
preds = model.predict_proba(test2)
submission = pd.DataFrame(preds, columns=le.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)
submission.to_csv('LGBM_final1.csv', index_label='Id')
############################
print('done')


# In[ ]:


from lightgbm import LGBMClassifier
model2 = LGBMClassifier(objective="multiclass", num_class=39, max_bin = 465, max_delta_step = 0.9,
                      learning_rate=0.4, num_leaves = 42, n_estimators=100, verbose=50)
model2.fit(train_2, y_train)
preds2 = model2.predict_proba(test_2)
submission2 = pd.DataFrame(preds2, columns=le.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)
submission2.to_csv('LGBM_final2.csv', index_label='Id')
############################
print('done')


# In[ ]:


sub_tot = (preds+preds2+prob)/3
submission4 = pd.DataFrame(sub_tot, columns=le.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)
submission4.to_csv('final.csv', index_label='Id')
############################
print('done')

