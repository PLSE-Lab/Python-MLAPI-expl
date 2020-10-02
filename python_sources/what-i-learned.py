#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.neighbors import NearestNeighbors,RadiusNeighborsClassifier
from sklearn import ensemble
from sklearn.metrics import classification_report
from scipy.spatial import KDTree,distance_matrix, cKDTree


# In[ ]:


x = pd.read_csv("../input/train_features.csv")
y = pd.read_csv("../input/train_labels.csv")
y = y.drop(["id"],axis=1)
test_features = pd.read_csv("../input/test_features.csv")


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


x.columns.values


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
Kn = KNeighborsClassifier(n_neighbors=20,weights='distance')
Kn.fit(x_train[["latitude","longitude"]],y_train)
Kn.predict(x_train[["latitude","longitude"]])
Kn.score(x_test[["latitude","longitude"]],y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
Kn = KNeighborsClassifier(n_neighbors=1,weights='distance')
Kn.fit(x_train[["latitude","longitude"]],y_train)
Kn.predict(x_train[["latitude","longitude"]])
Kn.score(x_test[["latitude","longitude"]],y_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
Kn = KNeighborsClassifier(n_neighbors=1)
Kn.fit(x_train[["latitude","longitude"]],y_train)
Kn.predict(x_train[["latitude","longitude"]])
Kn.score(x_test[["latitude","longitude"]],y_test)


# In[ ]:


y_test.values.ravel().shape


# In[ ]:


kr = RadiusNeighborsClassifier(radius=0.0075,outlier_label='functional')
kr.fit(x_train[["latitude","longitude"]],y_train)
kr.score(x_test[["latitude","longitude"]],y_test.values.ravel())


# In[ ]:


accuracy = []
for i in [0.001,0.01,0.1,0.5,.75,1,1.5,2]:
    kr= RadiusNeighborsClassifier(radius = i,outlier_label='functional')
    kr.fit(x_train[["latitude","longitude"]],y_train)
    accuracy.append(kr.score(x_test[["latitude","longitude"]],y_test))
accuracy


# Each degree of latitude is ~ 69 miles/111Km. 
# * So .1 is 6.9 miles.
# * 0.01 is .69 miles
# * 0.0145 is ~ 1 mile

# In[ ]:


accuracy = []
for i in [0.001,0.0145,0.03,0.5]:
    kr= RadiusNeighborsClassifier(radius = i,outlier_label='functional')
    kr.fit(x_train[["latitude","longitude"]],y_train)
    accuracy.append(kr.score(x_test[["latitude","longitude"]],y_test.values.ravel()))
accuracy


# # Getting an estimate from all the other wells within a 1 mile radius is a solid estimate.

# In[ ]:


j = Kn.predict_proba(x[["latitude","longitude"]])
kn_predictions =pd.DataFrame(j)


# In[ ]:


kn_predictions.isna().sum()


# In[ ]:


x[["k_0","k_1","k_2"]] = kn_predictions


# In[ ]:


z = x[["funder", "installer", "permit", "scheme_management","construction_year","payment","latitude","longitude",]]
z = z.replace([0,"unknown"],np.nan)
z = z.isna()
z["poorly_documented"] = False
for i in z.columns.values:
    z["poorly_documented"] = (z["poorly_documented"] | z[i])
x["poorly_documented"] = z["poorly_documented"]


# In[ ]:


x["location_approximation"] = x["basin"] + x["region"] + x["lga"]
x_work =pd.DataFrame(x[["location_approximation","longitude","latitude"]])
z = x_work.groupby(["location_approximation"]).agg({ 'longitude':np.mean,'latitude':np.mean})


# In[ ]:





# In[ ]:


# lmax = x["latitude"].max()
# for i in range(len(x["latitude"])):
#     if x["latitude"][i] == lmax:
#         x.latitude[i] = z.loc[ x["location_approximation"][i] ][1]
# lmin = x["longitude"].min()
# for i in range(len(x["longitude"])):
#     if x["longitude"][i] < 25:
#         x.longitude[i] = z.loc[ x["location_approximation"][i] ][0]


# In[ ]:





dar_es_salaam = (39.28333,-6.8)
mwanza = (32.9,-2.516667)
arusha = (36.683333,-3.366667)
dodoma = (35.741944,-6.173056)
mbeya = (33.45,-8.9)
morongoro = (37.66667,-6.816667)
tanga = (39.1,-5.0666667)
kahama = (32.6,-3.8375)
tabora = (32.8,-5.016667)
zanzibar = (39.199,-6.165)

x["dar_es_salaam"] = np.sqrt( (x["longitude"] - dar_es_salaam[0])**2 + ( x["latitude"]  - dar_es_salaam[1] )**2 )
x["mwanza"] = np.sqrt( (x["longitude"] - mwanza[0])**2 + ( x["latitude"]  - mwanza[1] )**2 )
x["arusha"] = np.sqrt( (x["longitude"] - arusha[0])**2 + ( x["latitude"]  - arusha[1] )**2 )
x["dodoma"] = np.sqrt( (x["longitude"] - dodoma[0])**2 + ( x["latitude"]  - dodoma[1] )**2 )
x["mbeya"] = np.sqrt( (x["longitude"] - mbeya[0])**2 + ( x["latitude"]  - mbeya[1] )**2 )
x["morongoro"] = np.sqrt( (x["longitude"] - morongoro[0])**2 + ( x["latitude"]  - morongoro[1] )**2 )
x["tanga"] = np.sqrt( (x["longitude"] - tanga[0])**2 + ( x["latitude"]  - tanga[1] )**2 )
x["kahama"] = np.sqrt( (x["longitude"] - kahama[0])**2 + ( x["latitude"]  - kahama[1] )**2 )
x["tabora"] = np.sqrt( (x["longitude"] - tabora[0])**2 + ( x["latitude"]  - tabora[1] )**2 )
x["zanzibar"] = np.sqrt( (x["longitude"] - zanzibar[0])**2 + ( x["latitude"]  - zanzibar[1] )**2 )
x["distance_to_nearest_city"] = x.loc[: , ["dar_es_salaam","mwanza","arusha","dodoma","mbeya","morongoro","tanga","kahama","tabora","zanzibar"]].min(axis=1)


# In[ ]:


z = x[(x["construction_year"] >1000)].median()
# x["construction_year"] = x["construction_year"].replace(0,z)
# x["construction_year"]
z["construction_year"]


# In[ ]:


x["construction_year"] = x["construction_year"].replace(0,2000)


# In[ ]:


sns.distplot(x["construction_year"] )


# In[ ]:


top_installers =  x["installer"].value_counts().head(100).index
top_installers = list(top_installers)
top_installers.remove('0')
top_installers
x["top_installer"] = x["installer"].apply(lambda x : x in top_installers)


# In[ ]:


top_funder =  x["funder"].value_counts().head(100).index
top_funder = list(top_funder)
top_funder.remove('0')
top_funder
x["top_funder"] = x["funder"].apply(lambda x : x in top_funder)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25)


# In[ ]:


x


# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


x.head()


# In[ ]:


len(x_train["management_group"].unique())


# In[ ]:


a = x_train[["longitude","latitude"]]
a = a.replace(0,np.nan)
a = a.dropna()
plt.scatter(a["latitude"],a["longitude"])


# In[ ]:


a = x_train[["longitude","latitude"]]
a = a.replace(0,np.nan)
a = a.dropna()

pca = PCA(2)
p = pca.fit_transform(RobustScaler().fit_transform(a[["longitude","latitude"]]))
plt.scatter(p.T[0],p.T[1])


# In[ ]:





# In[ ]:





# In[ ]:


cols=["funder","installer","wpt_name","basin","subvillage","region","lga","ward","public_meeting","scheme_management"     ,"scheme_name","extraction_type","extraction_type_group","extraction_type_class","management","payment", "payment_type",     "water_quality","quality_group","quantity","quantity_group","source","source_type","source_class","waterpoint_type","waterpoint_type_group"]


# In[ ]:


len(x_train["recorded_by"].unique())


# In[ ]:


sns.distplot(x_train["num_private"])


# In[ ]:





# In[ ]:





# In[ ]:


a = x_train["population"]
a =a.replace(0,a.mean())
a = np.log(a)
sns.distplot(a)


# In[ ]:


pipeline = make_pipeline(
LogisticRegression()
)


# # Best model

# In[ ]:


c = ["basin","region","region_code","district_code","scheme_management","extraction_type","management",         "extraction_type_group","quantity_group", "payment", "waterpoint_type","water_quality","quality_group","public_meeting"        ,"lga","poorly_documented","source","source_class","0","1","2"]


# In[ ]:


class_weights = {'functional':1,'non functional':1.2,'functional needs repair':1}


# In[ ]:


c = ["basin","region","region_code","district_code","scheme_management","extraction_type","management",         "extraction_type_group","quantity_group", "payment", "waterpoint_type","water_quality","quality_group","public_meeting"        ,"lga","poorly_documented","source","source_class","extraction_type_class", "top_installer","management_group",         "top_funder","distance_to_nearest_city", 'gps_height',"population","construction_year"]


# In[ ]:


cols1 = ["basin","region","region_code","district_code","scheme_management","extraction_type","management",         "extraction_type_group","quantity_group", "payment", "waterpoint_type","water_quality","quality_group","public_meeting"        ,"lga","poorly_documented","source","source_class","extraction_type_class", "top_installer","management_group",         "top_funder"]
pipeline = make_pipeline(
ce.OneHotEncoder(cols = cols1,use_cat_names=True),
    RobustScaler(),
    PCA(349),
LogisticRegression(solver='lbfgs',class_weight=class_weights)
)
pipeline.fit(x_train[c],y_train)
y_pred = pipeline.predict(x_test[c])
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:


# cols1 = ["basin","region","region_code","district_code","scheme_management","extraction_type","management",\
#          "extraction_type_group","quantity_group", "payment", "waterpoint_type","water_quality","quality_group","public_meeting"\
#         ,"lga","poorly_documented","source","source_class","extraction_type_class", "top_installer","management_group",\
#          "top_funder"]
# pipeline = make_pipeline(
# ce.OneHotEncoder(cols = cols1, use_cat_names=True),
#     RobustScaler(),
#     PCA(349),
# ensemble.GradientBoostingClassifier(subsample=0.9,n_estimators=200,max_depth=4)
# )
# pipeline.fit(x_train[c],y_train)
# y_pred = pipeline.predict(x_test[c])
# accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


# from sklearn.metrics import confusion_matrix
# confusion_matrix(y_test,y_pred)
# def confusion_viz(y_true, y_pred):
#     matrix = confusion_matrix(y_true,y_pred)
#     return sns.heatmap(matrix, annot=True,
#                       fmt=',', linewidths=1,linecolor='grey',
#                       square=True,
#                       xticklabels=['Predicted\nFunctional', 'Predicted\nNeeds Repair', "Predicted\nNon Functional"], 
#                        yticklabels=['Actual\nFunctional', 'Actual\nNeeds Repair', "Actual\nNon Functional"])
# confusion_viz(y_test, y_pred)


# In[ ]:


from xgboost import XGBClassifier

pipeline = make_pipeline(
ce.OneHotEncoder(cols = cols1, use_cat_names=True),
    RobustScaler(),
    PCA(349),
    XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 
                          num_class = 3, maximize = False, eval_metric = 'merror', eta = .1,
                          max_depth = 14, colsample_bytree = .4)
)
pipeline.fit(x_train[c],y_train)
y_pred = pipeline.predict(x_test[c])
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:


z = test_features[["funder", "installer", "permit", "scheme_management","construction_year","payment","latitude","longitude",]]
z = z.replace([0,"unknown"],np.nan)
z = z.isna()
z["poorly_documented"] = False
for i in z.columns.values:
    z["poorly_documented"] = (z["poorly_documented"] | z[i])
test_features["poorly_documented"] = z["poorly_documented"]


# In[ ]:


dar_es_salaam = (39.28333,-6.8)
mwanza = (32.9,-2.516667)
arusha = (36.683333,-3.366667)
dodoma = (35.741944,-6.173056)
mbeya = (33.45,-8.9)
morongoro = (37.66667,-6.816667)
tanga = (39.1,-5.0666667)
kahama = (32.6,-3.8375)
tabora = (32.8,-5.016667)
zanzibar = (39.199,-6.165)

test_features["dar_es_salaam"] = np.sqrt( (test_features["longitude"] - dar_es_salaam[0])**2 + ( test_features["latitude"]  - dar_es_salaam[1] )**2 )
test_features["mwanza"] = np.sqrt( (test_features["longitude"] - mwanza[0])**2 + ( test_features["latitude"]  - mwanza[1] )**2 )
test_features["arusha"] = np.sqrt( (test_features["longitude"] - arusha[0])**2 + ( test_features["latitude"]  - arusha[1] )**2 )
test_features["dodoma"] = np.sqrt( (test_features["longitude"] - dodoma[0])**2 + ( test_features["latitude"]  - dodoma[1] )**2 )
test_features["mbeya"] = np.sqrt( (test_features["longitude"] - mbeya[0])**2 + ( test_features["latitude"]  - mbeya[1] )**2 )
test_features["morongoro"] = np.sqrt( (test_features["longitude"] - morongoro[0])**2 + ( test_features["latitude"]  - morongoro[1] )**2 )
test_features["tanga"] = np.sqrt( (test_features["longitude"] - tanga[0])**2 + ( test_features["latitude"]  - tanga[1] )**2 )
test_features["kahama"] = np.sqrt( (test_features["longitude"] - kahama[0])**2 + ( test_features["latitude"]  - kahama[1] )**2 )
test_features["tabora"] = np.sqrt( (test_features["longitude"] - tabora[0])**2 + ( test_features["latitude"]  - tabora[1] )**2 )
test_features["zanzibar"] = np.sqrt( (test_features["longitude"] - zanzibar[0])**2 + ( test_features["latitude"]  - zanzibar[1] )**2 )

test_features["distance_to_nearest_city"] = test_features.loc[: , ["dar_es_salaam","mwanza","arusha","dodoma","mbeya","morongoro","tanga","kahama","tabora","zanzibar"]].min(axis=1)


# In[ ]:


test_features
test_features["construction_year"] = test_features["construction_year"].replace(0,2000)


# In[ ]:


test_features["top_installer"] = test_features["installer"].apply(lambda x : x in top_installers)
test_features["top_funder"] = test_features["funder"].apply(lambda x : x in top_funder)

y_test_pred = pipeline.predict(test_features[c])
y_test_pred
df = pd.DataFrame(y_test_pred,columns=["status_group"])
df = pd.concat([test_features["id"],df],axis=1)
df = df.set_index("id")
df.to_csv("y_test_pred.csv")


# In[ ]:


x_train


# In[ ]:


accuracy_score(y_test_pred)


# In[ ]:





# In[ ]:




