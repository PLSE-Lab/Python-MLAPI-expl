#!/usr/bin/env python
# coding: utf-8

# ## First Part: Exploratory Data Analysis.

# In[ ]:


from datetime import datetime

print("last update: {}".format(datetime.now())) 


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebrafor dirname, _, filenames in os.walk('/kaggle/input'):
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Some statistics about columns

# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


# Read the data
X_original = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
X = X_original.copy()


# In[ ]:


print('Train_shape:', X_original.shape, 'test_shape:', X_test_full.shape)


# In[ ]:


X.describe().iloc[:,15:]


# In[ ]:


#X.dtypes All variables are in type int64


# In[ ]:


# Target variable is Cover_Type
set(X.Cover_Type)


# ### Statistics on categoricals values

# #### 1. Soil Type 

# In[ ]:


# There are 30 types of soil, named Soil_type1, Soil_type2 so on unitil Soil_type30
soil_fields = ['Soil_Type'+ str(i) for i in range(1,41)]
gf = pd.DataFrame()
for feature in soil_fields:
    gf = pd.concat([gf,X[feature].value_counts()],axis = 1)


# In[ ]:


gf


# In[ ]:


gf.loc[:, gf.isna().any()]


# It appear that soil of type 7 and 15 just have unique value (0).
# Let's see if this kind of soil type appear in the given test set

# In[ ]:


X_test_full.head()


# In[ ]:


gf2 = pd.DataFrame()
for feature in ['Soil_Type'+ str(i) for i in [7,15]]:
    gf2 = pd.concat([gf2,X_test_full[feature].value_counts()], axis=1)
    
gf2.index = ['not_present', 'present']
gf2


# It appear That the soils of type 7 and 15 are rares. 105/565892 for type 7 and 3/565892 for type 15. They do not appear in the given training set. 
# We are going to clearly find how to deal with these variables.
# Analysis is capital because it is important that our model generalize well on unknown data.

# In[ ]:


#Create single soil field (reverse the one hot encoding)
soil_fields = ['Soil_Type'+ str(i) for i in range(1,41)]
train_soil = X[soil_fields]
X['Soil_Type'] = train_soil.idxmax(axis = 1).astype(str)
X.head()


# In[ ]:


#Visiluazing what we said about soil of type 7 and 15
ax = X['Soil_Type'].value_counts().plot(kind = 'barh', figsize = (10, 12))
# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.3, i.get_y()+.38,             str(round((i.get_width()/total)*100, 2))+'%', fontsize=15,
color='dimgrey')


# No comment Soil of type 7 and 15 do not appear!

# #### 2. Wilderness area field

# In[ ]:


#Create single wilderness area field (reverse the one hot encoding)
Wilderness_Area_Fields = ['Wilderness_Area'+ str(i) for i in range(1,5)]

train_wilderness = X[Wilderness_Area_Fields]
X['Wilderness_Area'] = train_wilderness.idxmax(axis = 1)
X.head()


# In[ ]:


# There are 4 types of Wilderness area field, named Wilderness_Area1 , 2, 3  and 4
Wilderness_Areas = ['Wilderness_Area'+ str(i) for i in range(1,5)]
wf = pd.DataFrame()
for feature in Wilderness_Areas:
    wf = pd.concat([wf,X[feature].value_counts()],axis = 1)

wf.index = ['not present', 'present']
wf


# In[ ]:


# I don't know if this graph is useful, but let's get a look at!
wf.plot(kind = 'bar')


# In[ ]:


eda = X.copy()
eda.head()


# In[ ]:


eda.drop(eda.columns[10:54], axis = 1, inplace = True)


# In[ ]:


eda.head()


# In[ ]:


cover_type_index_to_name = {1: 'Spruce/Fir', 2: 'Lodgepole Pine', 3 : 'Ponderosa Pine', 4: 'Cottonwood/Willow',5: 'Aspen',6: 'Douglas-fir',7: 'Krummholz'}


# In[ ]:


cover_type_index_to_name[2]


# In[ ]:


eda['Cover_Type_Name'] = eda['Cover_Type'].map(cover_type_index_to_name)


# In[ ]:


eda.head()


# In[ ]:


eda.drop(['Cover_Type'], axis = 1, inplace = True)


# In[ ]:


eda.head()


# ### It's time now to perform data analysis:
# 
# 1. Firstly we will start studying the relationship between the target variable Cover_Type_Name and our two categoricals variable (i.e see if variables are indenpedent)
# 
# 2. secondly We will do multivariate analysis with numerical variables (PCA, TSNE, color by target variable). This can helps us fit our classification
# 
# 
# 

# #### Khi 2 test

# In[ ]:


import scipy.stats as st


# In[ ]:


contengency = pd.crosstab(eda.Soil_Type, eda.Cover_Type_Name, margins=True, margins_name='Total')
contengency


# In[ ]:


st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(contengency)


# Display chi2 and the p_value of test in 

# In[ ]:


print('Khi2_calculate:', st_chi2, 'P value of test:', st_p, 'degree of freedom:', st_dof)


# We get all (p_value<0.01) to rejet the hypothesis of independance between cover_type and soil type, so that this two variable are correlated.

# In[ ]:


heatData1 = (contengency.values - st_exp)**2/st_exp
heatData1 = pd.DataFrame(heatData1, columns=contengency.columns, index = contengency.index)
plt.figure(figsize=(20, 20))
sns.heatmap(heatData1, cmap="YlGnBu", annot=True)


# In[ ]:


#barplot of soil type
contengency.iloc[0:-1,0:-1].T.plot(kind = 'barh', figsize=(17,15))
plt.grid()
plt.show()


# In[ ]:


# expected_Table in case of independance
expected_Table = pd.DataFrame(st_exp, columns=contengency.columns, index = contengency.index)
expected_Table


# In[ ]:


contengency2 = pd.crosstab(eda.Wilderness_Area, eda.Cover_Type_Name, margins=True, margins_name='Total')
contengency2


# In[ ]:


st_chi22, st_p2, st_dof2, st_exp2 = st.chi2_contingency(contengency2)


# In[ ]:


heatData = (contengency2.values - st_exp2)**2/st_exp2


# In[ ]:


heatData = pd.DataFrame(heatData, columns=contengency2.columns, index = contengency2.index)
heatData


# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(heatData, cmap="YlGnBu", annot=True)


# In[ ]:


print('Khi2_calculate:', st_chi22, 'P value of test:', st_p2, 'degree of freedom:', st_dof2)


# In[ ]:



contengency2.iloc[0:-1,0:-1].T.plot(kind = 'barh', figsize = (10,12))
plt.grid()


# cottonwood/Willow forest type have his area wilderness just in type 4. Area wildeness of type 2 is too scarce, and just belongs in forest of spruce-fr, Krummholz and Lodgepol pine

# A simple decision trees with just those two variable could be good to predict the type of forest, because of their dependances with cover type. It 's just an intuition!
# 
# Only looking for example the previous graph we could therefore say that:
# 
# - if forest have only his **wilderness_area of type 4** we could predict that the forest type is **cottonwood/Willow**
# 
# - We could predict other type of forest using both wilderness_area type and the soil type
# 
# 
# #### An Experiment to confirm our intution about the tree decision with it ability to good predict cottonwood/willow forest
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

Experiments = {"Algo":["Decicion Tree", "knnclassifier", "LogisticRegression", "svc_classif", 'XGBClassifier'],
              "object": [lambda: DecisionTreeClassifier(),
                        lambda: KNeighborsClassifier(),
                        lambda: LogisticRegression(random_state=0,solver ='newton-cg', multi_class='multinomial'),
                        lambda: SVC(gamma='scale', kernel='rbf', probability=True),
                        lambda: XGBClassifier( learning_rate =0.1, n_estimators=1000)],
              "prediction": [[] for _ in range(5)]}

X_copy = X_original.copy()
scale = StandardScaler()
X_copy.iloc[:,0:10] = scale.fit_transform(X_copy.iloc[:,0:10])
[_.shape for _ in train_test_split(X_copy.drop('Cover_Type', axis = 1), X['Cover_Type'], test_size = 0.25)]


# In[ ]:


actuals = []
for _ in range(3):
    X_train, X_valid, y_train, y_valid = train_test_split(X_copy.drop('Cover_Type', axis = 1), X['Cover_Type'], test_size = 0.25)
    for i, obj in enumerate(Experiments["object"]):
        if i == 0:
            model = obj()
            model.fit(X_train.iloc[:,10:], y_train)
            Experiments["prediction"][i] +=list(model.predict(X_valid.iloc[:,10:]))
        else:
            model = obj()
            model.fit(X_train, y_train)
            Experiments["prediction"][i] +=list(model.predict(X_valid))

    actuals += list(y_valid)
actuals = pd.Series(actuals)
Experiments["prediction"] = list(map(pd.Series, Experiments["prediction"]))


# In[ ]:





# In[ ]:


agg_model = pd.DataFrame()
agg_model_col = []
for i, algo in enumerate(Experiments["Algo"]):
    agg_model = pd.concat([agg_model, Experiments["prediction"][i]], axis = 1)
    agg_model_col.append(algo)

agg_model = pd.concat([agg_model, actuals], axis = 1)
agg_model_col.append("Real_Prediction")
agg_model.columns = agg_model_col
agg_model.head(10)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
for i, algo in enumerate(Experiments["Algo"]):
    print("Confusion Matrix for {}\n".format(algo))
    cm = confusion_matrix(agg_model.iloc[:,-1], agg_model.iloc[:,i])
    df_cm = pd.DataFrame(data = cm, columns=np.unique(agg_model.iloc[:,-1]), index = np.unique(agg_model.iloc[:,-1]))
    print("Acc_score: {}\n".format(accuracy_score(agg_model.iloc[:,-1], agg_model.iloc[:,i])))
    print(str(df_cm),"\n\n")


# Our intuition is not so well.
# However Our Experiment show that KNeighborsClassifier and svc can be good! We will combine its in our final model, using **VotingClassifier** from sklearn.ensemble
# 

# #### Multivariate analysis

# In[ ]:


from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import decomposition
import random


# In[ ]:


random.seed(0)
rge=random.sample(range(15120),  2000)


# In[ ]:


eda_matrix = eda.iloc[rge,:10].values


# In[ ]:


std_scale = preprocessing.StandardScaler().fit(eda_matrix)
eda_matrix_scaled = std_scale.transform(eda_matrix)


# In[ ]:


pca = decomposition.PCA(n_components=7)
pca.fit(eda_matrix_scaled )


# In[ ]:


print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.sum())


# In[ ]:


X_projected = pca.transform(eda_matrix_scaled)
pca_data = np.vstack((X_projected[:, 0], X_projected[:, 1], eda.Cover_Type_Name[rge])).T
pca_data = pd.DataFrame(data = pca_data)
pca_data.columns = ['dim1 28,55%', 'dim2 22,66%', 'name']
pca_data.head(10)


# In[ ]:


type(X_projected[:, 0])


# In[ ]:


plt.figure(figsize = (10, 10))
sns.scatterplot(x="dim1 28,55%", y="dim2 22,66%",
              hue="name",
              data=pca_data)


# #### The explained Variance is low, abandon with pca

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


eda_tsne = TSNE(n_components=3, random_state=0)


# In[ ]:


tsne_data = eda_tsne.fit_transform(X_projected)


# In[ ]:


tsne_data


# In[ ]:


tsne_df = pd.DataFrame(data = tsne_data, columns = ['dim1', 'dim2', 'dim3'])
tsne_df.shape


# In[ ]:


len(eda.Cover_Type_Name[rge].values)


# In[ ]:


tsne_df['name'] = list(eda.Cover_Type_Name[rge])
tsne_df.head(10)


# In[ ]:


plt.figure(figsize = (10, 10))
sns.scatterplot(x="dim1", y="dim2",
              hue="name",
              data=tsne_df)


# Not easy to visualize differents groups in 2 dimensions

# In[ ]:


### Visualising Tsne in 3 dimensiosn
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10,10))
ax = Axes3D(fig)
ax.scatter(tsne_df.iloc[:,0], tsne_df.iloc[:,1], tsne_df.iloc[:,2], cmap='tab10')
plt.show()


# ## See the distributions of numericals columns 
