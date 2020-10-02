#!/usr/bin/env python
# coding: utf-8

# # New York Property Data Fraud Analysis
# [Yulia Niu](https://xyniu0829.wixsite.com/yulia) - February 2019

# # Summary
# I built unsupervised fraud models on the NY Property Data in order to identify fraudulent events. 
# 
# I first took the time to understand the data and the business problem, which was determining which data records are fraudulent in the NY property dataset. The approach was to find anomalies within the dataset by building unsupervised fraud models. After cleaning the data and filling in missing fields using values that would not cause unwanted dramatic changes in the records, I created 45 expert variables. I then z-scaled the variables to give them equal importance and conducted the Principal Component Analysis to reduce the number of dimensionalities to eight, as well as reducing the correlation between variables. Next, I built a heuristic function of z-score model and an autoencoder model that generated two fraud scores, which I combined into a final fraud score. A higher score indicates a higher probability of fraud. With this score, I then rank-ordered all the entries and found the fraud records.
# 
# In the end, the results indicated that my models were effective in identifying fraudulent records for a few reasons. First, the fraud score distributions shared similar trends and displayed reasonable shapes. Second, the top 20 records identified as potential fraud did indeed demonstrate anomalies in some area, upon manual investigation. Interestingly, several of the records appeared to be government properties, and I believe that some industry expertise can help identify the validity of these records.

# # Description of Data
# The name of the Dataset is Property Valuation and Assessment Data, and it is updated annually. This dataset contains New York City Property valuation and assessment data provided by the Department of Finance (DOF). The data is primarily used to calculate property tax, grant eligible properties exemptions and/or abatements. It covers the time between year 2010 and 2011. In total, the data has 32 fields and 1070994 records.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Sequential
from keras.models import Model
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/ny-property-data/NY property data.csv')
df.head()


# I already did data exploration and data quality report in another document. Here started with feature selections, data cleaning feature creations since I was already familiar with all the data features. 
# 
# Some data types are not reasonal in the original dataset, so we change them to what they should be. For example, 'BLDGCL' represents the Building Class and it should be a category feature, but in original dataset it is numerical feature. 
# 
# Also, there are some features seem should be changed into Category features but did not being changed, such as ZIP, B and TAXCLASS, since we need to use them for calculation later.
# 
# After Data exploration, I can figure out that there are some meaningless data with value 0. I just replaced value '0' with NAN for making fill missing values more convenient.

# # Date cleaning
# ## Feature Selection

# I selected some features from the original data. 'BBLE' will be used to identify each record and others will be used for calculation later.

# In[ ]:


df = df[['BBLE','FULLVAL', 
         'AVLAND', 
         'AVTOT', 
         'LTFRONT', 
         'LTDEPTH', 
         'BLDFRONT', 
         'BLDDEPTH', 
         'STORIES', 
         'ZIP', 
         'TAXCLASS', 
         'B',
         'BLOCK',
         'BLDGCL']]
df['BLOCK'] = df['BLOCK'].astype('category')
df['BLDGCL'] = df['BLDGCL'].astype('category')
#combine 0 and NA
df.replace(0, np.nan, inplace=True)


# ## Fill missing value

# I fill missing value based on group by.

# In[ ]:


#fill ZIP nan
df['ZIP']=df.groupby(['B','BLOCK'])['ZIP'].transform(lambda x: x.fillna(x.median()))
df['ZIP']=df.groupby(['B'])['ZIP'].transform(lambda x: x.fillna(x.median()))
a=df['ZIP'].isnull().sum()
#fill NAs: FULLVAL, AVLAND, AVTOT
HV=['FULLVAL', 'AVLAND', 'AVTOT']
for i in HV:
    df[i]=df.groupby(['BLDGCL'])[i].transform(lambda x: x.fillna(x.median()) if len(x)>=5 else x)
    df[i]=df.groupby(['B'])[i].transform(lambda x: x.fillna(x.median()))
b=df[HV].isnull().sum()
#Filling NAs: LTFRONT, LTDEPTH,BLDFRONT, BLDDEPTH
HP=['LTFRONT', 'LTDEPTH', 'BLDFRONT', 'BLDDEPTH']
for hpi in HP:
    df[hpi]=df.groupby('BLDGCL')[hpi].transform(lambda x: x.fillna(x.median()))
    df[hpi]=df.groupby('B')[hpi].transform(lambda x: x.fillna(x.median()))

df[HP].isnull().sum()
c=df[HP].isnull().sum()
# Filling NAs: STORIES
df['STORIES'] = df.groupby('BLDGCL')['STORIES'].transform(lambda x: x.fillna(x.median()))
df['STORIES'] = df.groupby('B')['STORIES'].transform(lambda x: x.fillna(x.median()))
d=df['STORIES'].isnull().sum()
print(a,b,c,d)


# # Variable Creation(45)
# 
# Process of variable creation:
# 

# In[ ]:


from IPython.display import Image
Image("../input/create-variables/Screen Shot 2019-02-22 at 10.24.17 PM.png")


# In[ ]:


df['lotarea']=df['LTFRONT']*df['LTFRONT']
df['bldarea']=df['BLDFRONT']*df['BLDDEPTH']
df['bldvol']=df['bldarea']*df['STORIES']

new_cols=['lotarea','bldarea','bldvol']
norm_vols=['FULLVAL','AVLAND', 'AVTOT']

for n in norm_vols:
    for j in new_cols:
        df[n+'/'+j]=df[n]/df[j]
# create variables
df['zip3'] = df['ZIP']//100
df['zip5'] = df['ZIP']//1

#scale value
scale_value=['FULLVAL/lotarea',
 'FULLVAL/bldarea',
 'FULLVAL/bldvol',
 'AVLAND/lotarea',
 'AVLAND/bldarea',
 'AVLAND/bldvol',
 'AVTOT/lotarea',
 'AVTOT/bldarea',
 'AVTOT/bldvol',]
scale_facter=['zip3','zip5','TAXCLASS','B']
for i in scale_value:
    df[i+'_scale by_all']=df[i]/df[i].mean()
    for j in scale_facter:
        df[i+'_scale by_'+j]=df.groupby(j)[i].apply(lambda x: x/(x.mean()))
#clean process data
df=df.drop(['FULLVAL',
 'AVLAND',
 'AVTOT',
 'LTFRONT',
 'LTDEPTH',
 'BLDFRONT',
 'BLDDEPTH',
 'STORIES',
 'ZIP',
 'TAXCLASS',
 'B',
 'BLOCK',
 'BLDGCL',
 'lotarea',
 'bldarea',
 'bldvol',
 'FULLVAL/lotarea',
 'FULLVAL/bldarea',
 'FULLVAL/bldvol',
 'AVLAND/lotarea',
 'AVLAND/bldarea',
 'AVLAND/bldvol',
 'AVTOT/lotarea',
 'AVTOT/bldarea',
 'AVTOT/bldvol',
 'zip3',
 'zip5'], axis=1)
df.head()


# In[ ]:


df.shape


# ## Dimensionality Reduction
# 
# After variable creation, we had 45 fields, which were too many for further processing. Therefore, we took the necessary steps in Python to reduce dimensionality. There were three essential steps in the dimensionality reduction process:
# 
# *  Using the z-scale method to normalize 45 variables.
# *  Performing the Principal Component Analysis (PCA) to reduce variables to eight eigenvectors.
# *  Z-scaling again to assign equal weight to all eight eigenvectors.

# ### z scale
# Z-scaling is a common method used for normalization. The standard score of sample x in z scale is calculated as:
# 
# **z = (x - u) / s**

# In[ ]:


feature_values=df.columns.values.tolist()
feature_values.pop(0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def scaleColumns(df, feature_values):
    for col in feature_values:
        df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df
scaled_df = scaleColumns(df,feature_values)


# In[ ]:


scaled_df.head()


# In[ ]:


scaled_df.set_index('BBLE', inplace=True)
scaled_df.head()


# ## Principal Component Analysis (PCA)
# The main idea of principal component analysis (PCA) is to reduce the dimensionality of a dataset consisting of many variables correlated with each other, while retaining the variation present in the dataset. PCA transforms the variables to a new set of variables, known as the principal components (or simply, the PCs). The 1st PC retains maximum variation that was present in the original components.

# ### Selecting Principal Components
# In order to determine exactly how many eigenvectors we should keep, we first selected 20 PCs and graphed the relationship between number of PCs and the cumulative explained variance. 

# In[ ]:


from sklearn.decomposition import PCA
npc=20
pca = PCA(n_components=npc)
principalComponents = pca.fit_transform(scaled_df[feature_values].values)
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
explained_variance
X=list(range(1,npc+1))
plt.bar(X,explained_variance,label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


# In[ ]:


values = list(explained_variance)
total  = 0
sums   = []

for v in values:
  total = total + v
  sums.append(total)
sums


# we can see that there is an obvious reduction between the 8th PC and the 9th PC. Therefore, we decided to select the first eight PCs.

# In[ ]:


npc=8
pca = PCA(n_components=npc)
principalComponents = pca.fit_transform(scaled_df.values)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 
                          'principal component 2', 
                          'principal component 3', 
                          'principal component 4', 
                          'principal component 5', 
                          'principal component 6', 
                          'principal component 7', 
                          'principal component 8'],index=scaled_df.index)
principalDf['BBLE']=principalDf.index
principalDf.reset_index(drop=True, inplace=True)
principalDf.head()


# ### Second time of Z scale
# After PCA, we used the z-scale method again for the selected 8 PCs to make sure that all of them have same weights going into next steps.

# In[ ]:


feature_values2=principalDf.columns.values.tolist()
feature_values2.pop(-1)


# In[ ]:


scaled_principalDf= scaleColumns(principalDf,feature_values2)


# In[ ]:


scaled_principalDf.head()


# # Algorithms

# ## Heuristic Function
# I use these 8 PCs to get all first score----zscore

# In[ ]:


Image("../input/score-image/score1.png",width=200, height=600)


# In[ ]:


n=8
scaled_principalDf['zscore']=0
for pc in feature_values2:
    scaled_principalDf['zscore']=scaled_principalDf['zscore']+(scaled_principalDf[pc])**n
scaled_principalDf['zscore']=(scaled_principalDf['zscore'])**(1/n)
scaled_principalDf.head()


# ## Autoencoder
# To build the autoencoder, we employed a neural network model. Here we use Keras to create autoemcoder. Autoencoder are made by trainning and it try to model then pattern of most number. So We use autoencoder to help find outliers.
# 
#  In the decoding process, the autoencoder decodes the data in the hidden layer by using the same encoding function in the opposite way, reproducing each data point to make it as similar to the original data as possible. This process is demonstrated below:

# In[ ]:


Image("../input/score-image/score2.2.png",width=300, height=800)


# I calculated the differences between the original data and the reproduced data in order to examine and identify which data points failed to be accurately reproduced.

# In[ ]:


Image("../input/score-image/score2.1.png",width=200, height=600)


# In[ ]:


from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
input_data=scaled_principalDf[feature_values2]
input_dim = 8
encoding_dim = 4

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# One epoch means that you have trained all dataset(all records) once,if you have 384 records,one epoch means that you have trained your model for all on all 384 records.

# In[ ]:


nb_epoch = 1
#batch_size = 
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(input_data, input_data,
                    epochs=nb_epoch,
                    shuffle=True,
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
predictions = autoencoder.predict(input_data)


# In[ ]:


autoencoded_data = pd.DataFrame(predictions,columns=['enPC1','enPC2','enPC3','enPC4','enPC5','enPC6','enPC7','enPC8'])
autoencoded_data.head()


# Calculate difference between PCs and enPCs

# In[ ]:


n=8
for i in range(n):
    scaled_principalDf[i]=abs(autoencoded_data.iloc[:,i]-scaled_principalDf.iloc[:,i])
#data.iloc[:,1] # second column of data frame (last_name)


# In[ ]:


scaled_principalDf.head()


# In[ ]:


n=8
diff=list(range(n))
scaled_principalDf['autoencoder']=0
for pc in diff:
    scaled_principalDf['autoencoder']=scaled_principalDf['autoencoder']+(scaled_principalDf[pc])**n
scaled_principalDf['autoencoder']=(scaled_principalDf['autoencoder'])**(1/n)


# In[ ]:


scaled_principalDf.head()


# ## Final Score

# In[ ]:


df=scaled_principalDf
df['Zscore_Rank'] = df['zscore'].rank(ascending=True)
df['Autoencoder_Rank'] = df['autoencoder'].rank(ascending=True)
df_final=df[['BBLE','zscore','autoencoder','Zscore_Rank','Autoencoder_Rank']]
df_final['Final_Score']=df_final['zscore']+df_final['autoencoder']
df_final['Final_Rank']=0.7*df_final['Autoencoder_Rank'] +0.5*df_final['Zscore_Rank']
df_final = df_final.sort_values(by=['Final_Rank'], ascending=False)
df_final.head(10)


# In[ ]:


fraud=df_final['BBLE'].tolist()[:10]
df.loc[df['BBLE'].isin(fraud)]


# As we could see, several records have U.S. Government or NY City as owners. The characteristics of such property is that it tends to have a large value, zero building frontage and building depth, and almost full exemption of tax.
# 
# Properties owned by individuals or without an owner were also considered as fraudulent records by our models, since their property values are either 0 or very low. They displayed many signs indicating them as fraudulent records.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#xs = np.random.normal(size=int(1e6))
fig, ax = plt.subplots(1, 3, figsize=(16,5))
ax[0].hist(df_final['Final_Score'], bins=30)
ax[0].set_yscale('log')
ax[1].hist(df_final['zscore'], bins=30)
ax[1].set_yscale('log')
ax[2].hist(df_final['autoencoder'], bins=30)
ax[2].set_yscale('log')

