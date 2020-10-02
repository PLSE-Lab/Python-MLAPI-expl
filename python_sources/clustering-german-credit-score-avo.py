#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dir = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Load Data

# In[ ]:


df = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')


# In[ ]:


df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()


# ## Check Data Types

# In[ ]:


def check_dtypes_1(df):
    '''
    Parameters :
    ------------
    df : Dataframe name 

    Step :
    ------
    > 1. Do iteration for each feature to define which one categorical and nummerical feature. 
    > 2. Columns in dataframe will be seperated based on the dtypes
    > 3. All of the column will be entered to the list that have been created

    result :
    --------
    The result will be formed as dataframe
    '''
    # Make a list for both of the data type 
    categorical_list = []
    numerical_list = []
    
    #Looping 
    for col in df.columns.tolist():
        if df[col].dtype=='object':
            categorical_list.append(col)
        else:
            numerical_list.append(col)
    
    #make dataframe that have two feature, that is categorical and numerical feature
    categorical = pd.Series(categorical_list, name='Categorical Feature')
    numerical = pd.Series(numerical_list, name='Numerical Feature')
    df_dtypes = pd.concat([categorical,numerical], axis=1)
    
    return df_dtypes


# In[ ]:


#1. Input dataframe
dataframe = df
#2. Seperated 
check_dtypes_1(dataframe)


# ### List Types

# In[ ]:


def list_dtypes(df):
    categorical_list = []
    numerical_list = []
    for col in df.columns.tolist():
        if df[col].dtype=='object':
            categorical_list.append(col)
        else:
            numerical_list.append(col)
    print('Number of categorical features:', str(len(categorical_list)))
    print('Number of numerical features:', str(len(numerical_list)))

    return categorical_list, numerical_list


# In[ ]:


categorical_list, numerical_list = list_dtypes(df)


# ## Check Missing Value

# Function to make missing value gone. Using median for numerical and mode for categorical.

# In[ ]:


# module Check missing value 
def missing_value(df):
    '''
    Documentation :
    --------------
    * df : Dataframe Name
    '''
    #count the number of missing value 
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)
    missing  = pd.concat([total, percent], axis=1, keys=['Total_Missing', 'Percent(%)'])
    
    return missing.head(20)


# In[ ]:


#1. input dataframe
dataframe = df

#2. input dataframe to the module 
missing_value(dataframe)


# ## Fill Missing

# In[ ]:


def fill_missing(df, feature_list = None , vartype = None ):
    '''
    Documentation :
    ---------------
    df              : object, dataframe
    feature_list    : feature list is the set of numerical or categorical features 
                      that have been seperated before
    vartype         : variable type : continuos or categorical, default (numerical)
                        (0) numerical   : variable type continuos/numerical
                        (1) categorical : variable type categorical
    Note :
    ------
    > if numerical variable will be filled by median 
    > if categorical variabe will filled by modus
    > if have been made variebles based on the dtypes list before, 
      insert it into feature list in the function.     

    Example :
    ---------
    # 1. Define feature that will be filled in 
      num_feature = numeric_list
      
    # 2. Input Dataframe
      dataframe = df
      
    # 3. Vartype
      var_type = 0
      
    # 4. Filling Value
      Fill_missing(dataframe, num_feature, var_type)
    '''
    #default vartype 
    if vartype == None :
        vartype = 'numerical'

    # filling numerical data with median 
    if vartype == 'numerical' :
        for col in feature_list:
            df[col] = df[col].fillna(df[col].median())
    
    # filling categorical data with modus  
    if vartype == 'categorical' :
        for col in feature_list:
            df[col] = df[col].fillna(df[col].mode().iloc[0])


# In[ ]:


# 1. define feature that will be filled in 
num_feature = numerical_list

# 3. Vartype
var_type = 'numerical'

# 4. Filling Value
fill_missing(df, num_feature, var_type)


# In[ ]:


# 1. define feature that will be filled in 
cat_feature = categorical_list

# 2. Vartype
var_type = 'categorical'

# 3. Filling Value
fill_missing(df, cat_feature, var_type)


# In[ ]:


# Df Numeric 
df_num_list = df[numerical_list]
df_num_list.drop('Job',axis=1,inplace=True)
df_num_list.head()


# In[ ]:


# Df categoric 
df_categ = df[categorical_list]
df_categ['Job'] = df['Job']
df_categ.head()


# ## Label Encoder

# Change numerical to categorical with rules:
# Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

# In[ ]:


for x in range(len(df)):
    if df['Job'][x]==0:
        df['Job'][x]='unskilled and non-resident'
    if df['Job'][x]==1:
        df['Job'][x]='unskilled and resident'
    if df['Job'][x]==2:
        df['Job'][x]='skilled'
    if df['Job'][x]==3:
        df['Job'][x]='highly skilled'


# In[ ]:


df2=df.copy()


# In[ ]:


df2.head()


# In[ ]:


df2=pd.get_dummies(df2,prefix=['Sex','Job','Housing','Saving Account','Checking account','Purpose'], drop_first=True)


# Make variable label get them own column. Using get dummies

# ## See The Distribution Data

# In[ ]:


def distributions(df):
    plt.figure(figsize=(10,5))
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.distplot(df["Age"], ax=ax1)
    sns.distplot(df["Credit amount"], ax=ax2)
    sns.distplot(df["Duration"], ax=ax3)
    plt.tight_layout()


# In[ ]:


distributions(df2)


# In[ ]:


df2_log = df2.copy()
df2_log['Age'] = np.log(df2_log['Age'])
df2_log['Credit amount'] = np.log(df2_log['Credit amount'])
df2_log['Duration'] = np.log(df2_log['Duration'])
distributions(df2_log)


# ## Normalization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


df3 = df2_log.copy()
df3_num = df3[['Age','Credit amount','Duration']]


# In[ ]:


df3_num[['Age','Credit amount','Duration']] = scaler.fit_transform(df3[['Age','Credit amount','Duration']])


# In[ ]:


df3[['Age','Job_skilled',
       'Job_unskilled and non-resident', 'Job_unskilled and resident', 'Credit amount', 'Duration', 'Sex_male', 'Housing_own',
       'Housing_rent', 'Saving Account_moderate', 'Saving Account_quite rich',
       'Saving Account_rich', 'Checking account_moderate',
       'Checking account_rich', 'Purpose_car', 'Purpose_domestic appliances',
       'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV',
       'Purpose_repairs', 'Purpose_vacation/others']] = scaler.fit_transform(df3[['Age', 'Job_skilled',
       'Job_unskilled and non-resident', 'Job_unskilled and resident', 'Credit amount', 'Duration', 'Sex_male', 'Housing_own',
       'Housing_rent', 'Saving Account_moderate', 'Saving Account_quite rich',
       'Saving Account_rich', 'Checking account_moderate',
       'Checking account_rich', 'Purpose_car', 'Purpose_domestic appliances',
       'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV',
       'Purpose_repairs', 'Purpose_vacation/others']])


# ## K-Means

# ### Elbow Method

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wcss = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df3_num)
    wcss.append(km.inertia_)
plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of centroids')
plt.ylabel('')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


df_kmeans = df3_num.copy()
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_kmeans)


# ### Silhouette Score

# In[ ]:


from sklearn.metrics import silhouette_score
silhouette_score(df_kmeans, kmeans.labels_)


# In[ ]:


df_kmeans['kluster'] = kmeans.labels_


# In[ ]:


from mpl_toolkits import mplot3d


# In[ ]:


import seaborn as sns; sns.set()

plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')

xdata = df['Age']
ydata = df['Credit amount']
zdata = df['Duration']

ax.set_xlabel('Age')
ax.set_ylabel('Credit amount')
ax.set_zlabel('Duration')

ax.scatter3D(xdata, ydata, zdata, c=df_kmeans['kluster'], cmap='viridis');


# ### Dendogram

# In[ ]:


plt.figure(figsize=(15,5))
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(df3_num, method='ward'))


# ### DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


df_dbscan = df3_num.copy()


# In[ ]:


dbscan = DBSCAN(eps=0.09, min_samples=5)
dbscan.fit(df_dbscan)


# In[ ]:


import seaborn as sns; sns.set()

plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')

xdata = df['Age']
ydata = df['Credit amount']
zdata = df['Duration']

ax.set_xlabel('Age')
ax.set_ylabel('Credit amount')
ax.set_zlabel('Duration')

ax.scatter3D(xdata, ydata, zdata, c=dbscan.labels_, cmap='viridis');


# Seperti yang kita lihat, DBSCAN kita tidak cocok dengan data. Maka dari itu, k-means dipilih untuk klustering dan dipilih 3 kluster.

# ## Insight

# Setelah melakukan klustering, kita mau melihat insight dari kluster yang dihasilkan. Dengan dibandingkan oleh variabel, kita harap dapat ditemukan insight yang baru.

# Cluster 0<br>
#     Age: 21-63 (Sedang)<br>
#     Credit amount: 909-18424 (Tinggi)<br>
#     Duration: 12 - 72 (Panjang)<br>
# <br>
# Cluster 1<br>
#     Age: 19-35 (Muda)<br>
#     Credit amount: 276 - 14555 (Sedang)<br>
#     Duration: 4 - 30 (Rendah)<br>
# <br>
# Cluster 2<br>
#     Age: 36-75 (Tua)<br>
#     Credit: 250 -14896 (Kecil)<br>
#     Duration: 4 - 42 (Sedang)<br>
#     <br>
# Dengan melihat data, maka dapat disimpulkan bahwasanya<br>
#     Cluster 0 = Medium<br>
#     Cluster 1 = Good<br>
#     Cluster 2 = Bad<br>

# Cluster 0 = Medium, karena walaupun secara usia mereka sedang dan creditnya tinggi, tetapi durasinya terlalu panjang.
# <br>
# Cluster 1 = Good, karena secara credit memiliki tingkatan sedang dan durasinya rendah. Ditambah usia yang meminjam juga muda.
# <br>
# Cluster 2 = Bad, karena usia Tua (berumur) dan memiliki tingakatan kredit yang kecil.

# In[ ]:


df['k_means_label'] = kmeans.labels_


# In[ ]:


for x in range(len(df)):
    if df['k_means_label'][x]==0:
        df['k_means_label'][x]='medium'
    if df['k_means_label'][x]==1:
        df['k_means_label'][x]='good'
    if df['k_means_label'][x]==2:
        df['k_means_label'][x]='bad'


# ### Visualisasi

# **Jenis kelamin dengan kluster yang dihasilkan**

# In[ ]:


fig = plt.figure(figsize=(12,6))
plt.title('Count Plot',fontsize = 20)
ax=sns.countplot(data=df, x='Sex',hue="k_means_label")
ax.set_xlabel('Sex', fontsize = 15)
ax.tick_params(labelsize=12)


# Untuk jenis kelamin laki-laki, di dominasi oleh kelas medium risk. Sedangkan untuk perempuan, lebih kecil risknya.

# #### Count plot pada Job dengan label clustering

# In[ ]:


fig = plt.figure(figsize=(12,6))
plt.title('Count Plot',fontsize = 20)
ax=sns.countplot(data=df, x='Job',hue="k_means_label")
ax.set_xlabel('Job', fontsize = 15)
ax.tick_params(labelsize=12)


# Untuk kelas low risk (good), kebanyakan berada pada kelas yang memiliki kemampuan tinggi (skilled), sedangkan pada medium risk (medium), malah lebih baik dari sisi kemampuan.

# #### Count plot pada Tujuan dengan label klustering

# In[ ]:


fig = plt.figure(figsize=(12,6))
plt.title('Count Plot',fontsize = 20)
ax=sns.countplot(data=df, x='Purpose',hue="k_means_label")
ax.set_xlabel('Purpose', fontsize = 15)
ax.tick_params(labelsize=12)
plt.xticks(rotation=30)


# Dapat dilihat bahwasanya, radio/TV dan mobil merupakan hal yang menjadi tujuan dari kluster high risk (bad). Dapat terlihat juga bahwasanya mobil, radio/tv menjadi tujuan dari peminjaman.
# Yang menarik adalah vacation\others masih memiliki nilai yang rendah, ini sebenarnya adalah peluang yang dapat di stimulus agar ditingkatkan.
