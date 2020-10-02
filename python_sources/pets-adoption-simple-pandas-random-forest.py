#!/usr/bin/env python
# coding: utf-8

# # PetFinder Competition: a starting point

# Welcome everybody, this is my first Kernel and my first competition. This notebook is something like a diary of a journey. My aim was starting with something simple and adding details gradually. These have been the main steps:
# 
# * First step: logistic regression, object columns dropped, json files not used.
# * Second step: random forest, object columns simplified and categorised, json files not used.
# * Third sted: data preparation on  json files and move to my next kernel 
# 
# As you can see, there's a lot of WIP. I hope it could be useful to someone anyway. Pointings and suggestions are welcome.

# ### Import and Data Loading

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import os
train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### EDA and Data Preparation

# In[ ]:


train.head()


# In[ ]:


train['Breed1'].nunique()


# In[ ]:


train['Breed2'].nunique()


# In[ ]:


train['Color1'].nunique()


# In[ ]:


test.head()


# In[ ]:


color_labels.head()


# In[ ]:


state_labels.head()


# In[ ]:


breed_labels.head()


# In[ ]:


breed_labels.tail()


# To sum up:  some categorical variables implemented by integers. Some categorical columns with (too?) many unique values. The columns Type (1=Dog, 2=Cat) and Breed1/Breed2 have to be read together.

# In[ ]:


test['AdoptionSpeed']=-1
test=test.reindex(columns=train.columns)
df_all=pd.concat([train,test]).copy()
df_all.head()


# In[ ]:


def print_columns_with_null(df):
    dfn=df.isnull().sum()
    return dfn[dfn>0]


# In[ ]:


print_columns_with_null(df_all)


# In[ ]:


df_all['Name'].fillna('No Name', inplace=True)
df_all['Description'].fillna('*Missing*', inplace=True)


# In[ ]:


print_columns_with_null(df_all)


# In[ ]:


df_all['AdoptionSpeed']=df_all['AdoptionSpeed'].apply(str)


# In[ ]:


fig,ax = plt.subplots(figsize=(16,4))
ax = sns.distplot(df_all['Age'])


# From data description: The Age is expressed in months (250/12=20.8 years)

# In[ ]:


df_all['Breed1'].value_counts().head(10)


# In[ ]:


df_all['Breed2'].value_counts().head(10)


# ##### Fist Step: Simply drop string columns

# In[ ]:


#df_all=df_all.drop(['Name','RescuerID','Description'],axis=1)


# ##### Second Step: Transform string columns in something with less unique values

# In[ ]:


df_all['Name'].value_counts().head(10)


# In[ ]:


def simple_name(name) :
    if (name in ['No Name','Baby','Lucky','Brownie','Mimi','Blackie','Kitty']) :
        return name
    else:
        return 'Other Names'


# In[ ]:


simple_name('Baby')


# In[ ]:


simple_name('Aquaman')


# In[ ]:


df_all['SimpleName']=df_all['Name'].apply(simple_name)


# In[ ]:


df_all['Description'].value_counts().head()


# In[ ]:


df_all['DescriptionLength']=df_all['Description'].apply(len)


# In[ ]:


df_all['DescriptionLength'].describe()


# In[ ]:


df_all['DescriptionLength'].value_counts().head(10)


# In[ ]:


sns.distplot(df_all['DescriptionLength'])


# In[ ]:


def descr_cat(descr) :
    ldescr = len(descr)
    if (ldescr<=15) : return '015'
    if ((ldescr>15)&(ldescr<=50)) : return '050'
    if ((ldescr>50)&(ldescr<=250)) : return '250'
    if ((ldescr>250)&(ldescr<=750)) : return '750'
    if (ldescr>750) : return '750+'


# In[ ]:


descr_cat('For Adoption')


# In[ ]:


df_all['DescrLengthCat']=df_all['Description'].apply(descr_cat)


# In[ ]:


df_all['RescuerID'].value_counts().head(10)


# In[ ]:


def simple_rid(id) :
    if (id in ['fa90fa5b1ee11c86938398b60abc32cb','aa66486163b6cbc25ea62a34b11c9b91',
                 'c00756f2bdd8fa88fc9f07a8309f7d5d','b53c34474d9e24574bcec6a3d3306a0d',
                 'ee2747ce26468ec44c7194e7d1d9dad9','4475f31553f0170229455e3c5645644f',
                 '95481e953f8aed9ec3d16fc4509537e8','b770bac0ca797cf1433c48a35d30c4cb']) :
        return id
    else:
        return 'Other Rescuers'


# In[ ]:


simple_rid('c00756f2bdd8fa88fc9f07a8309f7d5d')


# In[ ]:


simple_rid('Ocean Master')


# In[ ]:


df_all['SimpleRescuerID']=df_all['RescuerID'].apply(simple_rid)


# In[ ]:


def simple_b2(breed) :
    if (breed==0) : 
        return '0'
    else :
        return '1'


# In[ ]:


df_all['SimpleBreed2']=df_all['Breed2'].apply(simple_b2)


# In[ ]:


def simple_b1(breed) :
    if (breed in [307,266,265,299,264,292,285]) :
        return str(breed)
    else:
        return 'Other Breeds'


# In[ ]:


df_all['SimpleBreed1']=df_all['Breed1'].apply(simple_b1)


# In[ ]:


print_columns_with_null(df_all)


# First Checkpoint for df_all

# In[ ]:


df_all.to_csv('df_all0.csv', index=False)


# ##### Third Step: Taking into the account data from json files

# *Text*

# In[ ]:


from pandas.io.json import json_normalize


# Just some tests/short scripts before coding the function ...

# In[ ]:


curr_path= '../input/train_sentiment/'
curr_name=os.listdir('../input/train_sentiment')[3]
curr_file = curr_path + curr_name
curr_file


# In[ ]:


curr_json=pd.read_json(curr_file, orient='index', typ='series')
curr_json


# In[ ]:


json_normalize(curr_json.sentences)


# In[ ]:


json_normalize(curr_json.entities)


# In[ ]:


curr_ds=json_normalize(curr_json.documentSentiment)
curr_ds.magnitude[0]


# In[ ]:


txt_df = pd.DataFrame(columns=['PetID','magnitude','score','language'])


# In[ ]:


curr_name.split('.')[0]


# In[ ]:


df_all[df_all['PetID']=='378fcc4fc']


# In[ ]:


curr_row=pd.DataFrame(data=[curr_name.split('.')[0],curr_json.language,curr_ds.magnitude[0],curr_ds.score[0]], index=['PetID','magnitude','score','language']).transpose()
curr_row


# In[ ]:


txt_df.append(curr_row)


# In[ ]:


curr_json.language


# Let's write a for-loop function to extract the most important information embedded in the sentiment-json files.

# In[ ]:


def fill_txt_df(curr_path) :
    curr_names=os.listdir(curr_path)
    txt_df = pd.DataFrame(columns=['PetID','magnitude','score','language'])
    for curr_name in curr_names :
        curr_file = curr_path + curr_name
        # print(curr_file)
        curr_json=pd.read_json(curr_file, orient='index', typ='series')
        curr_ds=json_normalize(curr_json.documentSentiment)
        curr_row=pd.DataFrame(data=[curr_name.split('.')[0],curr_ds.magnitude[0],curr_ds.score[0],curr_json.language],
                 index=['PetID','magnitude','score','language']).transpose()
        txt_df=txt_df.append(curr_row)
    return txt_df


# In[ ]:


txt_df_train = fill_txt_df('../input/train_sentiment/')
txt_df_train.describe()


# In[ ]:


txt_df_test = fill_txt_df('../input/test_sentiment/')
txt_df_test.describe()


# In[ ]:


txt_data=txt_df_train.append(txt_df_test)
txt_data.describe()


# In[ ]:


txt_data.dtypes


# In[ ]:


txt_data.isna().sum()


# In[ ]:


txt_data.dtypes


# In[ ]:


def cols_to_numeric(df, col_names):
    for k in range(0,len(col_names)) :
        df[col_names[k]]=pd.to_numeric(df[col_names[k]],errors='coerce')
    return df


# In[ ]:


col_names=['magnitude','score']
txt_data = cols_to_numeric(txt_data, col_names)
txt_data.dtypes


# In[ ]:


txt_data.to_csv('txt_data.csv', index=False)


# In[ ]:


fig,ax = plt.subplots(figsize=(12,4))
ax = sns.distplot(txt_data['magnitude'])


# In[ ]:


fig,ax = plt.subplots(figsize=(12,4))
ax = plt.hist(txt_data['score'], bins=15)


# *Images*

# In[ ]:


curr_path= '../input/train_metadata/'


# In[ ]:


curr_names=os.listdir('../input/train_metadata')
len(curr_names)


# So there are many json per pet, as we was expecting, due to PhotoAmt>1.

# In[ ]:


curr_name=os.listdir('../input/train_metadata')[432]
curr_file = curr_path + curr_name
curr_file


# In[ ]:


curr_json=pd.read_json(curr_file, orient='index', typ='series')
curr_json


# In[ ]:


jla=json_normalize(curr_json.labelAnnotations)
jla


# In[ ]:


jla['description'].str.cat(sep=', ')


# In[ ]:


jla[jla['score']==jla['score'].max()]


# In[ ]:


jpa=json_normalize(curr_json.imagePropertiesAnnotation)
jpa.iloc[0]


# In[ ]:


jpa.iloc[0][0]


# In[ ]:


len(jpa.iloc[0][0])


# In[ ]:


k=2


# In[ ]:


jpa.iloc[0][0][k]['color']


# In[ ]:


jpa.iloc[0][0][k]['score']


# In[ ]:


jpa.iloc[0][0][k]['pixelFraction']


# In[ ]:


df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
df_jpa


# In[ ]:


data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
pd.DataFrame(data=data,index=['color','score','pixelFraction']).transpose()


# In[ ]:


df_jpa=pd.DataFrame(columns=['color','score','pixelFraction'])
for k in range(0,len(jpa.iloc[0][0])) :
    curr_data=[str(jpa.iloc[0][0][k]['color']),jpa.iloc[0][0][k]['score'],jpa.iloc[0][0][k]['pixelFraction']]
    curr_row=pd.DataFrame(data=curr_data,index=['color','score','pixelFraction']).transpose()
    df_jpa=df_jpa.append(curr_row)
df_jpa['prod']=df_jpa['score']*df_jpa['pixelFraction']
df_jpa


# In[ ]:


len(df_jpa[df_jpa['prod']==df_jpa['prod'].max()])


# In[ ]:


jcha=json_normalize(curr_json.cropHintsAnnotation)
jcha.iloc[0]


# In[ ]:


jcha.iloc[0][0]


# In[ ]:


jcha.iloc[0][0][0]['boundingPoly']


# In[ ]:


jcha.iloc[0][0][0]['confidence']


# In[ ]:


jcha.iloc[0][0][0]['importanceFraction']


# Is it possible to summarise all these data? Honestly speaking, the following it's only an attempt. I read the documentation on Google API but I don't know if I made the right choices. In other words: any suggestion, especially for this part, will be highly appreciated. Anyway, let's go on. The idea is to create a dataframe to join later with df_all. As there are many images per pet, when we will join all the data we will have to choose one of the image rows, using a function like max() or sum().
# 
# First of all, some useful secondary functions:
# 

# In[ ]:


def not_empty(s) :
    if len(s)>=1 : 
        return s[0]
    else :
        return ''


# In[ ]:


def extr_jla_info(curr_json) :
    dfr=pd.DataFrame(columns=['score','description'])
    try :
        jlas=json_normalize(curr_json.labelAnnotations)
        curr_data=[jlas[jlas['score']==jlas['score'].max()]['score'][0],
                   jlas['description'].str.cat(sep=',')]
        curr_row=pd.DataFrame(data=curr_data,index=['score','description']).transpose()
        dfr=dfr.append(curr_row)
    except:
        print('Line Skipped in JLA')
        dfr=pd.DataFrame(data=[0,0], index=['score','description']).transpose()
    return dfr


# In[ ]:


def extr_jpa_info(curr_json) :
    dfr=pd.DataFrame(columns=['color','score','pixelFraction','prod'])
    try :
        jpas=json_normalize(curr_json.imagePropertiesAnnotation)
        for k in range(0,len(jpas.iloc[0][0])) :
            curr_data=[str(jpas.iloc[0][0][k]['color']),
                   jpas.iloc[0][0][k]['score'],
                   jpas.iloc[0][0][k]['pixelFraction'],
                   jpas.iloc[0][0][k]['score']*jpas.iloc[0][0][k]['pixelFraction']]
            curr_row=pd.DataFrame(data=curr_data,index=['color','score','pixelFraction','prod']).transpose()
            dfr=dfr.append(curr_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]    
    except :
        print('Line Skipped in JPA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['color','score','pixelFraction','prod']).transpose()
    return dfr


# In[ ]:


def extr_jcha_info(curr_json) :
    dfr=pd.DataFrame(columns=['boundingPoly','confidence','importanceFraction','prod'])
    try :
        jchas=json_normalize(curr_json.cropHintsAnnotation)
        for k in range(0,len(jchas.iloc[0][0])) :
            curr_data=[str(jchas.iloc[0][0][k]['boundingPoly']),
                       jchas.iloc[0][0][k]['confidence'],
                       jchas.iloc[0][0][k]['importanceFraction'],
                       jchas.iloc[0][0][k]['confidence']*jchas.iloc[0][0][k]['importanceFraction']]
            curr_row=pd.DataFrame(data=curr_data,index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
            dfr=dfr.append(curr_row)
        dfr=dfr[dfr['prod']==dfr['prod'].max()]
    except :
        print('Line Skipped in JCHA')
        dfr=pd.DataFrame(data=['',0,0,0], index=['boundingPoly','confidence','importanceFraction','prod']).transpose()
    return dfr


# Here is the main function:

# In[ ]:


import datetime


# In[ ]:


def fill_img_df(curr_path,k_from, k_to) :
    curr_names=os.listdir(curr_path)
    k_to=min(k_to,len( curr_names))
    curr_names=curr_names[k_from:k_to]
    col_names=['PetID','ImgID','jla_description','jla_score','jpa_color','jpa_score','jpa_pixel_fract','jcha_bounding','jcha_confidence','jcha_import_fract']
    img_df = pd.DataFrame(columns=col_names)
    i=k_from
    for curr_name in curr_names :
        i=i+1
        if (i%1000==0) : print('Current File nr.:{}'.format(i),datetime.datetime.now())
        curr_file = curr_path + curr_name
        pet_id=curr_name.split('.')[0].split('-')[0]
        img_id=curr_name.split('.')[0].split('-')[1]
        #
        curr_json=pd.read_json(curr_file, orient='index', typ='series')
        info_jla=extr_jla_info(curr_json)  # contains description e score
        info_jpa=extr_jpa_info(curr_json)  # contains RGB as string, score, pixelFraction
        info_jcha=extr_jcha_info(curr_json) # contains boundingPoly, confidence, importanceFraction
        #
        curr_row=pd.DataFrame(data=[pet_id, img_id,
                                    not_empty(info_jla['description']),
                                    not_empty(info_jla['score']),
                                    not_empty(info_jpa['color']),
                                    not_empty(info_jpa['score']),
                                    not_empty(info_jpa['pixelFraction']),
                                    not_empty(info_jcha['boundingPoly']),
                                    not_empty(info_jcha['confidence']),
                                    not_empty(info_jcha['importanceFraction'])],
                 index=col_names).transpose()
        img_df=img_df.append(curr_row)
    return img_df


# The process will take a lot of time, let's make up 3 restore points.

# In[ ]:


img_df1=fill_img_df('../input/train_metadata/',0,20000)
img_df1.to_csv('../img_df1.csv', index=True)


# In[ ]:


img_df2=fill_img_df('../input/train_metadata/',20000,40000)
img_df2.to_csv('../img_df2.csv', index=True)


# In[ ]:


img_df3=fill_img_df('../input/train_metadata/',40000,60000)
img_df3.to_csv('../img_df3.csv', index=True)


# In[ ]:


img_df=pd.concat([img_df1,img_df2,img_df3], axis=0)


# In[ ]:


col_names=['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract']
img_df = cols_to_numeric(img_df,col_names)
img_df.dtypes


# Let's add a compact rating index

# In[ ]:


img_df['rating']=img_df['jla_score']*img_df['jpa_score']*img_df['jpa_pixel_fract']*img_df['jcha_confidence']*img_df['jcha_import_fract']


# Sorry, but at the moment I dont't know how to use these columns

# In[ ]:


img_df.drop('jpa_color', axis=1, inplace=True)


# In[ ]:


img_df.drop('jcha_bounding', axis=1, inplace=True)


# With the others ...

# In[ ]:


img_df.sort_values(['PetID','jla_description'], inplace=True)


# In[ ]:


img_df.head()


# And now another hard choice: is it more important the best image or having several good images? 

# In[ ]:


img_dfd=img_df[['PetID','jla_description']].groupby(['PetID']).min()
img_dfd.head()


# In[ ]:


img_dfm=img_df.groupby(['PetID']).mean()
img_dfm.head()


# In[ ]:


img_dfc=img_df[['PetID']].groupby(['PetID']).size()
img_dfc.head()


# In[ ]:


img_data=img_dfd


# In[ ]:


col_names = ['jla_score','jpa_score','jpa_pixel_fract','jcha_confidence','jcha_import_fract']
for cn in col_names :
    img_data[cn]=img_dfm[cn]


# In[ ]:


img_data['cnt']=img_dfc
img_data.head()


# In[ ]:


img_data.to_csv('img_data.csv', index=True)


# *Fourth Step: Processing seriously text columns (with the help of others)*

# Before starting this competition I had only vague ideas about this point. But thanks to Kaggle I found this fantastic Kernel:
# 
# https://www.kaggle.com/fiancheto/petfinder-simple-lgbm-baseline-lb-0-399
# 
# So let's try to replicate something. I select the Description column, in a second moment is possible to extend the reasoning to the jla_description one.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[ ]:


tfv = TfidfVectorizer(analyzer='word', stop_words = 'english', token_pattern=u'\w{4,}')


# In[ ]:


corpus=list(df_all['Description'])
corpus[0:5]


# In[ ]:


des_trasf=tfv.fit_transform(corpus)


# In[ ]:


tfv.get_feature_names()[0:5]


# Ok, I dropped the world under 4 chars, but I have to understand better token_patter...

# In[ ]:


svd = TruncatedSVD(n_components=300)


# In[ ]:


svd.fit(des_trasf)


# This is the % of variance "captured" by the selected value of n_components

# In[ ]:


svd.explained_variance_ratio_.sum()


# In[ ]:


des_svd=svd.transform(des_trasf)


# In[ ]:


des_svd.shape


# In[ ]:


des_svd.shape[1]


# In[ ]:


des_svd_df0=pd.DataFrame(des_svd)


# In[ ]:


def gen_col_names(svd_array,prefix):
    col_names=[]
    for i in range(0,svd_array.shape[1]) :
        col_names.append(prefix+str(i))
    return col_names


# In[ ]:


col_names=gen_col_names(des_svd_df0,'des_svd')
col_names[0:5]


# In[ ]:


des_svd_df0.columns=col_names
des_svd_df0.iloc[0:5,0:5]


# In[ ]:


dpi=df_all[['PetID','Description']]
dpi.head()


# In[ ]:


des_svd_df=dpi.join(des_svd_df0)
des_svd_df=des_svd_df.reset_index(drop=True)
des_svd_df


# Let's do it with jla_description, too

# In[ ]:


img_data.head()


# In[ ]:


tfv = TfidfVectorizer(stop_words = 'english', token_pattern=u'\w{3,}')


# In[ ]:


corpus=list(img_data['jla_description'].apply(str))
corpus[0:3]


# In[ ]:


jlad_trasf=tfv.fit_transform(corpus)


# In[ ]:


tfv.get_feature_names()[0:5]


# In[ ]:


svd = TruncatedSVD(n_components=20)


# In[ ]:


svd.fit(jlad_trasf)


# Here we can do with a few of SVS

# In[ ]:


svd.explained_variance_ratio_.sum()


# In[ ]:


jlad_svd=svd.transform(jlad_trasf)


# In[ ]:


jlad_svd.shape


# In[ ]:


jlad_svd_df0=pd.DataFrame(jlad_svd)


# In[ ]:


col_names=gen_col_names(jlad_svd_df0,'jlad_svd')
col_names[0:5]


# In[ ]:


jlad_svd_df0.columns=col_names
jlad_svd_df0.head()


# In[ ]:


dpi=pd.DataFrame(img_data['jla_description'])
dpi=dpi.reset_index('PetID')
dpi.head()


# In[ ]:


jlad_svd_df=pd.concat([dpi,jlad_svd_df0],axis=1)
jlad_svd_df.head()


# Finally, resulting from the SVD process, we have two dataframes: des_svd_df and jlad_svd_df.

# In[ ]:


des_svd_df.to_csv('des_svd_df.csv',index=False)


# In[ ]:


jlad_svd_df.to_csv('jlad_svd_df.csv',index=False)


# ## To sum up
# At this point we have created these files:
# 
# * df_all0, all the basic columns + simplified/categorised ones
# * txt_data, img_data, information (partially) loaded from json files in dataframes
# * df_svd_df, jlad_svd_df, SVD of df_all.Description and img_data.jla_description (key=PetID)
# 
# To avoid the confusion present in some of previous versions, this kernel has been simplified to use only df_all0. A next kernel will be based on df_all0 + the svds.
# 

# *Final Drops*

# In[ ]:


df_all.head()


# In[ ]:


df_all.drop('Name', axis=1, inplace=True) # created categorical SimpleName
df_all.drop('RescuerID', axis=1, inplace=True) # created categorical SimpleRescuerID
df_all.drop('Description', axis=1, inplace=True) # created categorical DescrLengthCat
df_all.drop('DescriptionLength', axis=1, inplace=True) # created categorical DescrLengthCat


# Use Breeds or SimpleBreeds? Running the models I made this choice

# In[ ]:


df_all.drop('Breed2', axis=1, inplace=True)
#df_all.drop('SimpleBreed2', axis=1, inplace=True)


# In[ ]:


#df_all.drop('Breed1', axis=1, inplace=True)
df_all.drop('SimpleBreed1', axis=1, inplace=True)


# In[ ]:


df_all.head()


# ### Modeling

# In[ ]:


df_all.dtypes


# In[ ]:


simple_name_dummies=pd.get_dummies(df_all['SimpleName'],drop_first=True)
descr_len_dummies=pd.get_dummies(df_all['DescrLengthCat'],drop_first=True)
simple_rid_dummies=pd.get_dummies(df_all['SimpleRescuerID'],drop_first=True)
simple_b2_dummies=pd.get_dummies(df_all['SimpleBreed2'],drop_first=True)


# In[ ]:


df_all=pd.concat([df_all,simple_name_dummies,descr_len_dummies,
                  simple_rid_dummies], axis=1)
df_all.drop('SimpleName', axis=1, inplace=True)
df_all.drop('DescrLengthCat', axis=1, inplace=True)
df_all.drop('SimpleRescuerID', axis=1, inplace=True)
df_all.drop('SimpleBreed2', axis=1, inplace=True)


# In[ ]:


df_all.head()


# In[ ]:


dftrain=df_all[np.invert(df_all['AdoptionSpeed']=='-1')].copy()
dftest=df_all[df_all['AdoptionSpeed']=='-1'].copy()


# In[ ]:


dftest.head()


# In[ ]:


dftest_ids=dftest['PetID']
dftest_ids.head()


# In[ ]:


dftrain=dftrain.drop(['PetID'],axis=1)
dftest=dftest.drop(['PetID'],axis=1)


# In[ ]:


dftrain.head()


# In[ ]:


dftest=dftest.drop('AdoptionSpeed',axis=1)
y = dftrain['AdoptionSpeed']
X = dftrain.drop('AdoptionSpeed',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)


# In[ ]:


from sklearn.metrics import cohen_kappa_score


# In[ ]:


from sklearn.model_selection import GridSearchCV


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lrm = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)


# In[ ]:


lrm.fit(X_train,y_train)


# In[ ]:


y_test_pred=lrm.predict(X_test)


# In[ ]:


y_test.head()


# In[ ]:


y_test_pred[0:4]


# In[ ]:


cohen_kappa_score(y_test_pred, y_test, weights='quadratic')


# In[ ]:


lrm.fit(X,y)


# In[ ]:


y_pred_lrm=lrm.predict(dftest)


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier()


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_test_pred = rfc.predict(X_test)


# In[ ]:


dfi=pd.DataFrame(data=[X_test.columns,rfc.feature_importances_], index=['Col','Imp'])
dfi=dfi.transpose()
dfi.sort_values(by='Imp',ascending=False).head()


# In[ ]:


sns.boxplot(x='AdoptionSpeed', y='Age', data=dftrain)


# In[ ]:


sns.boxplot(x='AdoptionSpeed', y='PhotoAmt', data=dftrain)


# In[ ]:


cohen_kappa_score(y_test_pred, y_test, weights='quadratic')


# Tuning

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [6],
#     'max_features': [24],
#     'min_samples_leaf': [2],
#     'min_samples_split': [3],
#     'n_estimators': [64]
# }

# In[ ]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [6],
    'max_features': [24],
    'min_samples_leaf': [2],
    'min_samples_split': [3],
    'n_estimators': [64]
}


# In[ ]:


gs = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 3, n_jobs = 4, verbose=10)


# In[ ]:


gs.fit(X_train,y_train)


# In[ ]:


gs.best_params_


# In[ ]:


rfc=gs.best_estimator_


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_test_pred = rfc.predict(X_test)


# In[ ]:


cohen_kappa_score(y_test_pred, y_test, weights='quadratic')


# In[ ]:


rfc.fit(X,y)


# In[ ]:


y_pred_rf = rfc.predict(dftest)


# ### XGB

# In[ ]:


import xgboost as xgb
xgc = xgb.XGBClassifier()


# In[ ]:


xgc.fit(X_train,y_train)


# In[ ]:


y_test_pred = xgc.predict(X_test)


# In[ ]:


dfi=pd.DataFrame(data=[X_test.columns,xgc.feature_importances_], index=['Col','Imp'])
dfi=dfi.transpose()
dfi.sort_values(by='Imp',ascending=False).head(10)


# In[ ]:


cohen_kappa_score(y_test_pred, y_test, weights='quadratic')


# Tuning

# param_grid = {
#     'max_depth': [16],
#     'learning_rate': [0.3],
#     'gamma': [6],
#     'n_estimators': [64],
#     'subsample': [0.9],
#     'reg_lambda': [0.9],
#     'colsample_bytree': [0.9],
# }

# In[ ]:


param_grid = {
    'max_depth': [12,16],
    'learning_rate': [0.1],
    'gamma': [6,12],
    'n_estimators': [64,96],
    'subsample': [0.9],
    'reg_lambda': [0.9],
    'colsample_bytree': [0.9],
}


# In[ ]:


gs = GridSearchCV(estimator = xgc, param_grid = param_grid, cv = 3, n_jobs = 4 ,verbose=10)


# In[ ]:


gs.fit(X_train,y_train)


# In[ ]:


gs.best_params_


# In[ ]:


xgc=gs.best_estimator_


# In[ ]:


xgc.fit(X_train,y_train)


# In[ ]:


y_test_pred = xgc.predict(X_test)


# In[ ]:


cohen_kappa_score(y_test_pred, y_test, weights='quadratic')


# In[ ]:


xgc.fit(X,y)


# In[ ]:


y_pred_xgc = xgc.predict(dftest)


# And the winner is ...

# In[ ]:


y_pred = y_pred_xgc


# ### Submission

# In[ ]:


dftest_ids.head()


# In[ ]:


subm=pd.DataFrame({'PetID': dftest_ids,'AdoptionSpeed': y_pred})
subm.tail()


# In[ ]:


subm.to_csv('submission.csv', index=False)


# ### Credits & Thanks to ...

# https://www.kaggle.com/dochad/pet-adoption-starter-kernel-tutorial
# 
# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
# 
# https://www.kaggle.com/nicapotato/text-and-structured-data-lgbm

# 
