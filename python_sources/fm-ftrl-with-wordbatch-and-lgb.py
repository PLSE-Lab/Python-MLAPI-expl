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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import vstack, hstack, csr_matrix
from scipy import sparse
import gc
import psutil
from nltk.corpus import stopwords
import string
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
import sys
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
from tqdm import tqdm
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform


# In[ ]:


data_train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', delimiter='\t')


# In[ ]:


####split data train test first
y = np.log10(np.array(data_train['price'])+1)
X = data_train.drop('price',axis=1)

X_train,X_cv,Y_train,Y_cv = train_test_split(X, y, test_size=0.20, random_state=42)

del(X, y ,data_train)
gc.collect()


# In[ ]:


X_train.drop('train_id', axis=1, inplace=True)
X_cv.drop('train_id', axis=1, inplace=True)


# In[ ]:


def rmsle(y, y0):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


import re
from nltk.corpus import stopwords
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])


def concat_categories(x):
    return set(x.values)

def brandfinder(name, category):    
    for brand in brands_sorted_by_size:
        if brand in name and category in brand_names_categories[brand]:
            return brand
    return 'Unknown'


# function to count repetition of first name
def create_dictionary(col_name,data_frame= X_train):
    dictionary = dict(zip(data_frame[col_name],data_frame.groupby(col_name)[col_name].transform('count')))
    return dictionary
    
def transform_col(data_frame, col_name):
    dictionary = create_dictionary(col_name)
    transformed_column = []
    for value in data_frame[col_name].values:
        transformed_column.append(dictionary.get(value,1))
    dictionary = None
    del(dictionary)
    gc.collect()
    return transformed_column

# function returns only first name(first_word)
def clean_name(x):
    if len(x):
        x = non_alphanums.sub(' ', x).split()
        if len(x):
            return x[0].lower()
    return ''

def to_number(x):
    try:
        if not x.isdigit():
            return 0
        x = int(x)
        if x > 100:
            return 100
        else:
            return x
    except:
        return 0

def sum_numbers(desc):
    if not isinstance(desc, str):
        return 0
    try:
        return sum([to_number(s) for s in desc.split()])
    except:
        return 0
    
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

def decontracted(phrase):
    # specific
    try:
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
    except:
        return 0

def cleaning_text(df):
    from tqdm import tqdm
    preprocessed_item_description = []
    # tqdm is for printing the status bar
    for sentance in tqdm(df['item_description'].values):
        sent = decontracted(str(sentance))
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_item_description.append(sent.lower().strip())
    df = df.drop('item_description',axis=1)
    df['item_description'] = preprocessed_item_description
    preprocessed_item_description = None
    del(preprocessed_item_description)
    gc.collect()
    return(df)

def fill_brand(df):
    brand_names_categories = dict(df[df['brand_name'] != 'missing'][['brand_name','category_name']].astype('str')                              .groupby('brand_name').agg(concat_categories).reset_index().values.tolist())
    brands_sorted_by_size = list(sorted(filter(lambda y: len(y) >= 3,                                            list(brand_names_categories.keys())),                                             key = lambda x: -len(x)))
    
    train_names_unknown_brands_train = df[df['brand_name'] == 'missing'][['name','category_name']].                            astype('str').values
    train_estimated_brands_train = []
    
    
    for name, category in tqdm(train_names_unknown_brands_train):
        for brand in brands_sorted_by_size:
            if brand in name and category in brand_names_categories[brand]:
                brand_name = brand
            else:
                brand_name = 'missing'
        train_estimated_brands_train.append(brand_name)
        
    df.loc[df['brand_name'] == 'missing', 'brand_name'] = train_estimated_brands_train
    
    brand_names_categories = None
    brands_sorted_by_size = None
    train_names_unknown_brands_train = None
    train_estimated_brands_train = None
    brand_name= None
    del(brand_names_categories,brands_sorted_by_size,train_names_unknown_brands_train,train_estimated_brands_train,brand_name)
    gc.collect()
    return(df)


# In[ ]:


def preprocessing(df):
    #cleaning text
    cleaning_text(df)
    print("Text_cleaning Done")
    # filling missing values with brand_name as 'missing'
    df['brand_name'] = df['brand_name'].fillna('missing')
    df['brand_name'] = df['brand_name'].astype('category')
    print("preprocessing for brand_name Done")
    
    # filling missing values with category_name as missing/missing/missing or we can simply remove these rows
    df['category_name'] = df['category_name'].fillna('missing/missing/missing')
    df['category_name'] = df['category_name'].fillna('missing/missing/missing')
    print("preprocessing for category_name Done")
    
    df['item_description'] = df['item_description'].fillna('missing')
    print("preprocessing for item_description Done")
    
    df['category_name']= df['category_name'].str.split('/')
    df['main_category'] = df['category_name'].str.get(0).replace('', 'missing').astype('category')
    df['sub_category_1'] = df['category_name'].str.get(1).replace('', 'missing').astype('category')
    df['sub_category_2'] = df['category_name'].str.get(2).replace('', 'missing').astype('category')
    print("split main category in to 3 Done")
        
    df['item_condition_id'] = df['item_condition_id'].astype('category')
    df['item_condition_id'] = df['item_condition_id'].cat.add_categories(['missing']).fillna('missing')
    print("preprocessing for item_condition_id Done")
    
    df['shipping'] = df['shipping'].astype('category')
    df['shipping'] = df['shipping'].cat.add_categories(['missing']).fillna('missing')
    print("preprocessing for shipping Done")
    
    df['item_description'].fillna('missing', inplace=True)
    print("preprocessing for item_description Done")    
    fill_brand(df)
    return(df)

def adding_new_features(df):
    df['name_first'] = df['name'].apply(clean_name)
    df['name_first_count'] = transform_col(data_frame = df, col_name = 'name_first' )
    df['main_cat_count'] = transform_col(data_frame = df, col_name = 'main_category' )
    df['sub_cat_1_count'] = transform_col(data_frame = df, col_name = 'sub_category_1' )
    df['sub_cat_2_count'] = transform_col(data_frame = df, col_name = 'sub_category_2' )
    df['brand_name_count'] = transform_col(data_frame = df, col_name = 'brand_name' )
    df['DescriptionLower'] = df.item_description.str.count('[a-z]')
    df['NameLower'] = df.name.str.count('[a-z]')
    df['NameUpper'] = df.name.str.count('[A-Z]')
    df['DescriptionUpper'] = df.item_description.str.count('[A-Z]')
    df['name_len'] = df['name'].apply(lambda x: len(x))
    df['des_len'] = df['item_description'].apply(lambda x: len(x))
    df['name_desc_len_ratio'] = df['name_len']/df['des_len']
    df['desc_word_count'] = df['item_description'].apply(lambda x: len(x.split()))
    df['mean_des'] = df['item_description'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
    df['name_word_count'] = df['name'].apply(lambda x: len(x.split()))
    df['mean_name'] = df['name'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x))  * 10
    df['desc_letters_per_word'] = df['des_len'] / df['desc_word_count']
    df['name_letters_per_word'] = df['name_len'] / df['name_word_count']
    df['NameLowerRatio'] = df['NameLower'] / df['name_len']
    df['DescriptionLowerRatio'] = df['DescriptionLower'] / df['des_len']
    df['NameUpperRatio'] = df['NameUpper'] / df['name_len']
    df['DescriptionUpperRatio'] = df['DescriptionUpper'] / df['des_len']
    df['NamePunctCount'] = df.name.str.count(RE_PUNCTUATION)
    df['DescriptionPunctCount'] = df.item_description.str.count(RE_PUNCTUATION)
    df['NamePunctCountRatio'] = df['NamePunctCount'] / df['name_word_count']
    df['DescriptionPunctCountRatio'] = df['DescriptionPunctCount'] / df['desc_word_count']
    df['NameDigitCount'] = df.name.str.count('[0-9]')
    df['DescriptionDigitCount'] = df.item_description.str.count('[0-9]')
    df['NameDigitCountRatio'] = df['NameDigitCount'] / df['name_word_count']
    df['DescriptionDigitCountRatio'] = df['DescriptionDigitCount']/df['desc_word_count']
    df['stopword_ratio_desc'] = df['item_description'].apply(lambda x: len([w for w in x.split() if w in stopwords])) / df['desc_word_count']
    df['num_sum'] = df['item_description'].apply(sum_numbers) 
    df['weird_characters_desc'] = df['item_description'].str.count(non_alphanumpunct)
    df['weird_characters_name'] = df['name'].str.count(non_alphanumpunct)
    df['prices_count'] = df['item_description'].str.count('[rm]')
    df['price_in_name'] = df['item_description'].str.contains('[rm]', regex=False).astype('int')
    #df.drop('category_name', axis=1, inplace=True)
    return(df)


# In[ ]:


preprocessing(X_train)
adding_new_features(X_train)


# In[ ]:


X_train.head(5)


# In[ ]:


#check for the NAN values
X_train.columns[X_train.isna().any()].tolist()


# In[ ]:


preprocessing(X_cv)
adding_new_features(X_cv)


# In[ ]:


X_cv.columns[X_cv.isna().any()].tolist()


# In[ ]:


total_cols = set(X_train.columns.values)

basic_cols = {'name', 'item_condition_id', 'brand_name',
  'shipping', 'item_description', 'main_category',
  'sub_category_1', 'sub_category_2', 'name_first'}

numeric_cols = total_cols - basic_cols

cols_to_normalize = numeric_cols - {'price_in_name'}

text_cols = {'name', 'item_description'}

categorical_cols = {'item_condition_id','brand_name','shipping','main_category','sub_category_1','sub_category_2','name_first'}


# # Normalizing numerical columns

# In[ ]:


from sklearn.preprocessing import Normalizer

normalizer = Normalizer(copy=False)
normalizer.fit(X_train[list(cols_to_normalize)])


# In[ ]:


import pickle
pickle.dump(normalizer, open("normalizer.pickle", "wb"),protocol=4)
normalizer = None
del(normalizer)
gc.collect()


# In[ ]:


def normalize_dataframe(df):
    with open('normalizer.pickle',mode='rb') as model_f:
        normalizer_load = pickle.load(model_f)
    df[list(cols_to_normalize)]= normalizer_load.transform(df[list(cols_to_normalize)])
    df[list({'item_condition_id','shipping'})] = df[list({'item_condition_id','shipping'})].astype('category')
    normalizer_load = None 
    del(normalizer_load)
    gc.collect()
    return df


# In[ ]:


normalize_dataframe(X_train)
normalize_dataframe(X_cv)


# # item_desc at train time

# In[ ]:


wb_desc = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb_desc.dictionary_freeze = True


# In[ ]:


wb_desc.fit(X_train['item_description'])
X_description_train_wb = wb_desc.transform(X_train['item_description'])
X_description_cv_wb = wb_desc.fit_transform(X_cv['item_description'])


# In[ ]:


mask_desc = np.where(X_description_train_wb.getnnz(axis=0) > 3)[0]


# In[ ]:


X_description_train_wb = X_description_train_wb[:, mask_desc]
X_description_cv_wb = X_description_cv_wb[:, mask_desc]


# In[ ]:


X_description_train_wb.shape
X_description_cv_wb.shape


# In[ ]:


model_desc = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=5)
model_desc.fit(X_description_train_wb, Y_train)


# In[ ]:


pred_train_1 = model_desc.predict(X_description_train_wb)
pred_cv_1 = model_desc.predict(X_description_cv_wb)
print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_1-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_1-1)))


# In[ ]:


import pickle 
pickle.dump(mask_desc, open("mask_desc.pickle", "wb"),protocol=4)
pickle.dump(wb_desc, open("wb_desc.pickle", "wb"),protocol=4)
pickle.dump(model_desc, open("model_desc.pickle", "wb"),protocol=4)


# In[ ]:


mask_desc = None
wb_desc = None
model_desc = None
del(mask_desc,wb_desc,model_desc)
gc.collect()


# # name at train time

# In[ ]:


wb_name = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb_name.dictionary_freeze = True


# In[ ]:


wb_name.fit(X_train['name'])
X_name_train_wb = wb_name.transform(X_train['name'])
X_name_cv_wb = wb_name.fit_transform(X_cv['name'])


# In[ ]:


mask_name = np.where(X_name_train_wb.getnnz(axis=0) > 3)[0]


# In[ ]:


X_name_train_wb = X_name_train_wb[:, mask_name]
X_name_cv_wb = X_name_cv_wb[:, mask_name]


# In[ ]:


model_name = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha= 5)
model_name.fit(X_name_train_wb, Y_train)


# In[ ]:


pred_train_2 = model_name.predict(X_name_train_wb)
pred_cv_2 = model_name.predict(X_name_cv_wb)


# In[ ]:


print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_2-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_2-1)))


# In[ ]:


import pickle

pickle.dump(mask_name, open("mask_name.pickle", "wb"),protocol=4)
pickle.dump(wb_name, open("wb_name.pickle", "wb"),protocol=4)
pickle.dump(model_name, open("model_name.pickle", "wb"),protocol=4)


# In[ ]:


mask_name = None
wb_name = None
model_name = None
del(mask_name,wb_name,model_name)
gc.collect()


# # LB training

# In[ ]:


lb_brand_name = LabelBinarizer(sparse_output=True)

X_brand_train = lb_brand_name.fit_transform(X_train['brand_name'])

X_brand_cv = lb_brand_name.transform(X_cv['brand_name'])


# In[ ]:


lb_main_category = LabelBinarizer(sparse_output=True)

X_main_cat_train = lb_main_category.fit_transform(X_train['main_category'])

X_main_cat_cv = lb_main_category.transform(X_cv['main_category'])


# In[ ]:


lb_sub_category_1 = LabelBinarizer(sparse_output=True)

X_main_sub_cat_1_train = lb_sub_category_1.fit_transform(X_train['sub_category_1'])

X_main_sub_cat_1_cv = lb_sub_category_1.transform(X_cv['sub_category_1'])


# In[ ]:


lb_sub_category_2 = LabelBinarizer(sparse_output=True)

X_main_sub_cat_2_train = lb_sub_category_2.fit_transform(X_train['sub_category_2'])

X_main_sub_cat_2_cv = lb_sub_category_2.transform(X_cv['sub_category_2'])


# In[ ]:


X_dummies_train = csr_matrix(
    pd.get_dummies(X_train[list(total_cols - (basic_cols))],
                   sparse=True).values)

X_dummies_train_1 = csr_matrix(
    pd.get_dummies(X_train[list({'item_condition_id', 'shipping'})],
                   sparse=True).values)


# In[ ]:


X_dummies_cv = csr_matrix(
    pd.get_dummies(X_cv[list(total_cols - (basic_cols))],
                   sparse=True).values)

X_dummies_cv_1 = csr_matrix(
    pd.get_dummies(X_cv[list({'item_condition_id', 'shipping'})],
                   sparse=True).values)


# In[ ]:


sparse_merge_train = hstack((X_name_train_wb , X_description_train_wb, X_brand_train, X_main_cat_train,
                             X_main_sub_cat_1_train, X_main_sub_cat_2_train,X_dummies_train,X_dummies_train_1)).tocsr()


# In[ ]:


sparse_merge_cv = hstack(( X_name_cv_wb,X_description_cv_wb,X_brand_cv,X_main_cat_cv,
                             X_main_sub_cat_1_cv,X_main_sub_cat_2_cv,X_dummies_cv,X_dummies_cv_1)).tocsr()


# In[ ]:


X_dummies_train = None
X_description_train_wb = None
X_name_train_wb = None
X_dummies_cv = None
X_description_cv_wb = None
X_name_cv_wb = None
del(X_dummies_train, X_description_train_wb,X_name_train_wb)
del(X_dummies_cv, X_description_cv_wb,X_name_cv_wb)

#del(X_dummies_train, X_description_train, X_brand_train, X_main_cat_train,\
#                             X_main_sub_cat_1_train, X_main_sub_cat_2_train, X_name_train)
#del(X_dummies_cv, X_description_cv, X_brand_cv, X_main_cat_cv,\
#                             X_main_sub_cat_1_cv, X_main_sub_cat_2_cv, X_name_cv)
gc.collect()


# In[ ]:


print(sparse_merge_train.shape)
print(sparse_merge_cv.shape)


# In[ ]:


import pickle

pickle.dump(lb_brand_name, open("lb_brand_name.pickle", "wb"),protocol=4)
pickle.dump(lb_main_category, open("lb_main_category.pickle", "wb"),protocol=4)
pickle.dump(lb_sub_category_1, open("lb_sub_category_1.pickle", "wb"),protocol=4)
pickle.dump(lb_sub_category_2, open("lb_sub_category_2.pickle", "wb"),protocol=4)


# In[ ]:


lb_brand_name = None
lb_main_category = None
lb_sub_category_1 = None
lb_sub_category_2 = None
del(lb_brand_name,lb_main_category,lb_sub_category_1,lb_sub_category_2)
gc.collect()


# # FMFTRL training

# In[ ]:


model_FM_FTRL = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=sparse_merge_train.shape[1],
                alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,
                D_fm=100, e_noise=0, iters=1, inv_link="identity", threads=4)
model_FM_FTRL.fit(sparse_merge_train, Y_train)


# In[ ]:


pred_train_3 = model_FM_FTRL.predict(sparse_merge_train)
pred_cv_3 = model_FM_FTRL.predict(sparse_merge_cv)


# In[ ]:


print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_3-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_3-1)))


# In[ ]:


import pickle

pickle.dump(model_FM_FTRL, open("model_FM_FTRL.pickle", "wb"),protocol=4)
model_FM_FTRL = None
del(model_FM_FTRL)
gc.collect()


# In[ ]:


print(sparse_merge_train.shape)
print(sparse_merge_cv.shape)


# # KBestSelect Train model

# In[ ]:


fselect = SelectKBest(f_regression, k=48000)
train_kbest_features = fselect.fit_transform(sparse_merge_train, Y_train)
cv_kbest_features = fselect.transform(sparse_merge_cv)


# In[ ]:


import pickle
pickle.dump(fselect, open("fselect.pickle", "wb"),protocol=4)
fselect = None
del(fselect)
gc.collect()


# In[ ]:


cv_kbest_features.shape


# In[ ]:


sparse_merge_train= None
sparse_merge_cv = None
del(sparse_merge_train,sparse_merge_cv)
gc.collect()


# # TFIDF Desc Train

# In[ ]:


tfidf_desc = TfidfVectorizer(max_features=500000,
                     ngram_range=(1, 3),
                     stop_words=None)
X_desc_train_tfidf = tfidf_desc.fit_transform(X_train['item_description'])


# In[ ]:


X_desc_cv_tfidf = tfidf_desc.transform(X_cv['item_description'])


# In[ ]:


pickle.dump(tfidf_desc, open("tfidf_desc.pickle", "wb"),protocol=4)
tfidf_desc = None
del(tfidf_desc)
gc.collect()


# # TFIDF Name Train

# In[ ]:


tfidf_name = TfidfVectorizer(max_features=250000,
                     ngram_range=(1, 3),
                     stop_words=None)
X_name_train_tfidf = tfidf_name.fit_transform(X_train['name'])


# In[ ]:


X_name_cv_tfidf =tfidf_name.transform(X_cv['name'])


# In[ ]:


pickle.dump(tfidf_name, open("tfidf_name.pickle", "wb"),protocol=4)
tfidf_name = None
del(tfidf_name)
gc.collect()


# # hstack features set_2

# In[ ]:


sparse_merge_train_1 = hstack((X_name_train_tfidf , X_desc_train_tfidf, X_brand_train, X_main_cat_train,
                             X_main_sub_cat_1_train, X_main_sub_cat_2_train,X_dummies_train_1)).tocsr()


# In[ ]:


X_dummies_train_1 = None
X_brand_train = None
X_main_cat_train = None
X_main_sub_cat_1_train = None
X_main_sub_cat_2_train = None
X_name_train_tfidf = None
X_desc_train_tfidf = None
del(X_dummies_train_1, X_brand_train, X_main_cat_train,                            X_main_sub_cat_1_train, X_main_sub_cat_2_train, X_name_train_tfidf,X_desc_train_tfidf)
gc.collect()


# In[ ]:


sparse_merge_cv_1 = hstack((X_name_cv_tfidf , X_desc_cv_tfidf, X_brand_cv, X_main_cat_cv,
                             X_main_sub_cat_1_cv, X_main_sub_cat_2_cv,X_dummies_cv_1)).tocsr()


# In[ ]:


X_dummies_cv_1 = None
X_brand_cv = None
X_main_cat_cv = None
X_main_sub_cat_1_cv = None
X_main_sub_cat_2_cv = None
X_name_cv_tfidf = None
X_desc_cv_tfidf = None
del(X_dummies_cv_1, X_brand_cv, X_main_cat_cv,                            X_main_sub_cat_1_cv, X_main_sub_cat_2_cv, X_name_cv_tfidf,X_desc_cv_tfidf)
gc.collect()


# # Ridge on Feature set-2

# In[ ]:


model_Ridge_set_2 = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=5)
model_Ridge_set_2.fit(sparse_merge_train_1, Y_train)


# In[ ]:


pred_train_4 = model_Ridge_set_2.predict(sparse_merge_train_1)
pred_cv_4 =  model_Ridge_set_2.predict(sparse_merge_cv_1)
print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_4-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_4-1)))


# In[ ]:


sparse_merge_train_1.shape


# In[ ]:


pickle.dump(model_Ridge_set_2, open("model_Ridge_set_2.pickle", "wb"),protocol=4)
model_Ridge_set_2 = None
del(model_Ridge_set_2)
gc.collect()


# # MultinomialNB on Feature set-2

# In[ ]:


model_MNB_set_2 = MultinomialNB(alpha=1.0, fit_prior=True)
model_MNB_set_2.fit(sparse_merge_train_1, Y_train >= 4)


# In[ ]:


pred_train_5 = model_MNB_set_2.predict(sparse_merge_train_1)
pred_cv_5 =  model_MNB_set_2.predict(sparse_merge_cv_1)
print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_5-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_5-1)))


# In[ ]:


pickle.dump(model_MNB_set_2, open("model_MNB_set_2.pickle", "wb"),protocol=4)
model_MNB_set_2 = None
sparse_merge_train_1 = None
sparse_merge_cv_1 = None
del(model_MNB_set_2)
del(sparse_merge_train_1,sparse_merge_cv_1)
gc.collect()


# # Adding prediction to data frame

# # Target Encoding

# In[ ]:


f_cats = ['brand_name', 'main_category', 'sub_category_1', 'sub_category_2', 'name_first']


# In[ ]:


from category_encoders.target_encoder import TargetEncoder

targetencoder = TargetEncoder(min_samples_leaf=100, smoothing=10,cols=f_cats,return_df=False)


# In[ ]:


targetencoder.fit(X_train[f_cats],Y_train)


# In[ ]:


X_train_target_encode = targetencoder.transform(X_train[f_cats])


# In[ ]:


X_cv_target_encode = targetencoder.transform(X_cv[f_cats])


# In[ ]:


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


# In[ ]:


X_train_target_encode[:,0].shape


# In[ ]:


X_train_target_encode[:,0] = add_noise(X_train_target_encode[:,0],noise_level=0.01)
X_train_target_encode[:,1] = add_noise(X_train_target_encode[:,1],noise_level=0.01)
X_train_target_encode[:,2] = add_noise(X_train_target_encode[:,2],noise_level=0.01)
X_train_target_encode[:,3] = add_noise(X_train_target_encode[:,3],noise_level=0.01)
X_train_target_encode[:,4] = add_noise(X_train_target_encode[:,4],noise_level=0.01)


# In[ ]:


X_train_target_encode[0,:]


# In[ ]:


pickle.dump(targetencoder, open("targetencoder.pickle", "wb"),protocol=4)
targetencoder = None
del(targetencoder)
gc.collect()


# # Stacking all features 

# In[ ]:


train_features = hstack((pred_train_1.reshape(-1,1),                         pred_train_2.reshape(-1,1),                         pred_train_3.reshape(-1,1),                         pred_train_4.reshape(-1,1),                         pred_train_5.reshape(-1,1),                         X_train_target_encode,                         train_kbest_features)).tocsr()


# In[ ]:


cv_features = hstack((pred_cv_1.reshape(-1,1),                      pred_cv_2.reshape(-1,1),                      pred_cv_3.reshape(-1,1),                      pred_cv_4.reshape(-1,1),                      pred_cv_5.reshape(-1,1),                      X_cv_target_encode,                      cv_kbest_features)).tocsr()


# In[ ]:


pickle.dump(train_features, open("train_features.pickle", "wb"),protocol=4)
pickle.dump(cv_features, open("cv_features.pickle", "wb"),protocol=4)


# In[ ]:


pred_train_1 = None
pred_train_2 = None
pred_train_3 = None
pred_train_4 = None
pred_train_5 = None
X_train_target_encode = None
train_kbest_features = None

pred_cv_1 = None
pred_cv_2 = None
pred_cv_3 = None
pred_cv_4 = None
pred_cv_5 = None
X_cv_target_encode = None
cv_kbest_features = None


del(pred_train_1,    pred_train_2,    pred_train_3,    pred_train_4,    pred_train_5,    X_train_target_encode,    train_kbest_features)
del(pred_cv_1,    pred_cv_2,    pred_cv_3,    pred_cv_4,    pred_cv_5,    X_cv_target_encode,    cv_kbest_features)
gc.collect()


# # Final LGB model

# In[ ]:


d_train = lgb.Dataset(train_features, label=Y_train)
d_valid = lgb.Dataset(cv_features, label=Y_cv)
watchlist = [d_train, d_valid]


# In[ ]:


params = {
         'colsample_bytree': 0.42799939792816927,
          'max_depth': 8,
          'min_child_samples': 370,
          'min_child_weight': 0.01,
          'num_leaves': 29,
          'reg_lambda': 5,
          'subsample': 0.6739316550896339,
          'learning_rate':0.1,
          'reg_alpha' :0.5,
          'boosting_type': 'gbdt',
          'objective' : 'regression',
          'metric' : 'RMSE',
          'verbosity': -1,
          'lambda_l1': 10,
         'lambda_l2': 10
         }


# In[ ]:


model_lgb_final = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=3000,
                  valid_sets=watchlist,
                  verbose_eval=200,early_stopping_rounds=100)


# In[ ]:


pred_train_6 = model_lgb_final.predict(train_features)
pred_cv_6 = model_lgb_final.predict(cv_features)


# In[ ]:


print("Train rmsle: "+str(rmsle(10 ** Y_train-1, 10 ** pred_train_6-1)))
print("CV rmsle: "+str(rmsle(10 ** Y_cv-1, 10 ** pred_cv_6-1)))


# In[ ]:


pickle.dump(model_lgb_final, open("model_lgb_final.pickle", "wb"),protocol=4)
model_lgb_final = None
del(model_lgb_final)
gc.collect()


# In[ ]:


train_features= None
cv_features = None
d_train = None
d_valid = None
watchlist = None
del(train_features,cv_features,d_train,d_valid,watchlist)

gc.collect()


# In[ ]:


X_train = None
X_cv = None
Y_train = None
Y_test = None
del(X_train,X_cv,Y_train,Y_cv)
gc.collect()


# # Prediction on test_stg2

# In[ ]:


test_id = []
prediction = []


# In[ ]:


def predict_final(df):
    test_id.extend(list(df['test_id']))
    df.drop('test_id', axis=1, inplace=True)
    preprocessing(df)
    adding_new_features(df)
    normalize_dataframe(df)
    
######### Loading wordbag_desc_model & model_1 pred ###########    
    with open('mask_desc.pickle',mode='rb') as model_f:
        mask_desc = pickle.load(model_f)
    with open('wb_desc.pickle',mode='rb') as model_f:
        wb_desc= pickle.load(model_f)
    with open('model_desc.pickle',mode='rb') as model_f:
        model_desc = pickle.load(model_f)
    X_description_test_wb = wb_desc.transform(df['item_description'])
    X_description_test_wb = X_description_test_wb[:, mask_desc]
    print(X_description_test_wb.shape)
    
    pred_test_1 = model_desc.predict(X_description_test_wb)
    mask_desc = None
    wb_desc = None
    model_desc = None
    del(mask_desc,wb_desc,model_desc)
    gc.collect()

######### Loading wordbag_name_model & model_2 pred ###########    
    with open('mask_name.pickle',mode='rb') as model_f:
        mask_name = pickle.load(model_f)
    with open('wb_name.pickle',mode='rb') as model_f:
        wb_name= pickle.load(model_f)
    with open('model_name.pickle',mode='rb') as model_f:
        model_name = pickle.load(model_f)
    X_name_test_wb = wb_name.transform(df['name'])
    X_name_test_wb = X_name_test_wb[:, mask_name]
    
    print(X_name_test_wb.shape)
    pred_test_2 = model_name.predict(X_name_test_wb)
    mask_name = None
    wb_name = None
    model_name = None
    del(mask_name,wb_name,model_name)
    gc.collect()

###################### Lb and Dummies ##########################
    with open('lb_brand_name.pickle',mode='rb') as model_f:
        lb_brand_name = pickle.load(model_f)
    with open('lb_main_category.pickle',mode='rb') as model_f:
        lb_main_category = pickle.load(model_f)
    with open('lb_sub_category_1.pickle',mode='rb') as model_f:
        lb_sub_category_1 = pickle.load(model_f)
    with open('lb_sub_category_2.pickle',mode='rb') as model_f:
        lb_sub_category_2 = pickle.load(model_f)
        
    X_brand_test = lb_brand_name.transform(df['brand_name'])
    X_main_cat_test = lb_main_category.transform(df['main_category'])
    X_main_sub_cat_1_test = lb_sub_category_1.transform(df['sub_category_1'])
    X_main_sub_cat_2_test = lb_sub_category_2.transform(df['sub_category_2'])
    
    X_dummies_test = csr_matrix(
        pd.get_dummies(df[list(total_cols - (basic_cols))],
                   sparse=True).values)

    X_dummies_test_1 = csr_matrix(
        pd.get_dummies(df[list({'item_condition_id', 'shipping'})],
                   sparse=True).values)
    
##################### sparse matrix feature_set_1 #############################
    sparse_merge_test = hstack((X_name_test_wb , X_description_test_wb, X_brand_test, X_main_cat_test,
                             X_main_sub_cat_1_test, X_main_sub_cat_2_test,X_dummies_test,X_dummies_test_1)).tocsr()
    X_dummies_test = None 
    X_description_test_wb = None
    X_name_test_wb = None
    del(X_dummies_test, X_description_test_wb,X_name_test_wb)
    gc.collect()
    print(sparse_merge_test.shape)

############################### FMFTRL ###################################
    with open('model_FM_FTRL.pickle',mode='rb') as model_f:
        model_FM_FTRL = pickle.load(model_f)
    pred_test_3 = model_FM_FTRL.predict(sparse_merge_test)
    model_FM_FTRL = None
    del(model_FM_FTRL)
    gc.collect()
######################### Kbest Select ##################################
    with open('fselect.pickle',mode='rb') as model_f:
        fselect = pickle.load(model_f)
    test_kbest_features = fselect.transform(sparse_merge_test)
    fselect = None
    print(test_kbest_features.shape)
    del(fselect)
    gc.collect()
    sparse_merge_train= None
    sparse_merge_cv = None
    del(sparse_merge_train,sparse_merge_cv)
    gc.collect()
    
########################## TFIDF Desc Train ############################
    with open('tfidf_desc.pickle',mode='rb') as model_f:
        tfidf_desc = pickle.load(model_f)
    X_desc_test_tfidf = tfidf_desc.transform(df['item_description'])
    tfidf_desc = None
    del(tfidf_desc)
    gc.collect()
    
########################## TFIDF Name Train ############################
    with open('tfidf_name.pickle',mode='rb') as model_f:
        tfidf_name = pickle.load(model_f)
    X_name_test_tfidf = tfidf_name.transform(df['name'])
    tfidf_name = None
    del(tfidf_name)
    gc.collect()

##################### sparse matrix feature_set_2 #############################
    sparse_merge_test_1 = hstack((X_name_test_tfidf , X_desc_test_tfidf, X_brand_test, X_main_cat_test,
                             X_main_sub_cat_1_test, X_main_sub_cat_2_test,X_dummies_test_1)).tocsr()
    X_dummies_test_1 = None
    X_brand_test = None
    X_main_cat_test = None
    X_main_sub_cat_1_test = None
    X_main_sub_cat_2_test = None
    X_name_test_tfidf = None
    X_desc_test_tfidf = None
    del(X_dummies_test_1, X_brand_test, X_main_cat_test,                            X_main_sub_cat_1_test, X_main_sub_cat_2_test, X_name_test_tfidf,X_desc_test_tfidf)
    gc.collect()

######################### Ridge model on feature_set_2 #############################
    with open('model_Ridge_set_2.pickle',mode='rb') as model_f:
        model_Ridge_set_2 = pickle.load(model_f)
    pred_test_4 = model_Ridge_set_2.predict(sparse_merge_test_1)
    model_Ridge_set_2 = None
    del(model_Ridge_set_2)
    gc.collect()

########################## MNB model on feature_set_2 ############################
    with open('model_MNB_set_2.pickle',mode='rb') as model_f:
        model_MNB_set_2 = pickle.load(model_f)
    pred_test_5 = model_MNB_set_2.predict(sparse_merge_test_1)
    model_MNB_set_2 = None
    sparse_merge_train_1 = None
    sparse_merge_cv_1 = None
    del(model_MNB_set_2)
    del(sparse_merge_train_1,sparse_merge_cv_1)
    gc.collect()

############################ target_encoding ##################################
    with open('targetencoder.pickle',mode='rb') as model_f:
        targetencoder = pickle.load(model_f)
    f_cats = ['brand_name', 'main_category', 'sub_category_1', 'sub_category_2', 'name_first']
    X_test_target_encode = targetencoder.transform(df[f_cats])
    targetencoder = None
    del(targetencoder)
    gc.collect()
    
######################### Final sparse matrix #################################

    test_features = hstack((pred_test_1.reshape(-1,1),                         pred_test_2.reshape(-1,1),                         pred_test_3.reshape(-1,1),                         pred_test_4.reshape(-1,1),                         pred_test_5.reshape(-1,1),                         X_test_target_encode,                         test_kbest_features)).tocsr()
    
    pred_test_1 = None
    pred_test_2 = None
    pred_test_3 = None
    pred_test_4 = None
    pred_test_5 = None
    X_test_target_encode = None
    test_kbest_features = None


    del(pred_test_1,        pred_test_2,        pred_test_3,        pred_test_4,        pred_test_5,        X_test_target_encode,        test_kbest_features)
    gc.collect()

########################## Final lgbm prediction ##################
    with open('model_lgb_final.pickle',mode='rb') as model_f:
        model_lgb_final = pickle.load(model_f)
    #pred_test_6 = model_lgb_final.predict(test_features)
    prediction.extend(list(model_lgb_final.predict(test_features)))
    #prediction = prediction + list(model_lgb_final.predict(test_features))
########################## del everything ######################
    train_features = None
    cv_features = None 
    d_train = None
    d_valid = None
    watchlist = None
    del(train_features,cv_features,d_train,d_valid,watchlist)
    gc.collect()


# In[ ]:


chunksize = 10 ** 6
for chunk in pd.read_csv('../input/mercari-price-suggestion-challenge/test_stg2.tsv', delimiter='\t',chunksize=chunksize):
    predict_final(chunk)


# In[ ]:


submission = pd.DataFrame()
submission['test_id'] = np.asarray(test_id)
submission['price'] = (10 ** np.asarray(prediction) - 1)
submission.to_csv('stacked_submission_1.csv', index=False)

