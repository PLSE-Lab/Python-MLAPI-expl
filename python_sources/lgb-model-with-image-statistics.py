#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
train_df.info()


# Some missing values

# In[ ]:


test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
test_df.info()


# Some missing values

# In[ ]:


train_df.head()


# In[ ]:


breed_label = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
breed_label.head()


# In[ ]:


train_df['Mixed_Breed'] = train_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
test_df['Mixed_Breed'] = test_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)


# In[ ]:


color_label = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
color_label


# In[ ]:


train_df['Num_Color'] = train_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
test_df['Num_Color'] = test_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)


# In[ ]:


state_label = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
state_label


# ## Description Sentiment

# In[ ]:


train_df['Description'].fillna("", inplace=True)
test_df['Description'].fillna("", inplace=True)


# In[ ]:


train_df['Description_Length'] = train_df.Description.map(len)
test_df['Description_Length'] = test_df.Description.map(len)


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x='AdoptionSpeed', y='Description_Length', data=train_df, showfliers=False)


# In[ ]:


sentiment_list = os.listdir('../input/petfinder-adoption-prediction/train_sentiment')


# In[ ]:


sentiment_list[0]


# In[ ]:


example_text = pd.read_json('../input/petfinder-adoption-prediction/train_sentiment/{}'.format(sentiment_list[1000]), orient='index', typ='series')


# In[ ]:


example_text


# In[ ]:


def get_desc_stats(desc_json):
    example = pd.read_json('../input/petfinder-adoption-prediction/train_sentiment/{}'.format(desc_json), orient='index', typ='series')
    result = {}
    
    result['num_sentences'] = len(example.sentences)
    result['num_entities'] = len(example.entities)
    result['magnitude'] = example.documentSentiment['magnitude']
    result['score'] = example.documentSentiment['score']
    
    return result


# In[ ]:


sentiment = {}
for x in sentiment_list:
    sentiment[x[:9]] = get_desc_stats(x)


# In[ ]:


sentiment_df = pd.DataFrame.from_dict(sentiment).transpose()


# In[ ]:


train_df = train_df.join(sentiment_df, on='PetID')


# In[ ]:


train_df.magnitude.fillna(-1, inplace=True)
train_df.score.fillna(-1, inplace=True)
train_df.num_sentences.fillna(-1, inplace=True)
train_df.num_entities.fillna(-1, inplace=True)


# In[ ]:


sentiment_list_test = os.listdir('../input/petfinder-adoption-prediction/test_sentiment')
def get_desc_stats_test(desc_json):
    example = pd.read_json('../input/petfinder-adoption-prediction/test_sentiment/{}'.format(desc_json), orient='index', typ='series')
    result = {}
    
    result['num_sentences'] = len(example.sentences)
    result['num_entities'] = len(example.entities)
    result['magnitude'] = example.documentSentiment['magnitude']
    result['score'] = example.documentSentiment['score']
    
    return result

sentiment_test = {}
for x in sentiment_list_test:
    sentiment_test[x[:9]] = get_desc_stats_test(x)
sentiment_df_test = pd.DataFrame.from_dict(sentiment_test).transpose()
test_df = test_df.join(sentiment_df_test, on='PetID')
test_df.magnitude.fillna(-1, inplace=True)
test_df.score.fillna(-1, inplace=True)
test_df.num_sentences.fillna(-1, inplace=True)
test_df.num_entities.fillna(-1, inplace=True)


# ## Add Tfidf features

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


import re
pattern = re.compile('[\W_]+', re.UNICODE)
texts = [pattern.sub(' ', x) for x in train_df.Description]
texts_test = [pattern.sub(' ', x) for x in test_df.Description]


# In[ ]:


Tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3), max_features=20000,max_df=0.95,min_df=5)


# In[ ]:


train_x_tfidf_full = Tfidf.fit_transform(texts)
test_x_tfidf = Tfidf.transform(texts_test)


# In[ ]:


from sklearn.decomposition import TruncatedSVD


# In[ ]:


svd = TruncatedSVD(n_components=100, random_state=42)
train_x_svd = svd.fit_transform(train_x_tfidf_full)


# In[ ]:


train_df = train_df.join(pd.DataFrame(train_x_svd, columns=['svd_'+str(x) for x in np.arange(100)]))


# In[ ]:


test_x_svd = svd.transform(test_x_tfidf)
test_df = test_df.join(pd.DataFrame(test_x_svd, columns=['svd_'+str(x) for x in np.arange(100)]))


# ## Word Count statistics

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv = CountVectorizer(stop_words='english',ngram_range=(1,3), max_features=20000,max_df=0.95,min_df=5)


# In[ ]:


train_x_cv = cv.fit_transform(texts)
test_x_cv = cv.transform(texts_test)


# In[ ]:


train_x_cv


# In[ ]:


# create a dataframe from a word matrix
def wm2df(wm, feat_names):
    
    # create an index for each row
    
    df = pd.DataFrame(data=wm.toarray(),
                      columns=feat_names)
    return(df)


# In[ ]:


# retrieve the terms found in the corpora
tokens = cv.get_feature_names()

# create a dataframe from the matrix
wm2df(train_x_cv, tokens).head()


# In[ ]:


svd_cv = TruncatedSVD(n_components=100, random_state=42)
train_x_svd_cv = svd_cv.fit_transform(train_x_cv)
train_df = train_df.join(pd.DataFrame(train_x_svd_cv, columns=['svd_cv_'+str(x) for x in np.arange(100)]))
test_x_svd_cv = svd_cv.transform(test_x_cv)
test_df = test_df.join(pd.DataFrame(test_x_svd_cv, columns=['svd_cv_'+str(x) for x in np.arange(100)]))


# In[ ]:


train_df.head()


# ## Description Statistics

# In[ ]:


def isChinese(s):
    if len(re.findall(u'[\u4e00-\u9fff]', s)) > 0:
        return 1
    else:
        return 0


# In[ ]:


train_df['contains_chinese'] = train_df.Description.map(isChinese)
test_df['contains_chinese'] = test_df.Description.map(isChinese)


# In[ ]:


train_df.contains_chinese.value_counts()


# ## Photo Metadata

# In[ ]:


photo_list = os.listdir('../input/petfinder-adoption-prediction/train_metadata')


# In[ ]:


profile_photo = [x for x in photo_list if "-1." in x]


# In[ ]:


def get_dominant_color(photo_json):
    example = pd.read_json('../input/petfinder-adoption-prediction/train_metadata/{}'.format(photo_json), orient='index', typ='series')
    max_index = np.argmax([x['pixelFraction'] for x in example.imagePropertiesAnnotation['dominantColors']['colors']])
    result = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['color']
    result['score'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['score']
    result['pixelFraction'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['pixelFraction']
    try:
        result['image_description'] = example.labelAnnotations[0]['description']
        result['image_description_score'] = example.labelAnnotations[0]['score']
        
    except AttributeError:
        result['image_description'] = -1
        result['image_description_score'] = -1
        
    result['image_confidence'] = example.cropHintsAnnotation['cropHints'][0]['confidence']
    try:
        result['image_importanceFraction'] = example.cropHintsAnnotation['cropHints'][0]['importanceFraction']
    except KeyError:
        result['image_importanceFraction'] = -1
    return result


# In[ ]:


dominant_color_train = {}
for x in profile_photo:
    dominant_color_train[x[:9]] = get_dominant_color(x)


# In[ ]:


dominant_color_df = pd.DataFrame(dominant_color_train).transpose()
dominant_color_df.columns = ['photo_'+x for x in dominant_color_df.columns]
train_df = train_df.join(dominant_color_df, on='PetID', rsuffix='_color')


# In[ ]:


train_df[pd.DataFrame(dominant_color_df).columns.tolist()] = train_df[pd.DataFrame(dominant_color_df).columns.tolist()].fillna(-1)


# In[ ]:


photo_list_test = os.listdir('../input/petfinder-adoption-prediction/test_metadata')
profile_photo_test = [x for x in photo_list_test if "-1." in x]

def get_dominant_color_test(photo_json):
    example = pd.read_json('../input/petfinder-adoption-prediction/test_metadata/{}'.format(photo_json), orient='index', typ='series')
    max_index = np.argmax([x['pixelFraction'] for x in example.imagePropertiesAnnotation['dominantColors']['colors']])
    result = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['color']
    result['score'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['score']
    result['pixelFraction'] = example.imagePropertiesAnnotation['dominantColors']['colors'][max_index]['pixelFraction']
    try:
        result['image_description'] = example.labelAnnotations[0]['description']
        result['image_description_score'] = example.labelAnnotations[0]['score']
        
    except AttributeError:
        result['image_description'] = -1
        result['image_description_score'] = -1
        
    result['image_confidence'] = example.cropHintsAnnotation['cropHints'][0]['confidence']
    try:
        result['image_importanceFraction'] = example.cropHintsAnnotation['cropHints'][0]['importanceFraction']
    except KeyError:
        result['image_importanceFraction'] = -1
    return result


# In[ ]:


dominant_color_test = {}
for x in profile_photo_test:
    dominant_color_test[x[:9]] = get_dominant_color_test(x)


# In[ ]:


dominant_color_df_test = pd.DataFrame(dominant_color_test).transpose()
dominant_color_df_test.columns = ['photo_'+x for x in dominant_color_df_test.columns]
test_df = test_df.join(dominant_color_df_test, on='PetID', rsuffix='_color')

test_df[pd.DataFrame(dominant_color_df_test).columns.tolist()] = test_df[pd.DataFrame(dominant_color_df_test).columns.tolist()].fillna(-1)


# In[ ]:


photo_image_description = {}
for i,x in enumerate(pd.concat([train_df.photo_image_description, test_df.photo_image_description]).value_counts().index):
    photo_image_description[x] = i
    
train_df.photo_image_description = train_df.photo_image_description.map(lambda x: photo_image_description[x])
test_df.photo_image_description = test_df.photo_image_description.map(lambda x: photo_image_description[x])


# In[ ]:


dog_cat = [photo_image_description[x] for x in ['dog',
 'dog breed',
 'dog like mammal',
 'dog breed group',
 'pug',
 'bull terrier',
 'siberian husky',
 'beagle',
 'street dog',
 'boston terrier',
 'pomeranian',
 'harrier',
 'czechoslovakian wolfdog',
 'basset hound',
 'volpino italiano']]


# In[ ]:


cat_cat = [photo_image_description[x] for x in ['cat',
 'small to medium sized cats',
 'cat like mammal',
 'black cat']]


# In[ ]:


def main_category(x):
    if x in dog_cat:
        return 1
    elif x in cat_cat:
        return 2
    elif x == photo_image_description[-1]:
        return 0
    else:
        return 3


# In[ ]:


train_df['photo_category'] = train_df.photo_image_description.map(main_category)
test_df['photo_category'] = test_df.photo_image_description.map(main_category)


# ## Image Statistics

# In[ ]:


image_stat_train = pd.read_csv('../input/image-statistics-for-petfinder/train_image.csv')
image_stat_train.info()


# In[ ]:


image_stat_test = pd.read_csv('../input/image-statistics-for-petfinder/test_image.csv')
image_stat_test.info()


# In[ ]:


image_stat_train['PetID'] = image_stat_train.image.map(lambda x: x[:9])
image_stat_test['PetID'] = image_stat_test.image.map(lambda x: x[:9])


# In[ ]:


image_stat_train = image_stat_train.set_index('PetID')
image_stat_test = image_stat_test.set_index('PetID')


# In[ ]:


train_df = train_df.join(image_stat_train, on='PetID', rsuffix='_image')
test_df = test_df.join(image_stat_test, on='PetID', rsuffix='_image')


# ## LGB Model

# In[ ]:


import lightgbm as lgb


# In[ ]:


train_df.columns


# In[ ]:


features = [x for x in train_df.columns if x not in ['Name', 'Type','RescuerID','AdoptionSpeed','Description','PetID','image','dullness_whiteness','temp_size','photo_image_description']]


# In[ ]:


df_train, df_val = train_test_split(train_df, test_size=0.3, random_state=420)


# In[ ]:


df_train.columns


# In[ ]:


from sklearn.metrics import cohen_kappa_score


# In[ ]:


def kappa_scorer(pred, train_data):
    length = len(train_data.get_label())
    pred_results = [[pred[x], pred[x+length*1], pred[x+length*2], pred[x+length*3], pred[x+length*4]] for x in np.arange(length)]
    
    return 'kappa', cohen_kappa_score([np.argmax(x) for x in pred_results],train_data.get_label(), weights='quadratic'), True


# In[ ]:


d_train = lgb.Dataset(df_train[features], label=df_train['AdoptionSpeed'],feature_name=features, 
                      categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State','Mixed_Breed','photo_category','contains_chinese'])
d_val = lgb.Dataset(df_val[features], label=df_val['AdoptionSpeed'], reference=d_train,feature_name=features, 
                    categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'Mixed_Breed','photo_category','contains_chinese'])

params = {"objective" : "multiclass",
              "num_class": 5,
              "metric" : "None",
              "learning_rate" : 0.1,
              "feature_fraction_seed" : 420,          
              "feature_fraction" : 0.4,
              "early_stopping_rounds": 200
             }

evals_result = {}
model = lgb.train(params, d_train, num_boost_round=2000, valid_sets=[d_train, d_val], feval=kappa_scorer, evals_result=evals_result, verbose_eval=50)


# In[ ]:


d_train = lgb.Dataset(df_train[features], label=df_train['AdoptionSpeed'],feature_name=features, 
                      categorical_feature=['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State','Mixed_Breed','photo_category','contains_chinese'])

cv_dict = lgb.cv(params, d_train, num_boost_round=1000, feval=kappa_scorer, verbose_eval=50)


# In[ ]:


from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(random_state=1)
dummy_model.fit(df_train[features], df_train['AdoptionSpeed'])
cohen_kappa_score(dummy_model.predict(df_val[features]), df_val['AdoptionSpeed'])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,100))
lgb.plot_importance(model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel('Feature', size=12)
plt.xlabel('Importance', size=12)
plt.title("Importance of the Features of LightGBM Model", fontsize=15)
plt.show()


# ## Prediction

# In[ ]:


pred_test = model.predict(test_df[features], num_iteration=model.best_iteration)


# In[ ]:


submission = pd.concat([test_df.PetID,pd.DataFrame(pred_test, columns=['A','B','C','D','E'])],axis=1)


# In[ ]:


submission['AdoptionSpeed'] = submission.apply(lambda x: np.argmax([x.A,x.B,x.C,x.D,x.E]), axis=1)


# In[ ]:


submission = submission[['PetID','AdoptionSpeed']]
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:


submission.AdoptionSpeed.value_counts()


# In[ ]:




