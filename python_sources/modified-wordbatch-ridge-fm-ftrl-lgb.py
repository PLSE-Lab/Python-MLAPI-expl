#!/usr/bin/env python
# coding: utf-8

# This script is a fork of [some great initial work by Samrat](https://www.kaggle.com/samratp/wordbatch-ridge-fm-frtl-target-encoding-lgbm), orginally based on [my solution that got 18th place in the Mercari competition](https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604/code).
# 
# From his original solution, I...
# 
# * Fixed the normalization to avoid removing Russian characters.
# * Removed some variables I thought were overfitting and not providing value.
# * Reduced the dimensions of the sparse matricies for easier fitting.
# * Added standard scaling for numeric data.
# * Simplified the logic for employing one hot encoding.
# * Tuned the individual models a little bit, but not much. (More tuning is left as an exercise for the reader.)
# 
# Other things I'd suggest if improving this script:
# * Integrate data from train_active / test_active / periods_train / periods_test
# * Integrate image data
# * Test built-in LGB encoding for categoricals against TargetEncoding (or try both together)
# 
# Hope this helps. Good luck out there.

# In[1]:


get_ipython().run_cell_magic('time', '', '# Based on this wonderful notebook by Peter - https://www.kaggle.com/peterhurford/lgb-and-fm-18th-place-0-40604\nimport time\nstart_time = time.time()\n\n# Use SUBMIT_MODE = False to tune your script!\n# Use SUBMIT_MODE = True to generate a submission for Kaggle.\nSUBMIT_MODE = True\n\nimport pandas as pd\nimport numpy as np\nimport time\nimport gc\nimport string\nimport re\n\nfrom nltk.corpus import stopwords\n\nfrom scipy.sparse import csr_matrix, hstack\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.feature_selection.univariate_selection import SelectKBest, f_regression\nfrom sklearn.preprocessing import LabelBinarizer\n\nimport wordbatch\nfrom wordbatch.extractors import WordBag\nfrom wordbatch.models import FM_FTRL\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import Ridge\nfrom sklearn.naive_bayes import MultinomialNB\nimport lightgbm as lgb\n\n# Viz\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndef rmse(predicted, actual):\n    return np.sqrt(((predicted - actual) ** 2).mean())')


# In[2]:


get_ipython().run_cell_magic('time', '', '\nclass TargetEncoder:\n    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features\n    def __repr__(self):\n        return \'TargetEncoder\'\n\n    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):\n        self.cols = cols\n        self.smoothing = smoothing\n        self.min_samples_leaf = min_samples_leaf\n        self.noise_level = noise_level\n        self.keep_original = keep_original\n\n    @staticmethod\n    def add_noise(series, noise_level):\n        return series * (1 + noise_level * np.random.randn(len(series)))\n\n    def encode(self, train, test, target):\n        for col in self.cols:\n            if self.keep_original:\n                train[col + \'_te\'], test[col + \'_te\'] = self.encode_column(train[col], test[col], target)\n            else:\n                train[col], test[col] = self.encode_column(train[col], test[col], target)\n        return train, test\n\n    def encode_column(self, trn_series, tst_series, target):\n        temp = pd.concat([trn_series, target], axis=1)\n        # Compute target mean\n        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])\n        # Compute smoothing\n        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))\n        # Apply average function to all target data\n        prior = target.mean()\n        # The bigger the count the less full_avg is taken into account\n        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing\n        averages.drop([\'mean\', \'count\'], axis=1, inplace=True)\n        # Apply averages to trn and tst series\n        ft_trn_series = pd.merge(\n            trn_series.to_frame(trn_series.name),\n            averages.reset_index().rename(columns={\'index\': target.name, target.name: \'average\'}),\n            on=trn_series.name,\n            how=\'left\')[\'average\'].rename(trn_series.name + \'_mean\').fillna(prior)\n        # pd.merge does not keep the index so restore it\n        ft_trn_series.index = trn_series.index\n        ft_tst_series = pd.merge(\n            tst_series.to_frame(tst_series.name),\n            averages.reset_index().rename(columns={\'index\': target.name, target.name: \'average\'}),\n            on=tst_series.name,\n            how=\'left\')[\'average\'].rename(trn_series.name + \'_mean\').fillna(prior)\n        # pd.merge does not keep the index so restore it\n        ft_tst_series.index = tst_series.index\n        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)')


# In[3]:


get_ipython().run_cell_magic('time', '', '# Define helpers for text normalization\nstopwords = {x: 1 for x in stopwords.words(\'russian\')}\nnon_alphanums = re.compile(u\'[^A-Za-z0-9]+\')\nnon_alphanumpunct = re.compile(u\'[^A-Za-z0-9\\.?!,; \\(\\)\\[\\]\\\'\\"\\$]+\')\nRE_PUNCTUATION = \'|\'.join([re.escape(x) for x in string.punctuation])')


# In[4]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'../input/train.csv\', index_col = "item_id", parse_dates = ["activation_date"])\ntest = pd.read_csv(\'../input/test.csv\', index_col = "item_id", parse_dates = ["activation_date"])\nprint(\'[{}] Finished load data\'.format(time.time() - start_time))')


# In[5]:


get_ipython().run_cell_magic('time', '', "import string\n\ndef normalize_text(text):\n    text = text.lower().strip()\n    for s in string.punctuation:\n        text = text.replace(s, ' ')\n    text = text.strip().split(' ')\n    return u' '.join(x for x in text if len(x) > 1 and x not in stopwords)\n\nprint(train.description[0])\nprint(normalize_text(train.description[0]))")


# In[6]:


get_ipython().run_cell_magic('time', '', "train['is_train'] = 1\ntest['is_train'] = 0\nprint('[{}] Compiled train / test'.format(time.time() - start_time))\nprint('Train shape: ', train.shape)\nprint('Test shape: ', test.shape)\n\ny = train.deal_probability.copy()\nnrow_train = train.shape[0]\n\nmerge = pd.concat([train, test])\nsubmission = pd.DataFrame(test.index)\nprint('[{}] Compiled merge'.format(time.time() - start_time))\nprint('Merge shape: ', merge.shape)\n\ndel train\ndel test\ngc.collect()\nprint('[{}] Garbage collection'.format(time.time() - start_time))")


# In[7]:


get_ipython().run_cell_magic('time', '', 'print("Feature Engineering - Part 1")\nmerge["price"] = np.log(merge["price"]+0.001)\nmerge["price"].fillna(-999,inplace=True)\nmerge["image_top_1"].fillna(-999,inplace=True)\n\nprint("\\nCreate Time Variables")\nmerge["activation_weekday"] = merge[\'activation_date\'].dt.weekday\nprint(merge.head(5))\ngc.collect()')


# In[8]:


get_ipython().run_cell_magic('time', '', '# Create Validation Index and Remove Dead Variables\ntraining_index = merge.loc[merge.activation_date<=pd.to_datetime(\'2017-04-07\')].index\nvalidation_index = merge.loc[merge.activation_date>=pd.to_datetime(\'2017-04-08\')].index\nmerge.drop(["activation_date","image"],axis=1,inplace=True)\n\n#Drop user_id\nmerge.drop(["user_id"], axis=1,inplace=True)')


# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Meta Text Features\nprint("\\nText Features")\ntextfeats = ["description", "title"]\n\nfor cols in textfeats:\n    merge[cols] = merge[cols].astype(str) \n    merge[cols] = merge[cols].astype(str).fillna(\'missing\') # FILL NA\n    merge[cols] = merge[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently\n    merge[cols + \'_num_stopwords\'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords])) # Count number of Stopwords\n    merge[cols + \'_num_punctuations\'] = merge[cols].apply(lambda comment: (comment.count(RE_PUNCTUATION))) # Count number of Punctuations\n    merge[cols + \'_num_alphabets\'] = merge[cols].apply(lambda comment: len([c for c in comment if c.isupper()])) # Count number of Alphabets\n    merge[cols + \'_num_digits\'] = merge[cols].apply(lambda comment: (comment.count(\'[0-9]\'))) # Count number of Digits\n    merge[cols + \'_num_letters\'] = merge[cols].apply(lambda comment: len(comment)) # Count number of Letters\n    merge[cols + \'_num_words\'] = merge[cols].apply(lambda comment: len(comment.split())) # Count number of Words\n    merge[cols + \'_num_unique_words\'] = merge[cols].apply(lambda comment: len(set(w for w in comment.split())))\n    merge[cols + \'_words_vs_unique\'] = merge[cols+\'_num_unique_words\'] / merge[cols+\'_num_words\'] # Count Unique Words\n    merge[cols + \'_letters_per_word\'] = merge[cols+\'_num_letters\'] / merge[cols+\'_num_words\'] # Letters per Word\n    merge[cols + \'_punctuations_by_letters\'] = merge[cols+\'_num_punctuations\'] / merge[cols+\'_num_letters\'] # Punctuations by Letters\n    merge[cols + \'_punctuations_by_words\'] = merge[cols+\'_num_punctuations\'] / merge[cols+\'_num_words\'] # Punctuations by Words\n    merge[cols + \'_digits_by_letters\'] = merge[cols+\'_num_digits\'] / merge[cols+\'_num_letters\'] # Digits by Letters\n    merge[cols + \'_alphabets_by_letters\'] = merge[cols+\'_num_alphabets\'] / merge[cols+\'_num_letters\'] # Alphabets by Letters\n    merge[cols + \'_stopwords_by_words\'] = merge[cols+\'_num_stopwords\'] / merge[cols+\'_num_words\'] # Stopwords by Letters\n    merge[cols + \'_mean\'] = merge[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10 # Mean\n\n# Extra Feature Engineering\nmerge[\'title_desc_len_ratio\'] = merge[\'title_num_letters\']/merge[\'description_num_letters\']')


# In[10]:


merge.head()


# In[11]:


get_ipython().run_cell_magic('time', '', "df_test = merge.loc[merge['is_train'] == 0]\ndf_train = merge.loc[merge['is_train'] == 1]\ndel merge\ngc.collect()\ndf_test = df_test.drop(['is_train'], axis=1)\ndf_train = df_train.drop(['is_train'], axis=1)\n\nprint(df_train.shape)\nprint(y.shape)\n\nif SUBMIT_MODE:\n    y_train = y\n    del y\n    gc.collect()\nelse:\n    df_train, df_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=144)\n\nprint('[{}] Splitting completed.'.format(time.time() - start_time))")


# In[12]:


get_ipython().run_cell_magic('time', '', 'wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,\n                                                              "hash_ngrams_weights": [1.5, 1.0],\n                                                              "hash_size": 2 ** 29,\n                                                              "norm": None,\n                                                              "tf": \'binary\',\n                                                              "idf": None,\n                                                              }), procs=8)\nwb.dictionary_freeze = True\nX_name_train = wb.fit_transform(df_train[\'title\'])\nprint(X_name_train.shape)\nX_name_test = wb.transform(df_test[\'title\'])\nprint(X_name_test.shape)\ndel(wb)\ngc.collect()')


# In[13]:


get_ipython().run_cell_magic('time', '', "mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]\nX_name_train = X_name_train[:, mask]\nprint(X_name_train.shape)\nX_name_test = X_name_test[:, mask]\nprint(X_name_test.shape)\nprint('[{}] Vectorize `title` completed.'.format(time.time() - start_time))")


# In[14]:


get_ipython().run_cell_magic('time', '', 'X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train, y_train,\n                                                              test_size = 0.5,\n                                                              shuffle = False)\nprint(\'[{}] Finished splitting\'.format(time.time() - start_time))\n\nmodel = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)\nmodel.fit(X_train_1, y_train_1)\nprint(\'[{}] Finished to train name ridge (1)\'.format(time.time() - start_time))\nname_ridge_preds1 = model.predict(X_train_2)\nname_ridge_preds1f = model.predict(X_name_test)\nprint(\'[{}] Finished to predict name ridge (1)\'.format(time.time() - start_time))\nmodel = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)\nmodel.fit(X_train_2, y_train_2)\nprint(\'[{}] Finished to train name ridge (2)\'.format(time.time() - start_time))\nname_ridge_preds2 = model.predict(X_train_1)\nname_ridge_preds2f = model.predict(X_name_test)\nprint(\'[{}] Finished to predict name ridge (2)\'.format(time.time() - start_time))\nname_ridge_preds_oof = np.concatenate((name_ridge_preds2, name_ridge_preds1), axis=0)\nname_ridge_preds_test = (name_ridge_preds1f + name_ridge_preds2f) / 2.0\nprint(\'RMSLE OOF: {}\'.format(rmse(name_ridge_preds_oof, y_train)))\nif not SUBMIT_MODE:\n    print(\'RMSLE TEST: {}\'.format(rmse(name_ridge_preds_test, y_test)))\ngc.collect()')


# In[15]:


get_ipython().run_cell_magic('time', '', 'wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,\n                                                              "hash_ngrams_weights": [1.0, 1.0],\n                                                              "hash_size": 2 ** 28,\n                                                              "norm": "l2",\n                                                              "tf": 1.0,\n                                                              "idf": None}), procs=8)\nwb.dictionary_freeze = True\nX_description_train = wb.fit_transform(df_train[\'description\'].fillna(\'\'))\nprint(X_description_train.shape)\nX_description_test = wb.transform(df_test[\'description\'].fillna(\'\'))\nprint(X_description_test.shape)\nprint(\'-\')\ndel(wb)\ngc.collect()')


# In[16]:


get_ipython().run_cell_magic('time', '', "mask = np.where(X_description_train.getnnz(axis=0) > 8)[0]\nX_description_train = X_description_train[:, mask]\nprint(X_description_train.shape)\nX_description_test = X_description_test[:, mask]\nprint(X_description_test.shape)\nprint('[{}] Vectorize `description` completed.'.format(time.time() - start_time))")


# In[17]:


X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_description_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

# Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train desc ridge (1)'.format(time.time() - start_time))
desc_ridge_preds1 = model.predict(X_train_2)
desc_ridge_preds1f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds2 = model.predict(X_train_1)
desc_ridge_preds2f = model.predict(X_description_test)
print('[{}] Finished to predict desc ridge (2)'.format(time.time() - start_time))
desc_ridge_preds_oof = np.concatenate((desc_ridge_preds2, desc_ridge_preds1), axis=0)
desc_ridge_preds_test = (desc_ridge_preds1f + desc_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(desc_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(desc_ridge_preds_test, y_test)))
gc.collect()


# In[18]:


del X_train_1
del X_train_2
del y_train_1
del y_train_2
del name_ridge_preds1
del name_ridge_preds1f
del name_ridge_preds2
del name_ridge_preds2f
del desc_ridge_preds1
del desc_ridge_preds1f
del desc_ridge_preds2
del desc_ridge_preds2f
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


# In[19]:


get_ipython().run_cell_magic('time', '', "df_train.drop(['deal_probability', 'title', 'description'], axis=1, inplace=True)\ndf_test.drop(['title', 'description'], axis=1, inplace=True)\nprint('Remerged')\n\ndummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1',\n            'region', 'city', 'param_1', 'param_2', 'param_3', 'activation_weekday']\nnumeric_cols = list(set(df_train.columns.values) - set(dummy_cols))\nprint(numeric_cols)")


# In[20]:


get_ipython().run_cell_magic('time', '', "from sklearn.preprocessing import StandardScaler\nfrom sklearn.base import BaseEstimator, TransformerMixin\n\n# https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler\nclass Scaler(BaseEstimator, TransformerMixin):\n    def __init__(self, columns, copy=True, with_mean=True, with_std=True):\n        self.scaler = StandardScaler(copy, with_mean, with_std)\n        self.columns = columns\n\n    def fit(self, X, y=None):\n        self.scaler.fit(X[self.columns], y)\n        return self\n\n    def transform(self, X, y=None, copy=None):\n        init_col_order = X.columns\n        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns, index=X.index)\n        X_not_scaled = X[list(set(init_col_order) - set(self.columns))]\n        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]\n\nprint('Scaler')\nscaler = Scaler(columns=numeric_cols)\ndf_train = scaler.fit_transform(df_train)\ndf_test = scaler.transform(df_test)")


# In[21]:


df_train.head()


# In[22]:


get_ipython().run_cell_magic('time', '', "from sklearn.preprocessing import LabelBinarizer\n\nsparse_merge_train = hstack((X_name_train, X_description_train)).tocsr()\nsparse_merge_test = hstack((X_name_test, X_description_test)).tocsr()\nprint(sparse_merge_train.shape)\nfor col in dummy_cols:\n    print(col)\n    lb = LabelBinarizer(sparse_output=True)\n    sparse_merge_train = hstack((sparse_merge_train, lb.fit_transform(df_train[[col]].fillna('')))).tocsr()\n    print(sparse_merge_train.shape)\n    sparse_merge_test = hstack((sparse_merge_test, lb.transform(df_test[[col]].fillna('')))).tocsr()")


# In[26]:


del X_description_test, X_name_test
del X_description_train, X_name_train
del lb, mask
gc.collect()


# In[27]:


get_ipython().run_cell_magic('time', '', 'print("\\n FM_FTRL Starting...........")\nif SUBMIT_MODE:\n    iters = 1\nelse:\n    iters = 1\n    rounds = 1\n\nmodel = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=sparse_merge_train.shape[1],\n                alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,\n                D_fm=100, e_noise=0, iters=iters, inv_link="identity", threads=4)\n\nif SUBMIT_MODE:\n    model.fit(sparse_merge_train, y_train)\n    print(\'[{}] Train FM completed\'.format(time.time() - start_time))\n    predsFM = model.predict(sparse_merge_test)\n    print(\'[{}] Predict FM completed\'.format(time.time() - start_time))\nelse:\n    for i in range(rounds):\n        model.fit(sparse_merge_train, y_train)\n        predsFM = model.predict(sparse_merge_test)\n        print(\'[{}] Iteration {}/{} -- RMSLE: {}\'.format(time.time() - start_time, i + 1, rounds, rmse(predsFM, y_test)))\n\ndel model\ngc.collect()\nif not SUBMIT_MODE:\n    print("FM_FTRL dev RMSLE:", rmse(predsFM, y_test))\n# 0.23046 in 1/3')


# In[28]:


X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(sparse_merge_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train ridge (1)'.format(time.time() - start_time))
ridge_preds1 = model.predict(X_train_2)
ridge_preds1f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train ridge (2)'.format(time.time() - start_time))
ridge_preds2 = model.predict(X_train_1)
ridge_preds2f = model.predict(sparse_merge_test)
print('[{}] Finished to predict ridge (2)'.format(time.time() - start_time))
ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(ridge_preds_test, y_test)))


# In[29]:


fselect = SelectKBest(f_regression, k=48000)
train_features = fselect.fit_transform(sparse_merge_train, y_train)
test_features = fselect.transform(sparse_merge_test)
print('[{}] Select best completed'.format(time.time() - start_time))


del sparse_merge_train
del sparse_merge_test
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))


# In[30]:


del ridge_preds1
del ridge_preds1f
del ridge_preds2
del ridge_preds2f
del X_train_1
del X_train_2
del y_train_1
del y_train_2
del model
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


# In[31]:


df_train['ridge'] = ridge_preds_oof
df_train['name_ridge'] = name_ridge_preds_oof
df_train['desc_ridge'] = desc_ridge_preds_oof
df_test['ridge'] = ridge_preds_test
df_test['name_ridge'] = name_ridge_preds_test
df_test['desc_ridge'] = desc_ridge_preds_test
print('[{}] Finished adding submodels'.format(time.time() - start_time))


# In[32]:


f_cats = ["region","city","parent_category_name","category_name","user_type","image_top_1"]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
df_train, df_test = target_encode.encode(df_train, df_test, y_train)
print('[{}] Finished target encoding'.format(time.time() - start_time))


# In[33]:


df_train.head()


# In[34]:


del ridge_preds_oof
del ridge_preds_test
gc.collect()
print('[{}] Finished garbage collection'.format(time.time() - start_time))


# In[35]:


cols = ['region_te', 'city_te', 'parent_category_name_te', 'category_name_te',
        'user_type_te', 'image_top_1_te', 'desc_ridge', 'name_ridge', 'ridge']
train_dummies = csr_matrix(df_train[cols].values)
print('[{}] Finished dummyizing model 1/5'.format(time.time() - start_time))
test_dummies = csr_matrix(df_test[cols].values)
print('[{}] Finished dummyizing model 2/5'.format(time.time() - start_time))
del df_train
del df_test
gc.collect()
print('[{}] Finished dummyizing model 3/5'.format(time.time() - start_time))
train_features = hstack((train_features, train_dummies)).tocsr()
print('[{}] Finished dummyizing model 4/5'.format(time.time() - start_time))
test_features = hstack((test_features, test_dummies)).tocsr()
print('[{}] Finished dummyizing model 5/5'.format(time.time() - start_time))


# In[36]:


d_train = lgb.Dataset(train_features, label=y_train)
del train_features
gc.collect()
if SUBMIT_MODE:
    watchlist = [d_train]
else:
    d_valid = lgb.Dataset(test_features, label=y_test)
    watchlist = [d_train, d_valid]

params = {
    'learning_rate': 0.15,
    'application': 'regression',
    'max_depth': 13,
    'num_leaves': 400,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.6,
    'nthread': 4,
    'lambda_l1': 10,
    'lambda_l2': 10
}
print('[{}] Finished compiling LGB'.format(time.time() - start_time))


# In[37]:


get_ipython().run_cell_magic('time', '', 'modelL = lgb.train(params,\n                  train_set=d_train,\n                  num_boost_round=400,\n                  valid_sets=watchlist,\n                  verbose_eval=50)\n\npredsL = modelL.predict(test_features)\npredsL[predsL < 0] = 0\npredsL[predsL > 1] = 1\n\nif not SUBMIT_MODE:\n    print("LGB RMSLE:", rmse(predsL, y_test))')


# In[38]:


del d_train
del modelL
if not SUBMIT_MODE:
    del d_valid
gc.collect()


# In[44]:


preds_final = predsFM * 0.1 + predsL * 0.9
if not SUBMIT_MODE:
    print('Final RMSE: ', rmse(preds_final, y_test))


# In[ ]:


if SUBMIT_MODE:
    submission['deal_probability'] = preds_final
    submission['deal_probability'] = submission['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
    submission.to_csv('lgb_and_fm_separate_train_test.csv', index=False)
    print('[{}] Writing submission done'.format(time.time() - start_time))
    print(submission.head(5))

