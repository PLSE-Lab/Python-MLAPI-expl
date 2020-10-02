import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
import xgboost as xgb

# stemmer = SnowballStemmer('english')
stemmer = PorterStemmer()

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

num_train = df_train.shape[0]

def str_stemmer(s):
    # only take into account those words which have length of greater than or equal to 4
	return " ".join([stemmer.stem(word) for word in s.lower().split() if len(word) > 4])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

# rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
# clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.475, gamma=.75, colsample_bytree=0.75, subsample=0.55, seed=44, min_child_weight=4)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('xgb.csv',index=False)

