import pandas as pd
import numpy as np
from collections import Counter
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import difflib
import csv
from functools import partial


df=pd.read_csv("../input/train-dataset/train.tsv", sep='\t', lineterminator='\n')
df=df[  ( df.brand_name.notnull() ) & ( df.name.notnull() )  &  ( df.category_name.notnull()  )  &  (df.price!=0) & (df.brand_name=='Nike')
]

#df=df[  ( df.brand_name.notnull() ) & ( df.name.notnull() )  &  ( df.category_name.notnull()  )  &  (df.price!=0) ]
df['item_description']=df['name'].str.upper()+" "+df['item_description'].str.upper()
df.sort_values(by='train_id', ascending=True)


df['category_name'].fillna("NO_CAT",inplace=True)
df['brand_name'].fillna("NO_BRAND",inplace=True)
df['name'].fillna("NO_VALUE",inplace=True)
df['item_description'].fillna("",inplace=True)
df['shipping'].fillna(0,inplace=True)
df['item_condition_id'].fillna(5,inplace=True)


df2=df
#df2=df[ ( df.category_name=='NO_CAT' ) ]

df2[['item_description']]=df2.item_description.apply(lambda i: ' '.join(filter(lambda j: len(j) >= 6, i.split())))


#############################
df2_X=df2.groupby(['name'])['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(6))
df2=df2.join( df2_X ,on=('name'), rsuffix='_X' )
df2['CX']=""
df2['CX'] = df2["item_description_X"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2['CX'].fillna('',inplace=True)
print("df2 list")
print(df2.index)
#print(df2)
#############################


group_val=['category_name','brand_name','shipping','item_condition_id']
df1_2_a=df2.join( df2.groupby(group_val) ['price'].quantile(1) ,on=(group_val), rsuffix='_MAX' )
df1_2_b=df1_2_a.join( df2.groupby(group_val) ['price'].quantile(0.95) ,on=(group_val), rsuffix='_95' )
df1_2_c=df1_2_b.join( df2.groupby(group_val) ['price'].quantile(0.85) ,on=(group_val), rsuffix='_85' )
df1_2_d=df1_2_c.join( df2.groupby(group_val) ['price'].quantile(0.5) ,on=(group_val), rsuffix='_50' )
df1_2_e=df1_2_d.join( df2.groupby(group_val) ['price'].quantile(0.25) ,on=(group_val), rsuffix='_25' )
df1_2  =df1_2_e.join( df2.groupby(group_val) ['price'].quantile(0) ,on=(group_val), rsuffix='_00' )
#print(df1_2)


df2=df1_2
print("df2 list 2")
print(df2.index)

del df1_2
param_val=100


new_column='C_MAX'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price == df2.price_MAX) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)


'''
new_column='C_95'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price < df2.price_MAX) &  (df2.price >= df2.price_95) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)
'''

new_column='C_85'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price < df2.price_MAX) & (df2.price >= df2.price_85) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)



new_column='C_50'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price < df2.price_MAX)  & (df2.price >= df2.price_50) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)


new_column='C_25'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price < df2.price_MAX)  & (df2.price >= df2.price_25) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)


new_column='C_00'
group_val_tmp=['category_name','brand_name','shipping','item_condition_id',new_column]

df2_1X=df2[ (df2.price < df2.price_MAX)  & (df2.price >= df2.price_00) ]
df2_1=df2_1X.join( df2_1X.groupby(group_val)['item_description'].apply(lambda x: Counter((' '.join(x)).split(" ")).most_common(param_val)) ,
on=(group_val), rsuffix='_MAX_2' )
df2_1[new_column] = df2_1["item_description_MAX_2"].apply(lambda x : [y[0] for y in list(x)]).astype(str)
df2_1=df2_1[group_val_tmp].drop_duplicates()
df2=pd.merge(df2, df2_1[group_val_tmp], on =group_val,suffixes=('', '_W2_2'), how='outer')
df2[new_column].fillna('',inplace=True)


print("df2 list 3")
print(df2.index)



def apply_sm(s, c1, c2): 
    return difflib.SequenceMatcher(None, s[c1], s[c2]).ratio()

df2['X_MAX_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_MAX'), axis=1)
#df2['X_95_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_95'), axis=1)
df2['X_85_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_85'), axis=1)
df2['X_50_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_50'), axis=1)
df2['X_25_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_25'), axis=1)
df2['X_00_comp'] =df2.apply(partial(apply_sm, c1='CX', c2='C_00'), axis=1)



df5=df2




var_mod = ['category_name']
le = LabelEncoder()
for i in var_mod:
    df5[i] = le.fit_transform(df5[i])
    df5.dtypes 

var_mod = ['brand_name']
le = LabelEncoder()
for i in var_mod:
    df5[i] = le.fit_transform(df5[i])
    df5.dtypes 



print("DELETE COMPLETED !!")
print("ACC CALCULATION !!")
print(df5.index)


'''
df_test=pd.read_csv("../input/test2-data/test.tsv", sep='\t', lineterminator='\n')
#df=df[  ( df.brand_name.notnull() )   &  ( df.category_name.notnull()  ) ]
df_test['item_description']=df_test['name'].str.upper()+" "+df_test['item_description'].str.upper()
df_test.sort_values(by='test_id', ascending=True)
df_test['category_name'].fillna("NO_CAT",inplace=True)
df_test['brand_name'].fillna("NO_BRAND",inplace=True)
df_test['name'].fillna("NO_VALUE",inplace=True)
df_test['item_description'].fillna("",inplace=True)
df_test['shipping'].fillna(0,inplace=True)
df_test['item_condition_id'].fillna(5,inplace=True)
df_test=df_test.head(1000)


var_mod = ['category_name']
le = LabelEncoder()
for i in var_mod:
    df_test[i] = le.fit_transform(df_test[i])
    df_test.dtypes 

var_mod = ['brand_name']
le = LabelEncoder()
for i in var_mod:
    df_test[i] = le.fit_transform(df_test[i])
    df_test.dtypes 
'''


predictor_var = df5[["category_name","brand_name","shipping","item_condition_id",
"X_MAX_comp",
#"X_95_comp",
#"X_85_comp",
"X_50_comp",
#"X_25_comp",
"X_00_comp",

"price_MAX",
#"price_95",
#"price_85",
"price_50",
#"price_25",
"price_00"
]]


'''
predictor_var2 = df_test[["category_name","brand_name","shipping","item_condition_id"
#"X_Y_comp",
#"price_MAX","price_85","price_70","price_55","price_40","price_25","price_10","price_00"
]]

'''

outcome_var = df5['price']
model = RandomForestRegressor(n_estimators=50,oob_score = "TRUE")
model.fit(predictor_var,outcome_var)
predictions = model.predict(predictor_var)
print(predictions.round())
accuracy = metrics.accuracy_score(predictions.round(),outcome_var.round())


print("Accuracy : %s" % "{0:.3%}".format(accuracy) )

#df2=df2[(df2.train_id==103357)]
my_submission = pd.DataFrame({'Id': df2.train_id,
"price_MAX": df5.price_MAX,
#"price_95": df5.price_95,
#"price_85": df5.price_85,
"price_50": df5.price_50,
#"price_25":df5.price_25,
"price_00": df5.price_00,
"X_MAX_comp":df5.X_MAX_comp,
#"X_95_comp": df5.X_95_comp,
#"X_85_comp": df5.X_85_comp, 
"X_50_comp":df5.X_50_comp,
#"X_25_comp":df5.X_25_comp,
"X_00_comp":df5.X_00_comp,
'Price': df2.price ,'EstimatedPrice': predictions.round()})
# you could use any filename. We choose submission here
my_submission.to_csv('ozukun_out_4.csv', index=False)
print("Writing complete")