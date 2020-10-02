import pandas as pd
import catboost
# Load datasets
df = pd.read_csv("../input/train.csv")
df_tst = pd.read_csv("../input/test.csv")
# Fix missings
for d in df,df_tst:
    d['Cabin'].fillna('Unknown',inplace=True)
    d['Embarked'].fillna('Unknown',inplace=True)
# Select columns for X
cols = list(df.columns)
cols.remove('Survived')
cols.remove('PassengerId')
# Build model
cbc = catboost.CatBoostClassifier(random_seed=0).fit(df[cols].values,df['Survived'],cat_features=[0,1,2,6,8,9])
# Submit prediction
pd.DataFrame({'PassengerId':df_tst['PassengerId'],'Survived':cbc.predict(df_tst[cols].values).astype(int)}).to_csv('./demo_catboost.csv',index=False)