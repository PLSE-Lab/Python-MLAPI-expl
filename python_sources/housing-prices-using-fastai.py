import pandas as pd
import numpy as np
from fastai.tabular import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.describe()

# displaying with missing values
def missing_values(data):
    # Missing values
  missing_values = data.isnull().sum().sort_values(ascending = False)
  percentage_missing_values = (missing_values/len(data))*100

  return pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])

missing_values(df_train)
df_test.head()

# creating new features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def add_features(all_data):
  all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["GrLivArea"]
  all_data["TotalSF_HQ"] = all_data["TotalBsmtSF"] + all_data["GrLivArea"] - all_data["LowQualFinSF"] - all_data["BsmtUnfSF"]
  all_data["TotalBath"] = all_data["BsmtFullBath"] + all_data["FullBath"] + (all_data["BsmtHalfBath"]/2) + (all_data["HalfBath"]/2)
  all_data["TotalPorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
  all_data["OverallRating"] = all_data["OverallQual"] * all_data["OverallCond"]
  all_data["HasPool"] = all_data["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
  all_data["Has2ndFloor"] = all_data["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
  all_data["HasGarage"] = all_data["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
  all_data["HasBsmt"] = all_data["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
  all_data["HasFireplace"] = all_data["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)
  return all_data
transformer = FunctionTransformer(add_features, validate=False)
df_new = transformer.fit_transform(df_train)
test_new = transformer.transform(df_test)
test_new.head()
numerical_cols, categorical_cols = cont_cat_split(df_new, max_card=30, dep_var="SalePrice")
numerical_cols.remove("Id")

# Dealing with the missing values
def fix_missing(df_train, df_test, categorical_cols, numerical_cols):

  for col in categorical_cols:
    df_train[col] = df_train[col].fillna("None")
    df_test[col] = df_test[col].fillna("None")
  
  for col in numerical_cols:
    median_train = df_train[col].median()
    median_test = df_test[col].median()

    df_train[col] = df_train[col].fillna(median_train)
    df_test[col] = df_test[col].fillna(median_test)
  
  return df_train, df_test

df_train, df_test = fix_missing(df_train, df_test, categorical_cols, numerical_cols)
missing_values(df_train[numerical_cols])
missing_values(df_test[numerical_cols])

## Using fastai for model and predictions
procs = [FillMissing, Categorify, Normalize]

dep_var = 'SalePrice'
df = df_train[categorical_cols + numerical_cols + [dep_var]].copy()
data = (TabularList.from_df(df, path=Path("."), cat_names=categorical_cols, cont_names=numerical_cols, procs=procs,)
                .split_by_rand_pct(0.1)
                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                .add_test(TabularList.from_df(df_test, path=Path("."), cat_names=categorical_cols, cont_names=numerical_cols))
                .databunch())


max_y = np.log(np.max(df_train['SalePrice'])*1.2)
y_range = torch.tensor([0, max_y], device=defaults.device)
model_housing = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=exp_rmspe)
model_housing.model
model_housing.lr_find()
model_housing.recorder.plot()
lr = 5e-2
model_housing.fit_one_cycle(5, lr, wd=0.2)
model_housing.unfreeze()
model_housing.fit_one_cycle(20, 5e-02)



# Submission
def create_submission(learn:Learner, name='model'):
    name = name + '_submission.csv'
    
    test_data = pd.read_csv('../input/test.csv')
    result = pd.DataFrame(columns=['Id', 'SalePrice'])
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    result['SalePrice'] = np.exp(preds.data).numpy().T[0]
    result['Id'] = test_data['Id'].T
    return result


sub = create_submission(model_housing)
def create_submission(learn:Learner, name='model'):
    name = name + '_submission.csv'
    
    test_data = pd.read_csv('test.csv')
    result = pd.DataFrame(columns=['Id', 'SalePrice'])
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    result['SalePrice'] = np.exp(preds.data).numpy().T[0]
    result['Id'] = test_data['Id'].T
    return result
sub = create_submission(model_housing)
sub.head()
sub.index += 1
sub.head()
sub.to_csv("submission.csv", index=False)