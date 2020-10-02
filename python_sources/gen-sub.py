import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.externals import joblib

train_df = pd.read_csv('../input/train_file.csv')
test_df = pd.read_csv('../input/test_file.csv')


def label_encode(train_df, test_df):
    le_sex = LabelEncoder()
    le_st = LabelEncoder()
    le_qc = LabelEncoder()
    
    train_df.Sex = train_df.Sex.apply(lambda x: x.lower())
    test_df.Sex = test_df.Sex.apply(lambda x: x.lower())
    
    le_sex.fit(train_df.Sex)
    le_sex.fit(test_df.Sex)
    train_df.Sex = le_sex.transform(train_df.Sex)
    test_df.Sex = le_sex.transform(test_df.Sex)
    
    le_st.fit(train_df.StratificationType)
    train_df.StratificationType = le_st.transform(train_df.StratificationType)
    test_df.StratificationType = le_st.transform(test_df.StratificationType)
    
    le_qc.fit(train_df.QuestionCode)
    train_df.QuestionCode = le_qc.transform(train_df.QuestionCode)
    test_df.QuestionCode = le_qc.transform(test_df.QuestionCode)
    
    return train_df, test_df
    
train_df, test_df = label_encode(train_df, test_df)
    
def process_geo_coordinates(df):
    geo_lat = train_df.GeoLocation.apply(lambda x : float(str(x)[1:-1].split(',')[0]) if len(str(x)[1:-1].split(',')) > 1 else np.nan)
    geo_lon = train_df.GeoLocation.apply(lambda x : float(str(x)[1:-1].split(',')[1]) if len(str(x)[1:-1].split(',')) > 1 else np.nan)
    return geo_lat, geo_lon
    
train_df['geo_lat'], train_df['geo_lon'] = process_geo_coordinates(train_df)
test_df['geo_lat'], test_df['geo_lon'] = process_geo_coordinates(test_df)
train_df = train_df.drop('GeoLocation', axis=1)
test_df = test_df.drop('GeoLocation', axis=1)


def generate_extra_feature(df):
    question_has_marijuana = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('marijuana') !=-1 else 0)
    df['question_has_marijuana'] = question_has_marijuana
    question_has_alcohol = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('alcohol') !=-1 else 0)
    df['question_has_alcohol'] = question_has_alcohol
    question_has_heroin = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('heroin') !=-1 else 0)
    df['question_has_heroin'] = question_has_heroin
    injected_something = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('inject') !=-1 else 0) 
    df['injected_something'] = injected_something
    inhaled_something = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('inhalants') !=-1 else 0) 
    df['inhaled_something'] = inhaled_something
    used_cocaine = df.Greater_Risk_Question.apply(lambda x: 1 if x.lower().find('cocaine') !=-1 else 0)
    df['used_cocaine'] = used_cocaine
    is_hispanic = df.Race.apply(lambda x: 1 if x.lower().find('hispanic') !=-1 or x.lower().find('lation') !=-1 else 0)
    df['is_hispanic'] = is_hispanic
    is_black = df.Race.apply(lambda x: 1 if x.lower().find('black') !=-1 or x.lower().find('african') !=-1 else 0)
    df['is_black'] = is_black
    is_asian = df.Race.apply(lambda x: 1 if x.lower().find('asian') !=-1 else 0)
    df['is_asian'] = is_asian
    is_white = df.Race.apply(lambda x: 1 if x.lower().find('white') !=-1 else 0)
    df['is_white'] = is_white
    is_native = df.Race.apply(lambda x: 1 if x.lower().find('native') !=-1 else 0)
    df['is_native'] = is_native
    
    return df
    
    
train_df = generate_extra_feature(train_df)
test_df = generate_extra_feature(test_df)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
x_train = train_df.drop('LocationDesc', axis=1).drop('Greater_Risk_Question', axis=1) \
            .drop('Description', axis=1).drop('Race', axis=1).drop('Greater_Risk_Probability', axis=1)\
            .drop('Patient_ID', axis=1).values
            
y_train = train_df.Greater_Risk_Probability.values
y_train = y_train/100

x_test = test_df.drop('LocationDesc', axis=1).drop('Greater_Risk_Question', axis=1) \
            .drop('Description', axis=1).drop('Race', axis=1) \
            .drop('Patient_ID', axis=1).values
            
            
from sklearn.neural_network import MLPRegressor
nn1 = MLPRegressor(hidden_layer_sizes=(5,10,10,5), verbose=True, max_iter=50, activation='logistic', early_stopping=True)
nn1.fit(X=x_train, y=y_train,)

with open('model.pkl', 'wb') as model_file:
  pickle.dump(nn1, model_file,  protocol = 2)


preds_final = nn1.predict(x_test)
print(x_test[4])
result = pd.DataFrame(list(zip(test_df.Patient_ID, preds_final*100)), columns=['Patient_ID', 'Greater_Risk_Probability'])
result.to_csv('submission_mlp.csv', index=None,)

# from lightgbm import LGBMRegressor

# def rmsle(predictions, targets):
#     predictions = np.exp(predictions) - 1
#     targets = np.exp(targets) - 1
#     return np.sqrt(((predictions - targets) ** 2).mean())

# def rmsle_lgb(labels, preds):
#     return 'rmsle', rmsle(preds, labels), False

# # lgbm_params = {'n_estimators': 50, 'learning_rate': 0.6, 'max_depth': 12, 'n_jobs': 8 }
# lgbm_params = {'n_estimators': 100, 'learning_rate': 0.3, 'max_depth': 6,
#                'subsample': 0.9, 'colsample_bytree': 0.8,
#                'min_child_samples': 32, 'n_jobs': 8}
               
               
# model = LGBMRegressor(**lgbm_params)
# model.fit(x_train, y_train,
#          eval_metric=rmsle_lgb,
#          verbose=True)
         
# preds_final = model.predict(x_test)
# result = pd.DataFrame(list(zip(test_df.Patient_ID, preds_final*100)), columns=['Patient_ID', 'Greater_Risk_Probability'])
# result.to_csv('submission_lgb.csv', index=None,)
    