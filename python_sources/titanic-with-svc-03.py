# Libraries for preprocessing
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler

# Libraries for Machine Learning
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

# Load input data
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

# Sex
lb_enc_sex = LabelEncoder()
lb_sex = lb_enc_sex.fit_transform(df_train['Sex'])
oh_enc_sex = OneHotEncoder()
oh_enc_sex.fit(np.array(lb_sex).reshape(-1,1))

# Age
imp_age = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0, copy=True)
imp_age.fit(np.array(df_train['Age']).reshape(-1, 1))

# Fare
imp_fare = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0, copy=True)
imp_fare.fit(np.array(df_train['Fare']).reshape(-1, 1))

# Embarked
df_train['Embarked'] = df_train['Embarked'].fillna('U')
lb_enc_emb = LabelEncoder()
lb_emb = lb_enc_emb.fit_transform(df_train['Embarked'])
oh_enc_emb = OneHotEncoder()
oh_enc_emb.fit(np.array(lb_emb).reshape(-1,1))

def transform_data(df):
    # Pclass 
    oh_enc_pclass = OneHotEncoder()
    enc_pclass = oh_enc_pclass.fit_transform(np.array(df['Pclass']).reshape(-1, 1))
    df_pclass = DataFrame(enc_pclass.toarray(), columns=['1st', '2nd', '3rd'])
    
    # Sex
    lb_sex = lb_enc_sex.transform(df['Sex'])
    enc_sex = oh_enc_sex.transform(np.array(lb_sex).reshape(-1,1))
    df_sex = DataFrame(enc_sex.toarray(), columns=['female', 'male'])
    
    # Age
    age = imp_age.transform(np.array(df['Age']).reshape(-1, 1))
    df_age = DataFrame(age, columns=['Age'])

    # Fare
    fare = imp_fare.transform(np.array(df['Fare']).reshape(-1, 1))
    df_fare = DataFrame(fare, columns=['Fare'])
    
    df_nfamilies = df['Parch'] + df['SibSp']
    df_alone = df_nfamilies.copy()
    df_alone[df_nfamilies>0]=0
    df_alone[df_nfamilies==0]=1
    
    # Extract Salutation
    ser_mr = df.Name.apply(lambda x: contains(x, 'Mr.'))
    ser_mr.name = 'Mr.'
    ser_dr = df.Name.apply(lambda x: contains(x, 'Dr.'))
    ser_dr.name = 'Dr.'
    ser_rev = df.Name.apply(lambda x: contains(x, 'Rev.'))
    ser_rev.name = 'Rev.'
    ser_don =df.Name.apply(lambda x: contains(x, 'Don.'))
    ser_don.name = 'Don.'
    ser_dona = df.Name.apply(lambda x: contains(x, 'Dona.'))
    ser_dona.name = 'Dona'
    ser_sir = df.Name.apply(lambda x: contains(x, 'Sir.'))
    ser_sir.name = 'Sir.'
    ser_jonkheer = df.Name.apply(lambda x: contains(x, 'Jonkheer.'))
    ser_jonkheer.name = 'Jonkheer.'
    ser_master = df.Name.apply(lambda x: contains(x, 'Master'))
    ser_master.name = 'Master.'
    ser_major = df.Name.apply(lambda x: contains(x, 'Major'))
    ser_major.name = 'Major.'
    ser_col = df.Name.apply(lambda x: contains(x, 'Col.'))
    ser_col.name = 'Col.'
    ser_capt = df.Name.apply(lambda x: contains(x, 'Capt.'))
    ser_capt.name = 'Capt.'
    ser_mrs = df.Name.apply(lambda x: contains(x, 'Mrs'))
    ser_mrs.name = 'Mrs.'
    ser_mme = df.Name.apply(lambda x: contains(x, 'Mme'))
    ser_mme.name = 'Mme.'
    ser_ms = df.Name.apply(lambda x: contains(x, 'Ms'))
    ser_ms.name = 'Ms.'
    ser_mlle = df.Name.apply(lambda x: contains(x, 'Mlle'))
    ser_mlle.name = 'Mlle.'
    ser_miss = df.Name.apply(lambda x: contains(x, 'Miss'))
    ser_miss.name = 'Miss.'
    df_salutation = pd.concat([ser_mr, ser_dr, ser_rev, ser_don, ser_dona, ser_sir, ser_jonkheer, ser_master, ser_major, ser_col, ser_mrs, ser_mme, ser_ms, ser_mlle, ser_miss], axis=1)
        
    # Embarked
    lb_emb = lb_enc_emb.transform(df['Embarked'])
    enc_emb = oh_enc_emb.transform(np.array(lb_emb).reshape(-1,1))
    df_emb = DataFrame(enc_emb.toarray(), columns=['C', 'Q', 'S', 'U'])
    return pd.concat([df_fare, df_age, df_sex, df_nfamilies], axis=1), pd.concat([df['Pclass'], df_alone,df_emb, df_salutation],axis=1)
    
# This method is used in transform_data method to extract salutation
def contains(array, word):
    if word in array:
        return 1
    else:
        return 0

df_X_v, df_X_f = transform_data(df_train)
df_y = df_train['Survived']

df_X_v_test, df_X_f_test = transform_data(df_test)

sc = StandardScaler()
# StandardScaler is applied for continuous variables, not for discrete(categorical) variables
df_X_v = DataFrame(sc.fit_transform(df_X_v))
df_X = pd.concat([df_X_v, df_X_f], axis=1)

df_X_v_test = DataFrame(sc.fit_transform(df_X_v_test))
df_X_test = pd.concat([df_X_v_test, df_X_f_test], axis=1)

# Evaluate the performance
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='rbf', C=2, gamma=0.075), df_X, df_y, train_sizes=[50, 100, 200, 400, 712], cv=5)
plt.plot([50, 100, 200, 400, 800], train_scores.mean(axis=1), label='train')
plt.plot([50, 100, 200, 400, 800], valid_scores.mean(axis=1), label='valid')
plt.legend()
plt.show()
print(valid_scores.mean(axis=1))

# Train final model using all the training data   
model = SVC(kernel='rbf', C=2, gamma=0.075)
model.fit(np.array(df_X), np.array(df_y))

result = np.array(model.predict(df_X_test))
df_result = DataFrame(result, columns=['Survived'])
df_result = pd.concat([df_test['PassengerId'], df_result], axis = 1)

df_result['Survived'] = np.array(round(df_result['Survived']), dtype='int')

# Output the result csv file
df_result.to_csv('reult_svc_titanic_20180728_08.csv', index=False)
    
