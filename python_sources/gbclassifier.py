import numpy as np
import pandas as pd
from sklearn import preprocessing, ensemble

# columns to be used as features #
FEATURES = ["ncodpers","ind_empleado","pais_residencia","sexo","age", "ind_nuevo", "antiguedad", "nomprov", "segmento"]

TARGETS = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

if __name__ == "__main__":
        #data_path = "../input/"
        train_file = "../input/train_ver2.csv"
        test_file = "../input/test_ver2.csv"

        df_train = pd.read_csv(train_file, usecols=FEATURES)
        df_test = pd.read_csv(test_file, usecols=FEATURES)

        df_train = df_train.drop_duplicates(['ncodpers'], keep='last')
        #df_test = df_test.drop_duplicates(['ncodpers'], keep='last')

        df_train.fillna(-99, inplace=True)
        #df_train['age'] = df_train['age'].convert_objects(convert_numeric=True)
        #df_train['antiguedad']=df_train['antiguedad'].convert_objects(convert_numeric=True)
        df_test.fillna(-99, inplace=True)
        for ind, col in enumerate(FEATURES):
            if col!='ncodpers':
                print(col)
                if df_train[col].dtype == "object":
                        le = preprocessing.LabelEncoder()
                        le.fit(list(df_train[col].values) + list(df_test[col].values))
                        df_train[col] = le.transform(list(df_train[col].values)).reshape(-1,1)
                        df_test[col] = le.transform(list(df_test[col].values)).reshape(-1,1)
                else:
                        df_train[col] = np.array(df_train[col]).reshape(-1,1)
                        df_test[col] = np.array(df_test[col]).reshape(-1,1)
        print('train samples size %d,%d, test samples size %d,%d' % (df_train.shape[0],df_train.shape[1], df_test.shape[0],df_test.shape[1]))

        train_y = pd.read_csv(train_file, usecols=TARGETS, dtype='float16')
        train_y = train_y.reindex(index=df_train.index)
        train_y = np.array(train_y.fillna(0)).astype('int')

        train_X = df_train.values[:,1:]
        test_X = df_test.values[:,1:]

        print(train_X.shape, train_y.shape)
        print(test_X.shape)

        print("Running Model..")
        model = ensemble.ExtraTreesClassifier(min_samples_split=5,random_state=2016)
        model.fit(train_X, train_y)
        del train_X, train_y
        print("Predicting..")
        #print(model.predict_proba(test_X))
        preds = np.array(model.predict_proba(test_X))[:,:,1].T
        #print(preds.shape)
        del test_X

        print("Creating submission..")
        preds = np.argsort(preds, axis=1)
        preds = np.fliplr(preds)[:,:7]
        test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
        target_cols = np.array(TARGETS)
        preds = [" ".join(list(target_cols[pred])) for pred in preds]
        out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':preds})
        out_df.to_csv('sub_gb.csv', index=False)
