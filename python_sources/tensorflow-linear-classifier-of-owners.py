import tensorflow as tf
import pandas as pd
import tempfile
COLUMNS = ['HALTER_ID','ALTER','GESCHLECHT','STADTKREIS','STADTQUARTIER','RASSE1','RASSE1_MISCHLING','RASSE2',
           'RASSE2_MISCHLING','RASSENTYP','GEBURTSJAHR_HUND','GESCHLECHT_HUND','HUNDEFARBE']
CONTINUOUS_COLUMNS = ['AVG_AGE', 'GEBURTSJAHR_HUND']
#'STADTQUARTIER'
CATEGORICAL_COLUMNS = [ 'RASSE1',  'GESCHLECHT_HUND', 'HUNDEFARBE', 'STADTQUARTIER', 'STADTKREIS']
LABEL_COLUMN = 'GESCHLECHT'
def create_avg_age_col(df):
    # print('Creating Minage Maxage columns')
    df = df[pd.notnull(df['ALTER'])]
    # create age_min and age_max columns
    age_series = df['ALTER']
    avg_age_list = [];
    for index, value in age_series.iteritems():
        minage,maxage = value.split("-")
        avg_age_list.append(int(minage) + (int(maxage) - int(minage))/2)
    # add age_min and age_max columns
    return df.assign(AVG_AGE=avg_age_list)

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)], values=df[k].values,dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values, shape=[df[LABEL_COLUMN].size, 1])
    # Returns the feature columns and the label.
    return feature_cols, label
def build_estimator():

    dog_breed =  tf.contrib.layers.sparse_column_with_hash_bucket("RASSE1", hash_bucket_size=100)
    dog_gender =  tf.contrib.layers.sparse_column_with_hash_bucket("GESCHLECHT_HUND", hash_bucket_size=100)
    dog_color =  tf.contrib.layers.sparse_column_with_hash_bucket("HUNDEFARBE", hash_bucket_size=100)
    city_quarter = tf.contrib.layers.sparse_column_with_hash_bucket("STADTQUARTIER", hash_bucket_size=400)
    city_district = tf.contrib.layers.sparse_column_with_hash_bucket("STADTKREIS", hash_bucket_size=400)


    # city = tf.contrib.layers.real_valued_column("STADTKREIS")
    avg_age = tf.contrib.layers.real_valued_column("AVG_AGE")
    dog_byear = tf.contrib.layers.real_valued_column("GEBURTSJAHR_HUND")


    model_dir = tempfile.mkdtemp()
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,feature_columns=[city_district, city_quarter, dog_breed,dog_gender,
    dog_color,avg_age, dog_byear])
    return m
def convert_label_to_int(x):
    if(x == "w"):
        return 0
    else:
        return 1
def main():
    df = pd.read_csv('../input/20151001hundehalter.csv')
    df['STADTQUARTIER'] = df['STADTQUARTIER'].astype(str)
    df['STADTKREIS'] = df['STADTKREIS'].astype(str)
    # print(df['STADTKREIS'])
    df_train=df.sample(frac=0.8,random_state=200)
    df_test=df.drop(df_train.index)
    df_train[LABEL_COLUMN] =  (df_train[LABEL_COLUMN].apply(convert_label_to_int)).astype(int)
    df_test[LABEL_COLUMN] =  (df_test[LABEL_COLUMN].apply(convert_label_to_int)).astype(int)
    # print(df_train[LABEL_COLUMN])
    df_train = create_avg_age_col(df_train)
    df_test = create_avg_age_col(df_test)
    # # print(df_train['STADTKREIS'].dtype)
    df_train.dropna(subset=CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN])
    df_test.dropna(subset=CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN])
    # print(df_test)
    m = build_estimator()
    m.fit(input_fn=lambda: input_fn(df_train), steps=150)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print ("%s: %s" % (key, results[key]))
if __name__ == '__main__':
    main()
