import numpy as np
import pandas as pd
from sklearn import preprocessing,svm,neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

df=pd.read_csv('../input/Iris.csv')
df.replace('?',-99999,inplace=True)
df.drop(['Id'], 1, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

#df = handle_non_numerical_data(df)
#print(df.head())


X = np.array(df.drop(['Species'], 1))
y = np.array(df['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

##example_measures = np.array([4,2,1,1,1,2,3,2,1])
##example_measures = example_measures.reshape(1, -1)
##prediction = clf.predict(example_measures)
##print(prediction)
##
##example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
##example_measures = example_measures.reshape(2, -1)
##prediction = clf.predict(example_measures)
##print(prediction)
##
##example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
##example_measures = example_measures.reshape(len(example_measures), -1)
##prediction = clf.predict(example_measures)
##print(prediction)
