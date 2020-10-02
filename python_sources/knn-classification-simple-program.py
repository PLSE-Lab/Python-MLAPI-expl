from utility_programs import *

# Import Datasets
filePath = '../input/irisdata/iris.csv'
col_names=['sepal length', 'sepal width', 'petal length', "petal width",'Species']
data = read_CSV(filePath, names=col_names)

# Define target and features columns
target = "Species"
features = [col for col in col_names if col != target]

# print first 5 rows data
print(data.head())


# define X, Y
X,y = divide_X_and_y(data,target)

# label encode the target variabled
encoded_data,_ = sklearn_label_encoding(data, target)
data[target] = encoded_data[target]
data[target].unique()

# divide train and test
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y)

# Normalise Features
X_train, scaler = sklearn_scaler(X_train)
X_test = scaler.transform(X_test)

# KNN Classifier
y_predict, knn_classifer = KNeighborsClassifier_Model(X_train, y_train.values.reshape(-1,), X_test)

# calculate accuracy
print("Accuracy:",calculate_accuracy(knn_classifer, X_test, y_test))

# print classification report
print(classification_report(y_test, y_predict))

