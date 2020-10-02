import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from fancyimpute import KNN
from numpy import genfromtxt
import timeit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
import datetime
import itertools
import pandas

from keras import optimizers

# Getting data from *.csv file
raw_data = genfromtxt("../input/hospital-readmissions-orig.csv", delimiter=',', skip_header=1)

# Completting data KNN method in case of missing data
complete_data = raw_data
#complete_data = KNN(k=3).fit_transform(raw_data)

#Subseting a smaller dataset 
num_rows_2_sample = 200
subset = complete_data[numpy.random.choice(complete_data.shape[0], num_rows_2_sample, replace=False)]
# Selecting columns with data
data = subset[:,0:17]
# Getting number of columns in variable c
c = numpy.size(data, 1)


# Selecting columns with target
target = subset[:,17:18]
#target = target.astype(int)



# Getting date and time
now = datetime.datetime.now()
print(str(now))

# Creating Log file
log_file = "_test_log.txt"
log_file = str(now) + log_file
log_file = log_file.replace(" ","_")
log_file = log_file.replace(":","-")
f = open(log_file, "w+")
f.close()

f = open(log_file, "w+")
f.close()


# Printing and writing time and date at the file
f = open(log_file, "a+")
f.write(str(now))
f.write("\n")
f.close()

def create_model():
    # Create model
    model = Sequential()
    model.add(Dense(12, input_dim=d, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def cv(clf, data, cv):
    # Getting starting time
    start_time = start = timeit.default_timer()

    # Cross validation scores
    scores = cross_val_score(clf, data, target, cv=cv)

    # Getting finishing time
    finish_time = timeit.default_timer()

    # Printing number of folds results of cross validation and computing time
    folds = "k-fold number: " + str(cv)
    print(folds)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("Time: ", finish_time - start_time, "sec")
    time = "Time: " + str(finish_time - start_time) + "sec"

    # Adding data at the file
    f = open(log_file, "a+")
    f.write(folds)
    f.write("\n")
    f.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    f.write("\n")
    f.write(time)
    f.write("\n")
    f.close()
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Artificial Neural Net", "AdaBoost",
         "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs = -1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    model,
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]


f = open(log_file, "a+")
d = numpy.size(data, 1)
for i in range(len(classifiers)):
    print(names[i])
    f = open(log_file, "a+")
    f.write(names[i])
    f.write("\n")
    f.close()
    cv(classifiers[i], data, 10)