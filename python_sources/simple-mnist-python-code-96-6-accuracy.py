"""
Created on Sun Mar 20 17:24:24 2016
Forked on Sat Apr 28 2018

@OriginalAuthor: vzocca
"""

# Creates a simple random forest benchmark
import pandas

train = pandas.read_csv("../input/train.csv")
test = pandas.read_csv("../input/test.csv")

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

#Berechnet für eine Menge an Lösungen (predictions) Trefferquote    
def accuracy(predictions):
    count = 0.0
    for i in range(len(predictions)):
        if predictions[i] == train["label"][i]:
            count = count + 1.0
            
    accuracy = count/len(predictions)
    print ("--- Accuracy value is " + str(accuracy))
    return accuracy

#Erzeugt eine Liste an Elementen mit den Spaltennamen der Traininsdaten
predictors = []
for i in range(784):
    string = "pixel" + str(i)
    predictors.append(string)


# Erzeugt einen Algo, der n_estimators (150) Entscheidungsbäume aus Teilmengen der Traininsdaten aufbaut

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=2, min_samples_leaf=1)

print ("Using "+ str(alg))
print

# Test des Algos auf cv (3) Teilmengen der Trainingsdaten mit dem Komplement dieser Teilmenge als Testmenge

# Compute the accuracy score for all the cross validation folds. 
scores = cross_validation.cross_val_score(alg, train[predictors], train["label"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print (scores)
print("Cross validation scores = " + str(scores.mean()))


full_predictions = []
# Algorithmus einlernen
alg.fit(train[predictors], train["label"])

# Berechnet Wahrscheinlichkeit und liefert eine 2D-Liste mit [Versuch-Index][Ziffern-Index], wobei die Zellen die berechnete Wahrscheinlichkeit enthalten
predictions = alg.predict_proba(train[predictors]).astype(float)

# Nimmt den Index jeder 10er Liste (Axe index 0), dessen Zelle den höchsten Wert hat.
# Da der Listenindex mit der Ziffer übereinstimmt enthält predictions nun die Vorhersage
predictions = predictions.argmax(axis=1)

submission = pandas.DataFrame({
        "true value": train["label"],
        "label": predictions
    })

# Genauigkeit von Training gegen Training
accuracyV = accuracy(predictions)

# Compute accuracy by comparing to the training data.
#accuracy = (sum(predictions[predictions == train["label"]])).astype(float) / len(predictions)
#print accuracy

filename = str('%0.5f' %accuracyV) + "_test_mnist.csv"
submission.to_csv(filename, index=False)


# Tatsächliche Wettbewerbsteilnahme

full_predictions = []
# Lernen (warum nochmal? Überflüssig oder EIgenschaft der scikit-library?)
alg.fit(train[predictors], train["label"])
# Berechnet Wahrscheinlichkeit und liefert eine 2D-Liste mit [Versuch-Index][Ziffern-Index], wobei die Zellen die berechnete Wahrscheinlichkeit enthalten
predictions = alg.predict_proba(test[predictors]).astype(float)
# Nimmt den Index jeder 10er Liste (Axe index 0), dessen Zelle den höchsten Wert hat.
# Da der Listenindex mit der Ziffer übereinstimmt enthält predictions nun die Vorhersage
predictions = predictions.argmax(axis=1)
ImageId = []
for i in range(1, 28001):
    ImageId.append(i)

#Ergebnis formatieren
submission = pandas.DataFrame({
        "ImageId": ImageId,
        "Label": predictions
    })

#Output erzeugen    
submission.to_csv("kaggle_mnist.csv", index=False)

# Score on kaggle mnist competition = 0.96614
print ("End of program")
