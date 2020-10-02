import csv
import operator
import random
import math


forest = {}

def bucket(number):
    if not number:
        return 200.0
    return float(number) / 5

class Tree(object):
    def __init__(self, fieldname):
        self.fieldname = fieldname
        self.values = dict()

    def add(self, value, label):
        if value in self.values:
            if label in self.values[value]:
                self.values[value][label] += 1
            else:
                self.values[value][label] = 1
        else:
            self.values[value] = dict({label: 1})

    def most_likely_classes(self, value):
        if value not in self.values:
            return []
        classes = self.values[value]
        return list(reversed(sorted(classes.items(),
                key=operator.itemgetter(1))))

fields = ['passengerid', 'pclass', 'name', 'sex', 'age',
        'siblings', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
extrafields = ['class']

for f in fields + extrafields:
    forest[f] = Tree(f)

def get_class(cabin):
    for char in 'ABCDEFGH':
        if char in cabin:
            return char
    return 'U'


header = False
for row in csv.reader(open('../input/train.csv', 'r')):
    if not header:
        header = True
        continue
    (passengerid, survived, pclass, name, sex, age, siblings, parch, ticket,
            fare, cabin, embarked) = row
    forest['passengerid'].add(passengerid, survived)
    forest['pclass'].add(pclass, survived)
    forest['name'].add(name, survived)
    forest['sex'].add(sex, survived)
    forest['age'].add(bucket(age), survived)
    forest['siblings'].add(siblings, survived)
    forest['parch'].add(parch, survived)
    forest['ticket'].add(ticket, survived)
    forest['fare'].add(fare, survived)
    forest['cabin'].add(cabin, survived)
    forest['embarked'].add(embarked, survived)
    forest['class'].add(get_class(cabin), survived)


writer = open('output.csv', 'w')
writer.write('PassengerId,Survived\n')

header = False
for row in csv.reader(open('../input/test.csv', 'r')):
    if not header:
        header = True
        continue
    (passengerid, pclass, name, sex, age, siblings, parch, ticket,
            fare, cabin, embarked) = row
    all_probs = [forest['passengerid'].most_likely_classes(passengerid),
            forest['pclass'].most_likely_classes(pclass),
            forest['name'].most_likely_classes(name),
            forest['sex'].most_likely_classes(sex),
            forest['age'].most_likely_classes(bucket(age)),
            forest['siblings'].most_likely_classes(siblings),
            forest['parch'].most_likely_classes(parch),
            forest['ticket'].most_likely_classes(ticket),
            forest['fare'].most_likely_classes(fare),
            forest['cabin'].most_likely_classes(cabin),
            forest['embarked'].most_likely_classes(embarked),
            forest['class'].most_likely_classes(get_class(cabin))]
    classprobs = dict()
    for problist in all_probs:
        if not problist:
            continue
        for (xclass, count) in problist:
            if xclass in classprobs:
                classprobs[xclass] += count
            else:
                classprobs[xclass] = count
    prediction = list(reversed(sorted(classprobs.items(), key=operator.itemgetter(1))))[0]
    writer.write("%s,%s\n" % (passengerid, prediction[0]))
    